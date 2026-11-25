use super::*;
use std::time::{Duration, Instant};

// SplitMix64, fixed integer widths and little-endian instruction encoding.
// Each case owns a seed, so replay never depends on earlier cases.
struct Random(u64);
impl Random {
    fn word(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 32) as u32
    }
    fn small(&mut self, n: u8) -> u8 {
        (self.word() % u32::from(n)) as u8
    }
    fn alu(&mut self) -> Alu {
        match self.small(4) {
            0 => Alu::Set(self.small(65)),
            1 => Alu::Add(self.small(65)),
            2 => Alu::AddData,
            _ => Alu::XorData,
        }
    }
    fn region(&mut self, allow_loop: bool) -> Region {
        match self.small(if allow_loop { 5 } else { 4 }) {
            0 => Region::Plain(self.alu()),
            1 => Region::If {
                yes: self.alu(),
                no: self.alu(),
            },
            2 | 3 => {
                let mask = match self.small(3) {
                    0 => Mask::Input,
                    1 => Mask::Compare(self.small(65)),
                    _ => Mask::Joined(self.small(65)),
                };
                Region::Masked {
                    mask,
                    skip: self.small(2) == 1,
                    body: (0..1 + self.small(4)).map(|_| self.alu()).collect(),
                }
            }
            _ => Region::Repeat {
                times: 1 + self.small(6),
                body: (0..1 + self.small(3)).map(|_| self.region(false)).collect(),
            },
        }
    }
}

fn generate(seed: u64) -> (Vec<Region>, Inputs) {
    let mut r = Random(seed);
    let mask = match r.small(8) {
        0 => 0,
        1 => u32::MAX,
        2 => 1u32 << r.small(32),
        3 => 0xaaaa_aaaa,
        _ => r.word(),
    };
    let inputs = Inputs {
        mask,
        selector: r.small(2).into(),
        v: std::array::from_fn(|_| {
            std::array::from_fn(|_| match r.small(4) {
                0 => u32::MAX,
                1 | 2 => r.small(65).into(),
                _ => r.word(),
            })
        }),
    };
    (
        (0..1 + r.small(6)).map(|_| r.region(true)).collect(),
        inputs,
    )
}

// Each candidate is assembled afresh: labels and restores survive deletion.
// Walk subregions too, retaining nonempty bodies and positive trip counts.
fn smaller(body: &[Region]) -> Vec<Vec<Region>> {
    let mut candidates = Vec::new();
    for (i, r) in body.iter().enumerate() {
        let mut removed = body.to_vec();
        removed.remove(i);
        candidates.push(removed);
        match r {
            Region::Repeat { times, body: inner } => {
                for shorter in smaller(inner).into_iter().filter(|v| !v.is_empty()) {
                    let mut v = body.to_vec();
                    v[i] = Region::Repeat {
                        times: *times,
                        body: shorter,
                    };
                    candidates.push(v);
                }
                if *times > 1 {
                    let mut v = body.to_vec();
                    v[i] = Region::Repeat {
                        times: times - 1,
                        body: inner.clone(),
                    };
                    candidates.push(v);
                }
            }
            Region::Masked {
                mask,
                skip,
                body: inner,
            } if inner.len() > 1 => {
                for k in 0..inner.len() {
                    let mut shorter = inner.clone();
                    shorter.remove(k);
                    let mut v = body.to_vec();
                    v[i] = Region::Masked {
                        mask: *mask,
                        skip: *skip,
                        body: shorter,
                    };
                    candidates.push(v);
                }
            }
            _ => {}
        }
    }
    candidates
}

fn reduce(body: &[Region], inputs: &Inputs, witness: &Outcome, perturb: Perturb) -> Vec<Region> {
    assert!(matches!(witness, Outcome::Difference { .. }));
    let mut best = body.to_vec();
    while let Some(next) = smaller(&best)
        .into_iter()
        .find(|candidate| evaluate(candidate, inputs, perturb).outcome == *witness)
    {
        best = next;
    }
    best
}

#[test]
fn reducer_retains_the_guard_disagreement() {
    let active = Region::Masked {
        mask: Mask::Input,
        skip: false,
        body: vec![Alu::Add(0), Alu::Set(17)],
    };
    let body = [
        Region::Plain(Alu::Add(0)),
        active,
        Region::Plain(Alu::Add(0)),
    ];
    let i = fixture();
    let wanted = Outcome::Difference {
        stage: "structured",
        lane: 1,
        expected: 1,
        actual: 17,
    };
    assert_eq!(evaluate(&body, &i, drop_guard).outcome, Outcome::Agree); // dropped guard on add-zero
    // Target the observable writer; pruning an earlier op cannot retarget this mutation.
    let fault: Perturb = |p, s| {
        for (b, block) in p.blocks.iter().enumerate() {
            for (j, inst) in block.insts.iter().enumerate() {
                if inst.name() == "v_mov_b32" && inst.write(ir::Reg::V(0)).is_some() {
                    s.guards[b][j] = None;
                }
            }
        }
    };
    assert_eq!(evaluate(&body, &i, fault).outcome, wanted);
    let reduced = reduce(&body, &i, &wanted, fault);
    assert_eq!(
        reduced,
        vec![Region::Masked {
            mask: Mask::Input,
            skip: false,
            body: vec![Alu::Set(17)]
        }]
    );
    assert_eq!(source::program(&body).len(), 8);
    assert_eq!(source::program(&reduced).len(), 5);
    assert_eq!(evaluate(&reduced, &i, fault).outcome, wanted);
    assert_eq!(evaluate(&reduced, &i, unchanged).outcome, Outcome::Agree);
}

#[derive(Default, Debug)]
struct Counts {
    attempted: usize,
    ssa_agreed: usize,
    both_agreed: usize,
    refused: usize,
    unsupported: usize,
    invalid: usize,
    compiler_error: usize,
    limit: usize,
    crash: usize,
    differences: usize,
}
impl Counts {
    fn record(&mut self, r: &Comparison) {
        self.attempted += 1;
        self.ssa_agreed += usize::from(r.ssa_agreed);
        match r.outcome {
            Outcome::Agree => self.both_agreed += 1,
            Outcome::Refused(..) => self.refused += 1,
            Outcome::Stop(_, Stop::Unsupported(_)) => self.unsupported += 1,
            Outcome::Stop(_, Stop::Invalid(_)) => self.invalid += 1,
            Outcome::Stop(_, Stop::Limit) => self.limit += 1,
            Outcome::CompilerError(..) => self.compiler_error += 1,
            Outcome::Crash(..) => self.crash += 1,
            Outcome::Difference { .. } => self.differences += 1,
        }
    }
}
#[test]
fn inconclusive_outcomes_do_not_count_as_agreements() {
    let mut c = Counts::default();
    for o in [
        Outcome::Refused("build", 0, "test"),
        Outcome::Stop("source", Stop::Limit),
        Outcome::Stop("structured", Stop::Unsupported("test")),
        Outcome::Stop("ssa", Stop::Invalid("test")),
        Outcome::Crash("ssa", "test".into()),
        Outcome::CompilerError("build", "test".into()),
    ] {
        c.record(&Comparison {
            ssa_agreed: false,
            outcome: o,
        });
    }
    assert_eq!(
        (
            c.attempted,
            c.refused,
            c.limit,
            c.unsupported,
            c.invalid,
            c.crash,
            c.compiler_error
        ),
        (6, 1, 1, 1, 1, 1, 1)
    );
    assert_eq!((c.ssa_agreed, c.both_agreed, c.differences), (0, 0, 0));
}

fn option(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .map(|s| s.parse().expect("expected unsigned decimal integer"))
        .unwrap_or(default)
}

#[test]
#[ignore = "bounded generated campaign; use WAVE_SEED and WAVE_COUNT for replay"]
fn campaign() {
    let start = Instant::now();
    let first = option("WAVE_SEED", 0);
    let count = option("WAVE_COUNT", 10_000).min(10_000);
    let mut totals = Counts::default();
    for n in 0..count {
        if start.elapsed() >= Duration::from_secs(1800) {
            break;
        }
        let seed = first.wrapping_add(n);
        let (body, i) = generate(seed);
        let result = evaluate(&body, &i, unchanged);
        totals.record(&result);
        if result.outcome != Outcome::Agree {
            eprintln!("seed={seed} outcome={:?}", result.outcome);
            if matches!(result.outcome, Outcome::Difference { .. }) {
                eprintln!(
                    "reduced={:?}",
                    reduce(&body, &i, &result.outcome, unchanged)
                );
            }
        }
        if matches!(
            result.outcome,
            Outcome::Difference { .. }
                | Outcome::Crash(..)
                | Outcome::CompilerError(..)
                | Outcome::Stop(_, Stop::Invalid(_))
        ) {
            break; // inspect the first candidate before spending the remaining budget
        }
    }
    println!(
        "first_seed={first} elapsed={:?} {totals:?}",
        start.elapsed()
    );
    assert_eq!(
        totals.differences + totals.crash + totals.compiler_error + totals.invalid,
        0
    );
    assert!(
        totals.both_agreed > 0,
        "no completed structured comparisons"
    );
}
