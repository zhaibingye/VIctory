//! CPU-only differential execution. See tests/equivalence/README.md for the
//! language, equivalence boundary, campaign command and single-seed replay.
#[path = "equivalence/search.rs"]
mod search;
#[path = "equivalence/source.rs"]
mod source;
#[path = "equivalence/ssa.rs"]
mod ssa;
#[path = "equivalence/structured.rs"]
mod structured;

use source::{Alu, Mask, Region};
use victory_shader::{ShaderError, ir, structure};

const LIMIT: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Stop {
    Unsupported(&'static str),
    Invalid(&'static str),
    Limit,
}

#[derive(Clone, Debug)]
struct Inputs {
    mask: u32,
    selector: u32,
    v: [[u32; 32]; 3],
}

impl Inputs {
    fn scalar(&self, reg: ir::Reg) -> Result<u32, Stop> {
        use ir::Reg::*;
        match reg {
            S(0) => Ok(self.mask),
            S(1) => Ok(self.selector),
            S(2..=4) | S(ir::VCC) | Scc => Ok(0),
            S(ir::EXEC) => Ok(u32::MAX),
            _ => Err(Stop::Invalid("unexpected scalar input")),
        }
    }

    fn vector(&self, r: u8) -> Result<[u32; 32], Stop> {
        self.v
            .get(r as usize)
            .copied()
            .ok_or(Stop::Invalid("unexpected vector input"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Outcome {
    Agree,
    Stop(&'static str, Stop),
    Refused(&'static str, usize, &'static str),
    CompilerError(&'static str, String),
    Crash(&'static str, String),
    Difference {
        stage: &'static str,
        lane: usize,
        expected: u32,
        actual: u32,
    },
}

fn compare(stage: &'static str, expected: &[u32; 32], actual: [u32; 32]) -> Outcome {
    for (lane, (&x, y)) in expected.iter().zip(actual).enumerate() {
        if x != y {
            return Outcome::Difference {
                stage,
                lane,
                expected: x,
                actual: y,
            };
        }
    }
    Outcome::Agree
}

fn failure(stage: &'static str, error: ShaderError) -> Outcome {
    match error {
        ShaderError::Unsupported { offset, what } => Outcome::Refused(stage, offset, what),
        e => Outcome::CompilerError(stage, e.to_string()),
    }
}

fn checked<T>(stage: &'static str, run: impl FnOnce() -> T) -> Result<T, Outcome> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).map_err(|error| {
        let msg = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied())
            .unwrap_or("untyped panic");
        Outcome::Crash(stage, msg.to_owned())
    })
}

// Tests inject faults only in their own copy of actual transformed output.
type Perturb = fn(&ir::Program, &mut structure::Structured);
fn unchanged(_: &ir::Program, _: &mut structure::Structured) {}

#[derive(Debug)]
struct Comparison {
    ssa_agreed: bool,
    outcome: Outcome,
}

fn evaluate(body: &[Region], inputs: &Inputs, perturb: Perturb) -> Comparison {
    let mut ssa_agreed = false;
    let mut run = || -> Result<Outcome, Outcome> {
        let ops = source::program(body);
        let expected = checked("source", || source::direct(&ops, inputs))?
            .map_err(|e| Outcome::Stop("source", e))?;
        let code = source::encode(&ops).map_err(|e| Outcome::Stop("encode", e))?;
        let p =
            checked("build", || ir::build(&code, false, 3))?.map_err(|e| failure("build", e))?;
        let actual =
            checked("ssa", || ssa::execute(&p, inputs))?.map_err(|e| Outcome::Stop("ssa", e))?;
        let first = compare("ssa", &expected, actual);
        if first != Outcome::Agree {
            return Ok(first);
        }
        ssa_agreed = true;
        let mut s = checked("structure", || structure::structure(&p))?
            .map_err(|e| failure("structure", e))?;
        perturb(&p, &mut s);
        let values = checked("structured", || structured::execute(&p, &s, inputs))?
            .map_err(|e| Outcome::Stop("structured", e))?;
        Ok(compare("structured", &expected, values))
    };
    let outcome = run().unwrap_or_else(|e| e);
    Comparison {
        ssa_agreed,
        outcome,
    }
}

fn fixture() -> Inputs {
    Inputs {
        mask: 0x8000_0005,
        selector: 0,
        v: [
            std::array::from_fn(|i| i as u32),
            [3; 32],
            [0xfeed_0000; 32],
        ],
    }
}

fn hand(body: &[Region], inputs: &Inputs, expected: [u32; 32]) {
    let words = source::program(body);
    assert_eq!(source::direct(&words, inputs), Ok(expected));
    assert_eq!(evaluate(body, inputs, unchanged).outcome, Outcome::Agree);
}

#[test]
fn straight_line_and_overflow() {
    let mut i = fixture();
    i.v[0] = [u32::MAX; 32];
    hand(
        &[Region::Plain(Alu::Add(2)), Region::Plain(Alu::XorData)],
        &i,
        [2; 32],
    );
}

#[test]
fn both_uniform_phi_edges() {
    let mut i = fixture();
    let body = [
        Region::If {
            yes: Alu::Set(12),
            no: Alu::Set(19),
        },
        Region::Plain(Alu::Add(1)),
    ];
    hand(&body, &i, [13; 32]);
    i.selector = 1;
    hand(&body, &i, [20; 32]);
}

#[test]
fn inactive_lanes_restore_and_divergent_exits() {
    for skip in [false, true] {
        for mask in [0, 0x8000_0005, u32::MAX] {
            let mut i = fixture();
            i.mask = mask;
            let body = [
                Region::Masked {
                    mask: Mask::Input,
                    skip,
                    body: vec![Alu::Set(17)],
                },
                Region::Plain(Alu::Add(2)),
            ];
            let expect = std::array::from_fn(|lane| match mask {
                0 => lane as u32 + 2,
                u32::MAX => 19,
                _ if matches!(lane, 0 | 2 | 31) => 19,
                _ => lane as u32 + 2,
            });
            hand(&body, &i, expect);
        }
    }
}

#[test]
fn joined_mask_uses_incoming_edge() {
    use structure::Atom;
    let mut i = fixture();
    let body = [Region::Masked {
        mask: Mask::Joined(7),
        skip: false,
        body: vec![Alu::Add(4)],
    }];
    let p = ir::build(&source::encode(&source::program(&body)).unwrap(), false, 3).unwrap();
    let s = structure::structure(&p).unwrap();
    assert!(
        s.guards.iter().flatten().flatten().any(|&g| s
            .bools
            .support(g)
            .iter()
            .any(|a| matches!(a, Atom::Phi(_))))
    );
    hand(
        &body,
        &i,
        std::array::from_fn(|l| l as u32 + if l > 7 { 4 } else { 0 }),
    );
    i.selector = 1;
    hand(
        &body,
        &i,
        std::array::from_fn(|l| l as u32 + if matches!(l, 0 | 2 | 31) { 4 } else { 0 }),
    );
}

#[test]
fn loop_carried_vector_and_counter() {
    hand(
        &[Region::Repeat {
            times: 3,
            body: vec![Region::Plain(Alu::AddData)],
        }],
        &fixture(),
        std::array::from_fn(|l| l as u32 + 9),
    );
}

#[test]
fn wrong_backedge_and_inactive_write_change_results() {
    let i = fixture();
    let ops = source::program(&[Region::Repeat {
        times: 3,
        body: vec![Region::Plain(Alu::AddData)],
    }]);
    let mut p = ir::build(&source::encode(&ops).unwrap(), false, 3).unwrap();
    let b = p
        .blocks
        .iter_mut()
        .find(|b| b.phis.iter().any(|f| f.reg == ir::Reg::V(0)))
        .unwrap();
    let entry = b.preds.iter().position(|&pred| pred == 0).unwrap();
    let f = b.phis.iter_mut().find(|f| f.reg == ir::Reg::V(0)).unwrap();
    f.args[1 - entry] = f.args[entry];
    assert_eq!(
        compare(
            "ssa",
            &source::direct(&ops, &i).unwrap(),
            ssa::execute(&p, &i).unwrap()
        ),
        Outcome::Difference {
            stage: "ssa",
            lane: 0,
            expected: 9,
            actual: 3
        }
    );

    let body = [Region::Masked {
        mask: Mask::Input,
        skip: false,
        body: vec![Alu::Set(17)],
    }];
    let ops = source::program(&body);
    let mut p = ir::build(&source::encode(&ops).unwrap(), false, 3).unwrap();
    let full = p
        .inputs
        .iter()
        .find(|x| x.0 == ir::Reg::S(ir::EXEC))
        .unwrap()
        .1;
    let inst = p.blocks[0]
        .insts
        .iter_mut()
        .find(|i| i.write(ir::Reg::V(0)).is_some())
        .unwrap();
    inst.reads
        .iter_mut()
        .find(|x| x.0 == ir::Reg::S(ir::EXEC))
        .unwrap()
        .1 = full;
    let expected = Outcome::Difference {
        stage: "ssa",
        lane: 1,
        expected: 1,
        actual: 17,
    };
    assert_eq!(
        compare(
            "ssa",
            &source::direct(&ops, &i).unwrap(),
            ssa::execute(&p, &i).unwrap()
        ),
        expected
    );
    assert_eq!(evaluate(&body, &i, unchanged).outcome, Outcome::Agree);
}

fn drop_guard(p: &ir::Program, s: &mut structure::Structured) {
    for (block, guards) in p.blocks.iter().zip(&mut s.guards) {
        for (inst, g) in block.insts.iter().zip(guards) {
            if inst.write(ir::Reg::V(0)).is_some() && g.is_some() {
                *g = None;
                return;
            }
        }
    }
}

#[test]
fn inverted_lane_exit_is_observable() {
    let body = [Region::Masked {
        mask: Mask::Compare(5),
        skip: true,
        body: vec![Alu::Set(17)],
    }];
    let broken = evaluate(&body, &fixture(), |_, s| {
        let exit = s
            .exits
            .iter_mut()
            .find(|e| matches!(e, structure::LaneExit::Branch { .. }))
            .unwrap();
        if let structure::LaneExit::Branch {
            taken, not_taken, ..
        } = exit
        {
            std::mem::swap(taken, not_taken);
        }
    });
    assert_eq!(
        broken.outcome,
        Outcome::Difference {
            stage: "structured",
            lane: 0,
            expected: 0,
            actual: 17
        }
    );
    assert_eq!(
        evaluate(&body, &fixture(), unchanged).outcome,
        Outcome::Agree
    );
}

#[test]
fn mask_changes_between_loop_iterations() {
    let mut i = fixture();
    i.v[0] = std::array::from_fn(|lane| if lane % 2 == 0 { 4 } else { 12 });
    let body = [Region::Repeat {
        times: 3,
        body: vec![
            Region::Masked {
                mask: Mask::Compare(7),
                skip: true,
                body: vec![Alu::Set(0)],
            },
            Region::Plain(Alu::Add(5)),
        ],
    }];
    // 4 -> 9 -> 5 -> 10, and 12 -> 5 -> 10 -> 5.
    hand(
        &body,
        &i,
        std::array::from_fn(|lane| if lane % 2 == 0 { 10 } else { 5 }),
    );
}

#[test]
fn false_observer_guard_preserves_the_defined_old_value() {
    let r = evaluate(&[], &fixture(), |p, s| {
        let b = &p.blocks[0];
        let j = b
            .insts
            .iter()
            .position(|i| i.write(ir::Reg::V(2)).is_some())
            .unwrap();
        s.guards[0][j] = Some(structure::Bool::FALSE);
    });
    assert_eq!(
        r.outcome,
        Outcome::Difference {
            stage: "structured",
            lane: 0,
            expected: 0,
            actual: 0xfeed_0000
        }
    );
}

#[test]
fn bounded_execution_and_compiler_refusal() {
    use source::Op::{End, Jump, Mov, Observe};
    let i = fixture();
    // Reachable self-loop after a preheader, with an observer on the untaken
    // exit. Both compiler representations have a valid loop header.
    let ops = [
        Mov(2, source::Src::Imm(0)),
        source::Op::Less(2, 1),
        source::Op::Scc1(1),
        Observe,
        End,
    ];
    assert_eq!(source::direct(&ops, &i), Err(Stop::Limit));
    let p = ir::build(&source::encode(&ops).unwrap(), false, 3).unwrap();
    assert_eq!(ssa::execute(&p, &i), Err(Stop::Limit));
    let s = structure::structure(&p).unwrap();
    assert_eq!(structured::execute(&p, &s, &i), Err(Stop::Limit));
    // The builder explicitly refuses a branch to the program entry.
    let error = ir::build(&source::encode(&[Jump(0), End]).unwrap(), false, 3).unwrap_err();
    assert!(matches!(
        failure("build", error),
        Outcome::Refused("build", 0, _)
    ));
}
