//! Synthetic, single-dword GFX10 instructions. Targets are absolute source
//! instruction indices; only the encoder converts them to PC-relative words.
use super::{Inputs, LIMIT, Stop};

const EXEC: u8 = 126;
const VCC: u8 = 106;
const SAVE: u8 = 3;
const MASK: u8 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Alu {
    Add(u8), // v0 += inline constant
    XorData, // v0 ^= v1
    AddData, // v0 += v1
    Set(u8), // v0 = inline constant
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mask {
    Input,
    Compare(u8), // unsigned threshold < current v0
    Joined(u8),  // uniform choice between a compare and the input mask
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Region {
    Plain(Alu),
    If {
        yes: Alu,
        no: Alu,
    },
    Masked {
        mask: Mask,
        skip: bool,
        body: Vec<Alu>,
    },
    Repeat {
        times: u8,
        body: Vec<Region>,
    },
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Src {
    S(u8),
    Imm(u8),
}
use Src::{Imm, S};

#[derive(Clone, Copy, Debug)]
pub(super) enum Op {
    Mov(u8, Src),
    Inc,
    Less(u8, u8),
    Vector(Alu),
    Compare(u8),
    Narrow(u8),
    Scc1(usize),
    ExecZ(usize),
    Jump(usize),
    Observe,
    End,
}
use Op::*;

fn regions(body: &[Region], ops: &mut Vec<Op>) {
    for region in body {
        match region {
            Region::Plain(a) => ops.push(Vector(*a)),
            Region::If { yes, no } => {
                let b = ops.len();
                ops.extend([
                    Less(1, 1),
                    Scc1(b + 4),
                    Vector(*no),
                    Jump(b + 5),
                    Vector(*yes),
                ]);
            }
            Region::Masked { mask, skip, body } => {
                let source = match mask {
                    Mask::Input => 0,
                    Mask::Compare(k) => {
                        ops.push(Compare(*k));
                        VCC
                    }
                    Mask::Joined(k) => {
                        // s4's phi joins a VALU-derived mask with an input SGPR.
                        let b = ops.len();
                        ops.extend([
                            Compare(*k),
                            Mov(MASK, S(VCC)),
                            Less(1, 1),
                            Scc1(b + 5),
                            Mov(MASK, S(0)),
                        ]);
                        MASK
                    }
                };
                ops.push(Narrow(source));
                if *skip {
                    ops.push(ExecZ(ops.len() + 1 + body.len()));
                }
                ops.extend(body.iter().copied().map(Vector));
                ops.push(Mov(EXEC, S(SAVE)));
            }
            Region::Repeat { times, body } => {
                // Generator uses one loop level and a positive trip count.
                ops.push(Mov(2, Imm(0)));
                let header = ops.len();
                regions(body, ops);
                ops.extend([Inc, Less(2, *times), Scc1(header)]);
            }
        }
    }
}

pub(super) fn program(body: &[Region]) -> Vec<Op> {
    let mut ops = Vec::new();
    regions(body, &mut ops);
    ops.extend([Observe, End]);
    ops
}

impl Src {
    fn bits(self) -> u32 {
        match self {
            S(r) => u32::from(r),
            Imm(k) => 128 + u32::from(k),
        }
    }
}

pub(super) fn encode(ops: &[Op]) -> Result<Vec<u8>, Stop> {
    let mut bytes = Vec::with_capacity(ops.len() * 4);
    for (pc, op) in ops.iter().enumerate() {
        let branch = |base: u32, t: usize| -> Result<u32, Stop> {
            if t >= ops.len() {
                return Err(Stop::Invalid("branch outside program"));
            }
            let displacement = i16::try_from(t as isize - pc as isize - 1)
                .map_err(|_| Stop::Invalid("branch out of encoding range"))?;
            Ok(base | u32::from(displacement as u16))
        };
        let word = match *op {
            Mov(r, src) => 0xbe80_0300 | u32::from(r) << 16 | src.bits(),
            Inc => 0x8002_8102, // s_add_u32 s2,s2,1
            Less(r, k) => 0xbf0a_0000 | Imm(k).bits() << 8 | u32::from(r),
            Vector(Alu::Add(k)) => (0x25 << 25) | Imm(k).bits(),
            Vector(Alu::AddData) => (0x25 << 25) | 257,
            Vector(Alu::XorData) => (0x1d << 25) | 257,
            Vector(Alu::Set(k)) => 0x7e00_0200 | Imm(k).bits(),
            Compare(k) => 0x7d82_0000 | Imm(k).bits(),
            Narrow(r) => 0xbe83_3c00 | u32::from(r),
            Scc1(t) => branch(0xbf85_0000, t)?,
            ExecZ(t) => branch(0xbf88_0000, t)?,
            Jump(t) => branch(0xbf82_0000, t)?,
            Observe => 0x7e04_0300, // v_mov_b32 v2,v0
            End => 0xbf81_0000,
        };
        bytes.extend(word.to_le_bytes());
    }
    Ok(bytes)
}

// Independent register machine: never decodes bytes or consults compiler IR.
pub(super) fn direct(ops: &[Op], inputs: &Inputs) -> Result<[u32; 32], Stop> {
    let mut sgpr = [0u32; 128];
    sgpr[0] = inputs.mask;
    sgpr[1] = inputs.selector;
    sgpr[EXEC as usize] = u32::MAX;
    let mut v = inputs.v;
    let mut scc = false;
    let mut observed = false;
    let mut pc = 0;
    for _ in 0..LIMIT {
        let op = *ops
            .get(pc)
            .ok_or(Stop::Invalid("source PC out of bounds"))?;
        pc += 1;
        let exec = sgpr[EXEC as usize];
        match op {
            Mov(r, src) => {
                sgpr[r as usize] = match src {
                    S(s) => sgpr[s as usize],
                    Imm(k) => k.into(),
                }
            }
            Inc => (sgpr[2], scc) = sgpr[2].overflowing_add(1),
            Less(r, k) => scc = sgpr[r as usize] < k.into(),
            Compare(k) => {
                let mut mask = 0u32;
                for (lane, value) in v[0].iter().enumerate() {
                    if u32::from(k) < *value {
                        mask |= 1 << lane;
                    }
                }
                sgpr[VCC as usize] = mask & exec;
            }
            Narrow(r) => {
                let narrowed = exec & sgpr[r as usize];
                sgpr[SAVE as usize] = exec;
                sgpr[EXEC as usize] = narrowed;
                scc = narrowed != 0;
            }
            Vector(a) => {
                let [accumulator, data, _] = &mut v;
                for (lane, (x, &y)) in accumulator.iter_mut().zip(data.iter()).enumerate() {
                    if exec & (1 << lane) != 0 {
                        *x = match a {
                            Alu::Add(k) => x.wrapping_add(k.into()),
                            Alu::AddData => x.wrapping_add(y),
                            Alu::XorData => *x ^ y,
                            Alu::Set(k) => k.into(),
                        };
                    }
                }
            }
            Scc1(t) if scc => pc = t,
            ExecZ(t) if exec == 0 => pc = t,
            Jump(t) => pc = t,
            Scc1(_) | ExecZ(_) => {}
            Observe => {
                if observed || exec != u32::MAX {
                    return Err(Stop::Invalid("observer must run once at full EXEC"));
                }
                v[2] = v[0];
                observed = true;
            }
            End if observed => return Ok(v[2]),
            End => return Err(Stop::Invalid("missing observer")),
        }
    }
    Err(Stop::Limit)
}
