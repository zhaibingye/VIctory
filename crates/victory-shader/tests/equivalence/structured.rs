//! Executes the compiler's lane control flow and lookup tables directly.
//! The corpus puts uniform scalar work outside masked regions and uses
//! mask registers only for control flow. No numeric EXEC or VCC result is
//! reconstructed from other lanes.

use std::collections::{HashMap, HashSet};

use victory_shader::{
    gfx10::{Fields, Operand, Valu},
    ir::{Block, EXEC, Inst, Program, Reg, VCC, Value},
    structure::{Atom, Bool, LaneExit, Structured},
};

use super::{Inputs, LIMIT, Stop};

#[derive(Clone, Copy)]
enum Read {
    Value(Value),
    Literal(u32),
}

impl Read {
    fn operand(inst: &Inst, op: Operand, vector: bool) -> Result<Self, Stop> {
        match op {
            Operand::Scalar {
                code: code @ 1..=2,
                dwords: 1,
            } => Ok(Self::Value(read(inst, Reg::S(code))?)),
            Operand::Vector {
                reg: reg @ 0..=1,
                dwords: 1,
            } if vector => Ok(Self::Value(read(inst, Reg::V(reg))?)),
            Operand::Literal(n) => Ok(Self::Literal(n)),
            Operand::Const(code @ 128..=192) => Ok(Self::Literal(u32::from(code - 128))),
            Operand::Const(code @ 193..=208) => {
                Ok(Self::Literal((192i32 - i32::from(code)) as u32))
            }
            _ => Err(Stop::Unsupported("structured numeric operand")),
        }
    }
}

#[derive(Clone, Copy)]
enum Alu {
    Move(Read),
    Add(Read, Read),
    Xor(Read, Read),
    Less(Read, Read),
}

#[derive(Clone, Copy)]
enum Step {
    Value(Value, Alu),
    AddWithCarry(Value, Value, Read, Read),
    Compare(Value, Read, Read),
    Vector {
        dst: Value,
        old: Value,
        alu: Alu,
        observer: bool,
    },
    Symbolic,
}

impl Step {
    fn decode(inst: &Inst) -> Result<Self, Stop> {
        let s = |op| Read::operand(inst, op, false);
        let v = |op| Read::operand(inst, op, true);
        let scalar = |code| write(inst, Reg::S(code));
        let scc = || write(inst, Reg::Scc);
        let unsupported = || Stop::Unsupported("structured opcode or operand form");
        match (&inst.fields, inst.name()) {
            (
                Fields::Sop {
                    dst,
                    src,
                    imm: None,
                },
                name,
            ) => match (name, dst, src.as_slice()) {
                (
                    "s_mov_b32",
                    Some(Operand::Scalar {
                        code: code @ 1..=2,
                        dwords: 1,
                    }),
                    &[src],
                ) => Ok(Self::Value(scalar(*code)?, Alu::Move(s(src)?))),
                (
                    "s_add_u32",
                    Some(Operand::Scalar {
                        code: code @ 1..=2,
                        dwords: 1,
                    }),
                    &[a, b],
                ) => Ok(Self::AddWithCarry(scalar(*code)?, scc()?, s(a)?, s(b)?)),
                ("s_cmp_lt_u32", None, &[a, b]) => Ok(Self::Value(scc()?, Alu::Less(s(a)?, s(b)?))),
                (
                    "s_mov_b32",
                    Some(Operand::Scalar {
                        code: 3 | 4 | EXEC,
                        dwords: 1,
                    }),
                    &[src],
                )
                | ("s_and_saveexec_b32", Some(Operand::Scalar { code: 3, dwords: 1 }), &[src])
                    if mask_operand(src) =>
                {
                    // Uses resolve these writes through the structured mask expressions.
                    Ok(Self::Symbolic)
                }
                _ => Err(unsupported()),
            },
            (
                Fields::Sopp { .. },
                "s_cbranch_scc1" | "s_cbranch_execz" | "s_branch" | "s_endpgm",
            ) => Ok(Self::Symbolic),
            (
                Fields::Valu(Valu {
                    dst,
                    src,
                    carry: None,
                    neg: 0,
                    abs: 0,
                    neg_hi: 0,
                    opsel: 0,
                    opsel_hi: 0,
                    clamp: false,
                    omod: 0,
                    sdwa: None,
                    dpp: None,
                }),
                name,
            ) => {
                if let (
                    "v_cmp_lt_u32",
                    Some(Operand::Scalar {
                        code: VCC,
                        dwords: 1,
                    }),
                    &[a, b],
                ) = (name, dst, src.as_slice())
                {
                    return Ok(Self::Compare(scalar(VCC)?, v(a)?, v(b)?));
                }
                let Some(Operand::Vector { reg, dwords: 1 }) = *dst else {
                    return Err(unsupported());
                };
                let (alu, observer) = match (name, reg, src.as_slice()) {
                    ("v_mov_b32", 0, &[src]) => (Alu::Move(v(src)?), false),
                    ("v_add_nc_u32", 0, &[a, b]) => (Alu::Add(v(a)?, v(b)?), false),
                    ("v_xor_b32", 0, &[a, b]) => (Alu::Xor(v(a)?, v(b)?), false),
                    ("v_mov_b32", 2, &[src @ Operand::Vector { reg: 0, dwords: 1 }]) => {
                        (Alu::Move(v(src)?), true)
                    }
                    _ => return Err(unsupported()),
                };
                Ok(Self::Vector {
                    dst: write(inst, Reg::V(reg))?,
                    old: read(inst, Reg::V(reg))?,
                    alu,
                    observer,
                })
            }
            _ => Err(unsupported()),
        }
    }
}

fn mask_operand(op: Operand) -> bool {
    matches!(
        op,
        Operand::Scalar {
            code: 0 | 3 | 4 | VCC | EXEC,
            dwords: 1,
        } | Operand::Const(128..=208)
            | Operand::Literal(_)
    )
}

fn read(inst: &Inst, reg: Reg) -> Result<Value, Stop> {
    inst.read(reg)
        .ok_or(Stop::Invalid("missing structured SSA read"))
}

fn write(inst: &Inst, reg: Reg) -> Result<Value, Stop> {
    inst.write(reg)
        .ok_or(Stop::Invalid("missing structured SSA write"))
}

fn incoming_mask(s: &Structured, arg: Value) -> Option<Bool> {
    s.masks.get(&arg).or_else(|| s.other.get(&arg)).copied()
}

fn used_mask_phis(p: &Program, s: &Structured) -> Result<HashSet<Value>, Stop> {
    let branch_masks = s.exits.iter().filter_map(|exit| match exit {
        LaneExit::Branch { when, .. } => Some(*when),
        _ => None,
    });
    let mut pending: Vec<_> = s
        .guards
        .iter()
        .flatten()
        .flatten()
        .copied()
        .chain(branch_masks)
        .collect();
    let phis: HashMap<_, _> = p
        .blocks
        .iter()
        .flat_map(|block| &block.phis)
        .map(|phi| (phi.value, phi))
        .collect();
    let (mut visited, mut used) = (HashSet::new(), HashSet::new());
    // Only control-flow roots consume a mask in this subset. The tables can
    // also contain speculative mask expressions for ordinary scalar values.
    while let Some(mask) = pending.pop() {
        if visited.insert(mask) {
            for atom in s.bools.support(mask) {
                if let Atom::Phi(id) = atom
                    && used.insert(id)
                {
                    let phi = phis
                        .get(&id)
                        .ok_or(Stop::Invalid("structured mask atom without SSA phi"))?;
                    pending.extend(phi.args.iter().filter_map(|&arg| incoming_mask(s, arg)));
                }
            }
        }
    }
    Ok(used)
}

struct Lane<'a> {
    index: usize,
    structured: &'a Structured,
    values: HashMap<Value, u32>,
    bits: HashMap<Atom, bool>,
    observed: Option<Value>,
}

impl Lane<'_> {
    fn value(&self, id: Value) -> Result<u32, Stop> {
        self.values
            .get(&id)
            .copied()
            .ok_or(Stop::Invalid("undefined structured SSA value"))
    }

    fn read(&self, arg: Read) -> Result<u32, Stop> {
        match arg {
            Read::Value(id) => self.value(id),
            Read::Literal(n) => Ok(n),
        }
    }

    fn alu(&self, alu: Alu) -> Result<u32, Stop> {
        let get = |arg| self.read(arg);
        Ok(match alu {
            Alu::Move(a) => get(a)?,
            Alu::Add(a, b) => get(a)?.wrapping_add(get(b)?),
            Alu::Xor(a, b) => get(a)? ^ get(b)?,
            Alu::Less(a, b) => u32::from(get(a)? < get(b)?),
        })
    }

    fn boolean(&self, mut b: Bool) -> Result<bool, Stop> {
        while let Some((atom, low, high)) = self.structured.bools.node(b) {
            let bit = match atom {
                Atom::Lane(_) | Atom::Phi(_) => *self
                    .bits
                    .get(&atom)
                    .ok_or(Stop::Invalid("undefined structured lane bit"))?,
                Atom::Uniform(id) => self.value(id)? != 0,
                Atom::Opaque(id) => self.value(id)? >> self.index & 1 != 0,
                Atom::Constant(bits) => bits >> self.index & 1 != 0,
                Atom::Quad(_) => return Err(Stop::Unsupported("structured quad mask")),
            };
            b = if bit { high } else { low };
        }
        Ok(b == Bool::TRUE)
    }

    fn step(&mut self, step: Step, guard: Option<Bool>) -> Result<(), Stop> {
        match step {
            Step::Value(dst, alu) => {
                self.values.insert(dst, self.alu(alu)?);
            }
            Step::AddWithCarry(dst, scc, a, b) => {
                let (result, carry) = self.read(a)?.overflowing_add(self.read(b)?);
                self.values.insert(dst, result);
                self.values.insert(scc, u32::from(carry));
            }
            Step::Compare(dst, a, b) => {
                // A Lane atom is the raw predicate. The BDD for the mask
                // already incorporates the compare's EXEC.
                self.bits
                    .insert(Atom::Lane(dst), self.read(a)? < self.read(b)?);
            }
            Step::Vector {
                dst,
                old,
                alu,
                observer,
            } => {
                let enabled = self.boolean(guard.unwrap_or(Bool::TRUE))?;
                if observer && self.observed.replace(dst).is_some() {
                    return Err(Stop::Invalid("repeated structured observer"));
                }
                let result = if enabled {
                    self.alu(alu)?
                } else {
                    self.value(old)?
                };
                self.values.insert(dst, result);
            }
            Step::Symbolic => {}
        }
        Ok(())
    }

    fn enter(
        &mut self,
        block: &Block,
        pred: Option<usize>,
        mask_phis: &HashSet<Value>,
    ) -> Result<(), Stop> {
        if block.phis.is_empty() {
            return Ok(());
        }
        let edge = block
            .preds
            .iter()
            .position(|&p| Some(p) == pred)
            .ok_or(Stop::Invalid(
                "structured phi entered from a non-predecessor",
            ))?;
        let mut values = Vec::new();
        let mut bits = Vec::new();
        for phi in &block.phis {
            let arg = *phi
                .args
                .get(edge)
                .ok_or(Stop::Invalid("missing structured phi argument"))?;
            if mask_phis.contains(&phi.value) {
                let mask = incoming_mask(self.structured, arg)
                    .ok_or(Stop::Unsupported("structured phi incoming mask"))?;
                bits.push((Atom::Phi(phi.value), self.boolean(mask)?));
            }
            if matches!(phi.reg, Reg::S(1..=2) | Reg::Scc | Reg::V(0..=1)) {
                values.push((phi.value, self.value(arg)?));
            }
        }
        // Both sets of incoming expressions see the predecessor's values.
        // In particular, a back edge can read a previous iteration's Phi atom.
        self.values.extend(values);
        self.bits.extend(bits);
        Ok(())
    }
}

pub(super) fn execute(p: &Program, s: &Structured, inputs: &Inputs) -> Result<[u32; 32], Stop> {
    if p.wave64 || !s.discards.is_empty() {
        return Err(Stop::Unsupported("structured wave size or discard"));
    }
    if s.exits.len() != p.blocks.len() || s.guards.len() != p.blocks.len() {
        return Err(Stop::Invalid("structured block table size"));
    }
    let steps: Vec<Vec<_>> = p
        .blocks
        .iter()
        .zip(&s.guards)
        .map(|(block, guards)| {
            if block.insts.len() != guards.len() {
                return Err(Stop::Invalid("structured guard table size"));
            }
            block.insts.iter().map(Step::decode).collect()
        })
        .collect::<Result<_, _>>()?;
    let mask_phis = used_mask_phis(p, s)?;

    let mut result = [0; 32];
    for (index, output) in result.iter_mut().enumerate() {
        let mut lane = Lane {
            index,
            structured: s,
            values: HashMap::new(),
            bits: HashMap::new(),
            observed: None,
        };
        for &(reg, id) in &p.inputs {
            let value = match reg {
                // Entry EXEC is encoded as true by the structure pass.
                Reg::S(EXEC) => continue,
                Reg::V(reg) => inputs.vector(reg)?[index],
                _ => inputs.scalar(reg)?,
            };
            lane.values.insert(id, value);
        }
        let (mut b, mut pred) = (0, None);
        let mut remaining = LIMIT;
        loop {
            let block = p
                .blocks
                .get(b)
                .ok_or(Stop::Invalid("structured exit target"))?;
            remaining = remaining.checked_sub(1).ok_or(Stop::Limit)?;
            lane.enter(block, pred, &mask_phis)?;
            for (&step, &guard) in steps[b].iter().zip(&s.guards[b]) {
                lane.step(step, guard)?;
            }
            let next = match s.exits[b] {
                LaneExit::Jump(target) => target,
                LaneExit::Branch {
                    when,
                    taken,
                    not_taken,
                } => {
                    if lane.boolean(when)? {
                        taken
                    } else {
                        not_taken
                    }
                }
                LaneExit::End => {
                    *output = lane.value(
                        lane.observed
                            .ok_or(Stop::Invalid("missing structured observer"))?,
                    )?;
                    break;
                }
                LaneExit::Indirect => return Err(Stop::Unsupported("structured indirect exit")),
            };
            (b, pred) = (next, Some(b));
        }
    }
    Ok(result)
}
