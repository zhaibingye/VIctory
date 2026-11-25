use std::collections::{HashMap, HashSet};

use victory_shader::{
    gfx10::{Fields, NULL, Operand, Valu},
    ir::{self, Exit, Inst, Reg, Value},
};

use super::{Inputs, LIMIT, Stop};

const EXEC: Reg = Reg::S(ir::EXEC);

#[derive(Clone, Copy)]
enum Data {
    Scalar(u32),
    Vector([u32; 32]),
}

impl Data {
    fn for_reg(self, reg: Reg) -> Result<Self, Stop> {
        match (reg, self) {
            (Reg::V(_), Self::Vector(_)) | (Reg::S(_) | Reg::Scc, Self::Scalar(_)) => Ok(self),
            _ => Err(Stop::Invalid("SSA value has the wrong register type")),
        }
    }
}

#[derive(Default)]
struct Values(HashMap<Value, Data>);

impl Values {
    fn get(&self, value: Value, reg: Reg) -> Result<Data, Stop> {
        self.0
            .get(&value)
            .copied()
            .ok_or(Stop::Invalid("SSA value has no reaching definition"))?
            .for_reg(reg)
    }

    fn define(&mut self, value: Value, reg: Reg, data: Data) -> Result<(), Stop> {
        if value == Value(u32::MAX) || self.0.insert(value, data.for_reg(reg)?).is_some() {
            return Err(Stop::Invalid("invalid or duplicate SSA definition"));
        }
        Ok(())
    }
}

struct Reads<'a> {
    values: &'a Values,
    inst: &'a Inst,
    used: Vec<Reg>,
}

impl Reads<'_> {
    fn get(&mut self, reg: Reg) -> Result<Data, Stop> {
        if !self.used.contains(&reg) {
            self.used.push(reg);
        }
        let value = self
            .inst
            .read(reg)
            .ok_or(Stop::Invalid("missing SSA read binding"))?;
        self.values.get(value, reg)
    }

    fn scalar(&mut self, reg: Reg) -> Result<u32, Stop> {
        match self.get(reg)? {
            Data::Scalar(word) => Ok(word),
            Data::Vector(_) => Err(Stop::Invalid("expected scalar SSA value")),
        }
    }

    fn vector(&mut self, reg: Reg) -> Result<[u32; 32], Stop> {
        match self.get(reg)? {
            Data::Vector(lanes) => Ok(lanes),
            Data::Scalar(_) => Err(Stop::Invalid("expected vector SSA value")),
        }
    }

    fn scalar_operand(&mut self, operand: Operand) -> Result<u32, Stop> {
        match operand {
            Operand::Scalar { .. } => self.scalar(scalar_reg(operand)?),
            Operand::Const(code @ 128..=192) => Ok(u32::from(code - 128)),
            Operand::Literal(value @ 0..=64) => Ok(value),
            Operand::Const(_) | Operand::Literal(_) => Err(Stop::Unsupported("SSA constant")),
            _ => Err(Stop::Invalid("expected scalar SSA operand")),
        }
    }

    fn vector_operand(&mut self, operand: Operand) -> Result<[u32; 32], Stop> {
        match operand {
            Operand::Vector { .. } => self.vector(vector_reg(operand)?),
            _ => Ok([self.scalar_operand(operand)?; 32]),
        }
    }
}

fn scalar_reg(operand: Operand) -> Result<Reg, Stop> {
    match operand {
        Operand::Scalar { code, dwords: 1 } if code < 127 && code != NULL => Ok(Reg::S(code)),
        _ => Err(Stop::Invalid("expected one scalar dword")),
    }
}

fn vector_reg(operand: Operand) -> Result<Reg, Stop> {
    match operand {
        Operand::Vector { reg, dwords: 1 } => Ok(Reg::V(reg)),
        _ => Err(Stop::Invalid("expected one vector dword")),
    }
}

#[derive(Clone, Copy)]
enum Control {
    Next,
    Jump,
    Branch(bool),
    End,
}

fn scalar_effects(reads: &mut Reads<'_>, writes: &mut Vec<(Reg, Data)>) -> Result<(), Stop> {
    let Fields::Sop {
        dst,
        src,
        imm: None,
    } = &reads.inst.fields
    else {
        return Err(Stop::Invalid("invalid SSA scalar fields"));
    };
    let mut write = |reg, word| writes.push((reg, Data::Scalar(word)));
    match (reads.inst.name(), dst, src.as_slice()) {
        ("s_mov_b32", Some(dst), [src]) => {
            write(scalar_reg(*dst)?, reads.scalar_operand(*src)?);
        }
        ("s_add_u32", Some(dst), [left, right]) => {
            let left = reads.scalar_operand(*left)?;
            let right = reads.scalar_operand(*right)?;
            let (sum, carry) = left.overflowing_add(right);
            write(scalar_reg(*dst)?, sum);
            write(Reg::Scc, u32::from(carry));
        }
        ("s_cmp_lt_u32", None, [left, right]) => {
            let left = reads.scalar_operand(*left)?;
            let right = reads.scalar_operand(*right)?;
            write(Reg::Scc, u32::from(left < right));
        }
        ("s_and_saveexec_b32", Some(dst), [src]) => {
            let old = reads.scalar(EXEC)?;
            let new = old & reads.scalar_operand(*src)?;
            write(scalar_reg(*dst)?, old);
            write(EXEC, new);
            write(Reg::Scc, u32::from(new != 0));
        }
        _ => return Err(Stop::Invalid("invalid SSA scalar operands")),
    }
    Ok(())
}

fn vector_effects(reads: &mut Reads<'_>, writes: &mut Vec<(Reg, Data)>) -> Result<(), Stop> {
    let Fields::Valu(valu) = &reads.inst.fields else {
        return Err(Stop::Invalid("invalid SSA vector fields"));
    };
    let Valu {
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
    } = valu
    else {
        return Err(Stop::Unsupported("SSA vector modifiers"));
    };
    let dst = dst.ok_or(Stop::Invalid("missing SSA vector instruction destination"))?;
    let exec = reads.scalar(EXEC)?;
    let (reg, data) = match (reads.inst.name(), src.as_slice()) {
        ("v_cmp_lt_u32", [left, right]) => {
            let left = reads.vector_operand(*left)?;
            let right = reads.vector_operand(*right)?;
            let mut mask = 0;
            for lane in 0..32 {
                if exec & (1 << lane) != 0 && left[lane] < right[lane] {
                    mask |= 1 << lane;
                }
            }
            (scalar_reg(dst)?, Data::Scalar(mask))
        }
        (name @ ("v_mov_b32" | "v_add_nc_u32" | "v_xor_b32"), src) => {
            let reg = vector_reg(dst)?;
            let mut lanes = reads.vector(reg)?;
            let result = match (name, src) {
                ("v_mov_b32", [src]) => reads.vector_operand(*src)?,
                ("v_add_nc_u32" | "v_xor_b32", [left, right]) => {
                    let left = reads.vector_operand(*left)?;
                    let right = reads.vector_operand(*right)?;
                    let op = if name == "v_add_nc_u32" {
                        u32::wrapping_add
                    } else {
                        |a, b| a ^ b
                    };
                    std::array::from_fn(|lane| op(left[lane], right[lane]))
                }
                _ => return Err(Stop::Invalid("invalid SSA vector operands")),
            };
            for (lane, dst) in lanes.iter_mut().enumerate() {
                if exec & (1 << lane) != 0 {
                    *dst = result[lane];
                }
            }
            (reg, Data::Vector(lanes))
        }
        _ => return Err(Stop::Invalid("invalid SSA vector operands")),
    };
    writes.push((reg, data));
    Ok(())
}

fn execute_inst(values: &mut Values, inst: &Inst) -> Result<Control, Stop> {
    let mut reads = Reads {
        values,
        inst,
        used: Vec::new(),
    };
    let mut writes = Vec::new();
    let control = match inst.name() {
        "s_mov_b32" | "s_add_u32" | "s_cmp_lt_u32" | "s_and_saveexec_b32" => {
            scalar_effects(&mut reads, &mut writes)?;
            Control::Next
        }
        "v_cmp_lt_u32" | "v_mov_b32" | "v_add_nc_u32" | "v_xor_b32" => {
            vector_effects(&mut reads, &mut writes)?;
            Control::Next
        }
        name @ ("s_cbranch_scc1" | "s_cbranch_execz" | "s_branch" | "s_endpgm") => {
            if !matches!(inst.fields, Fields::Sopp { .. }) {
                return Err(Stop::Invalid("invalid SSA control fields"));
            }
            match name {
                "s_cbranch_scc1" => Control::Branch(reads.scalar(Reg::Scc)? != 0),
                "s_cbranch_execz" => Control::Branch(reads.scalar(EXEC)? == 0),
                "s_branch" => Control::Jump,
                _ => Control::End,
            }
        }
        name => return Err(Stop::Unsupported(name)),
    };
    if reads.used.len() != inst.reads.len() || writes.len() != inst.writes.len() {
        return Err(Stop::Invalid("unexpected SSA operand bindings"));
    }
    for (reg, data) in writes {
        let value = inst
            .write(reg)
            .ok_or(Stop::Invalid("missing SSA write binding"))?;
        values.define(value, reg, data)?;
    }
    Ok(control)
}

fn observer(program: &ir::Program) -> Result<Value, Stop> {
    let mut observers = program
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .filter(|inst| {
            matches!(
                (&inst.fields, inst.name()),
                (Fields::Valu(Valu { dst: Some(Operand::Vector { reg: 2, dwords: 1 }), src, .. }),
                 "v_mov_b32") if matches!(src.as_slice(), [Operand::Vector { reg: 0, dwords: 1 }])
            )
        });
    let value = observers
        .next()
        .and_then(|inst| inst.write(Reg::V(2)))
        .ok_or(Stop::Invalid("missing SSA observer definition"))?;
    if observers.next().is_some() {
        return Err(Stop::Invalid("multiple SSA observers"));
    }
    Ok(value)
}

pub(super) fn execute(program: &ir::Program, inputs: &Inputs) -> Result<[u32; 32], Stop> {
    if program.wave64 {
        return Err(Stop::Unsupported("Wave64 SSA program"));
    }
    let observation = observer(program)?;
    let mut values = Values::default();
    let mut defined = HashSet::new();
    let mut input_regs = HashSet::new();
    for &(reg, value) in &program.inputs {
        if !input_regs.insert(reg) || !defined.insert(value) {
            return Err(Stop::Invalid("duplicate SSA input"));
        }
        let data = match reg {
            Reg::V(reg) => Data::Vector(inputs.vector(reg)?),
            _ => Data::Scalar(inputs.scalar(reg)?),
        };
        values.define(value, reg, data)?;
    }
    for block in &program.blocks {
        let definitions = block.phis.iter().map(|phi| phi.value).chain(
            block
                .insts
                .iter()
                .flat_map(|inst| inst.writes.iter().map(|&(_, value)| value)),
        );
        for value in definitions {
            if value == Value(u32::MAX) || !defined.insert(value) {
                return Err(Stop::Invalid("invalid or duplicate SSA definition"));
            }
        }
    }

    let mut block_id = 0;
    let mut predecessor = None;
    let mut observed = None;
    for _ in 0..LIMIT {
        let block = program
            .blocks
            .get(block_id)
            .ok_or(Stop::Invalid("missing SSA block"))?;
        let phis = if let Some(predecessor) = predecessor {
            let mut edges = block
                .preds
                .iter()
                .enumerate()
                .filter_map(|(edge, &pred)| (pred == predecessor).then_some(edge));
            let edge = edges
                .next()
                .ok_or(Stop::Invalid("missing SSA predecessor edge"))?;
            if edges.next().is_some() {
                return Err(Stop::Invalid("duplicate SSA predecessor edge"));
            }
            block
                .phis
                .iter()
                .map(|phi| {
                    if phi.args.len() != block.preds.len() {
                        return Err(Stop::Invalid(
                            "SSA phi argument count differs from predecessors",
                        ));
                    }
                    values.get(phi.args[edge], phi.reg)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            if !block.preds.is_empty() || !block.phis.is_empty() {
                return Err(Stop::Invalid("invalid SSA entry block"));
            }
            Vec::new()
        };
        // Snapshot every incoming value before updating phi outputs or invalidating
        // definitions from an earlier visit to this block.
        for inst in &block.insts {
            for &(_, value) in &inst.writes {
                values.0.remove(&value);
            }
        }
        for (phi, data) in block.phis.iter().zip(phis) {
            values.0.remove(&phi.value);
            values.define(phi.value, phi.reg, data)?;
        }

        let mut control = Control::Next;
        for inst in &block.insts {
            if !matches!(control, Control::Next) {
                return Err(Stop::Invalid("SSA control instruction precedes block end"));
            }
            control = execute_inst(&mut values, inst)?;
            if inst.write(Reg::V(2)) == Some(observation) {
                let Data::Vector(lanes) = values.get(observation, Reg::V(2))? else {
                    return Err(Stop::Invalid("SSA observer result is not a vector"));
                };
                if observed.replace(lanes).is_some() {
                    return Err(Stop::Invalid("SSA path repeated the observer"));
                }
            }
        }
        predecessor = Some(block_id);
        block_id = match (block.exit, control) {
            (Exit::Jump(target), Control::Next | Control::Jump) => target,
            (Exit::Branch { taken, not_taken }, Control::Branch(choice)) => {
                if choice {
                    taken
                } else {
                    not_taken
                }
            }
            (Exit::End, Control::End) => {
                return observed.ok_or(Stop::Invalid("SSA path skipped the observer"));
            }
            (Exit::Indirect, _) => return Err(Stop::Unsupported("indirect SSA exit")),
            _ => return Err(Stop::Invalid("SSA instruction and block exit disagree")),
        };
    }
    Err(Stop::Limit)
}
