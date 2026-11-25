//! The console's rectangle lists on the host. A rectangle comes as three
//! corners, the second between the other two, and the hardware adds the
//! fourth at `v0 + v2 - v1`, for the position and every interpolated
//! attribute. The host has no such primitive; a geometry shader made for the
//! vertex shader that feeds it takes each triangle `(v0, v1, v2)` and emits
//! it with its completion `(v0, v2, v3)`, both turning the same way. Flat
//! outputs keep `v0`'s value at the fourth corner, and channels the vertex
//! shader leaves unwritten stay unwritten.

use super::bitcode::{Attr, Bin, Term};
use super::shader::Shader;
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};

/// The geometry shader that completes the rectangles of the vertex shader
/// container `vs` (its output signature is the shader's input and output).
pub fn rectangle_gs(vs: &[u8]) -> Result<Vec<u8>, ShaderError> {
    let c = Container::parse(vs)?;
    let part = c
        .part(FourCC::OSG1)
        .ok_or(ShaderError::Malformed("a vertex shader without outputs"))?;
    let mut elements = super::signature(part.data)?;
    let base = crate::dxbc::u32_at(part.data, 4)? as usize;
    for (i, e) in elements.iter_mut().enumerate() {
        let at = base + 32 * i;
        let rw = part
            .data
            .get(at + 24..at + 26)
            .ok_or(ShaderError::Malformed("truncated"))?;
        let mask = ((1u32 << e.cols) - 1) as u8;
        e.usage = ((rw[0] & !rw[1] & 0xf) >> e.col) & mask;
    }
    elements.retain(|e| e.row.is_some());
    if elements.iter().all(|e| e.usage == 0) {
        return Err(ShaderError::Malformed("a vertex shader without outputs"));
    }
    let inputs = elements.clone();
    for e in &mut elements {
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            (13, _) => 1,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
    }
    super::container::same_mode_per_row(&mut elements, false);

    let mut s = Shader::new(false);
    s.gs = Some(6);
    let b = s.entry;
    // Corner k's values, element by element, component by component.
    let mut corners: Vec<Vec<Vec<Option<super::bitcode::V>>>> = Vec::new();
    for k in 0..3u32 {
        let mut values = Vec::new();
        for (id, e) in elements.iter().enumerate() {
            let (ty, suffix) = if e.comp == 9 {
                (s.t.f32, ".f32")
            } else {
                (s.t.i32, ".i32")
            };
            let mut comps = Vec::with_capacity(usize::from(e.cols));
            for col in 0..e.cols {
                let v = (e.usage >> col & 1 != 0).then(|| {
                    let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), s.m.i32(k)];
                    s.op(b, &format!("loadInput{suffix}"), Attr::ReadNone, ty, 4, &args)
                });
                comps.push(v);
            }
            values.push(comps);
        }
        corners.push(values);
    }
    let fourth: Vec<Vec<_>> = elements
        .iter()
        .enumerate()
        .map(|(id, e)| {
            (0..usize::from(e.cols))
                .map(|col| {
                    let (v0, v1, v2) = (
                        corners[0][id][col]?,
                        corners[1][id][col]?,
                        corners[2][id][col]?,
                    );
                    Some(if e.comp == 9 && e.interp != 1 {
                        let sum = s.m.bin(b, Bin::Add, v0, v2);
                        s.m.bin(b, Bin::Sub, sum, v1)
                    } else {
                        v0
                    })
                })
                .collect()
        })
        .collect();
    corners.push(fourth);
    for triangle in [[0, 1, 2], [0, 2, 3]] {
        for k in triangle {
            for (id, e) in elements.iter().enumerate() {
                let suffix = if e.comp == 9 { ".f32" } else { ".i32" };
                for col in 0..e.cols {
                    let Some(v) = corners[k][id][usize::from(col)] else {
                        continue;
                    };
                    let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), v];
                    s.op(b, &format!("storeOutput{suffix}"), Attr::NoUnwind, s.t.void, 5, &args);
                }
            }
            let stream = [s.m.i8(0)];
            s.op(b, "emitStream", Attr::NoUnwind, s.t.void, 97, &stream);
        }
        let stream = [s.m.i8(0)];
        s.op(b, "cutStream", Attr::NoUnwind, s.t.void, 98, &stream);
    }
    s.inputs = inputs;
    s.outputs = elements;
    s.m.terminate(b, Term::Ret);
    Ok(s.finish())
}

#[cfg(all(test, windows))]
mod tests {
    use super::super::bitcode::{Attr, Term};
    use super::super::container::SigElement;
    use super::super::host::{Draw, Host};
    use super::super::shader::Shader;

    fn element(name: &str, kind: u8, interp: u8, row: u32, cols: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp,
            row: Some(row),
            col: 0,
            cols,
            usage: (1 << cols) - 1,
            half: false,
        }
    }

    /// A shader copying its inputs to its outputs (vertex), or writing
    /// target 0 from TEXCOORD (pixel).
    fn shader(pixel: bool) -> Vec<u8> {
        let mut s = Shader::new(pixel);
        let b = s.entry;
        let (inputs, outputs) = if pixel {
            (
                vec![element("SV_Position", 3, 4, 0, 4), element("TEXCOORD", 0, 2, 1, 2)],
                vec![element("SV_Target", 16, 0, 0, 4)],
            )
        } else {
            (
                vec![element("POSITION", 0, 0, 0, 4), element("TEXCOORD", 0, 0, 1, 2)],
                vec![element("SV_Position", 3, 0, 0, 4), element("TEXCOORD", 0, 0, 1, 2)],
            )
        };
        let load = |s: &mut Shader, id: u32, col: u8| {
            let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)];
            s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args)
        };
        let values: Vec<_> = if pixel {
            let one = s.m.f32(1f32.to_bits());
            vec![load(&mut s, 1, 0), load(&mut s, 1, 1), one, one]
        } else {
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
                .into_iter()
                .map(|(id, c)| load(&mut s, id, c))
                .collect()
        };
        let mut k = 0;
        for (id, e) in outputs.iter().enumerate() {
            for col in 0..e.cols {
                let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), values[k]];
                s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
                k += 1;
            }
        }
        s.inputs = inputs;
        s.outputs = outputs;
        s.m.terminate(b, Term::Ret);
        s.finish()
    }

    /// One rectangle (top left, bottom left, bottom right) with a
    /// coordinate attribute: every pixel of the square is drawn, the
    /// attribute interpolated over it, the top-right half included.
    #[test]
    fn rectangles_get_their_fourth_corner() {
        let (vs, ps) = (shader(false), shader(true));
        let gs = super::rectangle_gs(&vs).unwrap();
        if let Ok(mut v) =
            super::super::validator::Validator::load(super::super::validator::Validator::default_path().as_deref())
        {
            for c in [&vs, &gs, &ps] {
                v.validate(&mut c.clone()).unwrap();
            }
        }
        #[rustfmt::skip]
        let vertices = [
            -0.5, 0.5, 0.5, 1.0, -0.5, 0.5,
            -0.5, -0.5, 0.5, 1.0, -0.5, -0.5,
            0.5, -0.5, 0.5, 1.0, 0.5, -0.5,
        ];
        for warp in [false, true] {
            let Ok(host) = Host::new(warp) else { continue };
            let root = host.root_signature_for(&[]).unwrap();
            let pso = host.pipeline_gs(&root, &vs, &gs, &ps).unwrap();
            let draw = Draw {
                roots: Vec::new(),
                textures: &[],
                samplers: &[],
                vertices: &vertices,
                stride: 24,
                size: (32, 32),
                mesh_groups: 0,
            };
            let texels = host.draw(&root, &pso, &draw).unwrap();
            for (i, t) in texels.iter().enumerate() {
                let (x, y) = ((i % 32) as f32, (i / 32) as f32);
                let (u, v) = ((x + 0.5) / 16.0 - 1.0, 1.0 - (y + 0.5) / 16.0);
                let inside = u.abs() < 0.5 && v.abs() < 0.5;
                let want = if inside { [u, v, 1.0, 1.0] } else { [0.0; 4] };
                let near = t.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-4);
                assert!(near, "warp {warp} pixel ({x}, {y}): {t:?}, want {want:?}");
            }
        }
    }

    /// Unwritten elements and channels in the vertex shader's `OSG1` stay
    /// unread and unwritten in the geometry shader, and outputs get their
    /// DXIL interpolation modes (flat across a shared float-and-integer row),
    /// so stage linkage and drawing succeed on both hardware and WARP.
    #[test]
    fn unwritten_channels_and_flat_outputs_link_and_draw() {
        use super::super::bitcode::Cast;
        use crate::dxbc::{Container, FourCC};

        let vs = {
            let mut s = Shader::new(false);
            let b = s.entry;
            s.inputs = vec![
                element("POSITION", 0, 0, 0, 4),
                element("TEXCOORD", 0, 0, 1, 4),
            ];
            s.outputs = vec![
                element("SV_Position", 3, 4, 0, 4),
                SigElement {
                    usage: 0,
                    ..element("COLOR", 0, 2, 1, 4)
                },
                SigElement {
                    usage: 0b0101,
                    ..element("TEXCOORD", 0, 2, 2, 4)
                },
                SigElement {
                    index: 1,
                    ..element("TEXCOORD", 0, 2, 3, 1)
                },
                SigElement {
                    index: 2,
                    comp: 5,
                    col: 1,
                    ..element("TEXCOORD", 0, 1, 3, 1)
                },
            ];
            let load = |s: &mut Shader, id: u32, col: u8| {
                let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)];
                s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args)
            };
            for col in 0..4u8 {
                let v = load(&mut s, 0, col);
                let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), v];
                s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
            }
            for (in_col, out_col) in [(0u8, 0u8), (1u8, 2u8)] {
                let v = load(&mut s, 1, in_col);
                let args = [s.m.i32(2), s.m.i32(0), s.m.i8(out_col), v];
                s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
            }
            let flat_f = load(&mut s, 1, 2);
            let args = [s.m.i32(3), s.m.i32(0), s.m.i8(0), flat_f];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
            let flat_u_f = load(&mut s, 1, 3);
            let flat_u = s.m.cast(b, Cast::FpToUi, flat_u_f, s.t.i32);
            let args = [s.m.i32(4), s.m.i32(0), s.m.i8(0), flat_u];
            s.op(b, "storeOutput.i32", Attr::NoUnwind, s.t.void, 5, &args);
            s.m.terminate(b, Term::Ret);
            s.finish()
        };

        let ps = {
            let mut s = Shader::new(true);
            let b = s.entry;
            s.inputs = vec![
                element("SV_Position", 3, 4, 0, 4),
                SigElement {
                    usage: 0,
                    ..element("COLOR", 0, 2, 1, 4)
                },
                SigElement {
                    usage: 0b0101,
                    ..element("TEXCOORD", 0, 2, 2, 4)
                },
                SigElement {
                    index: 1,
                    ..element("TEXCOORD", 0, 1, 3, 1)
                },
                SigElement {
                    index: 2,
                    comp: 5,
                    col: 1,
                    ..element("TEXCOORD", 0, 1, 3, 1)
                },
            ];
            s.outputs = vec![element("SV_Target", 16, 0, 0, 4)];
            let load_f = |s: &mut Shader, id: u32, col: u8| {
                let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)];
                s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args)
            };
            let u = load_f(&mut s, 2, 0);
            let v = load_f(&mut s, 2, 2);
            let flat_f = load_f(&mut s, 3, 0);
            let args = [s.m.i32(4), s.m.i32(0), s.m.i8(0), s.m.undef(s.t.i32)];
            let flat_u = s.op(b, "loadInput.i32", Attr::ReadNone, s.t.i32, 4, &args);
            let flat_u = s.m.cast(b, Cast::UiToFp, flat_u, s.t.f32);
            for (col, val) in [u, v, flat_f, flat_u].into_iter().enumerate() {
                let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col as u8), val];
                s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
            }
            s.m.terminate(b, Term::Ret);
            s.finish()
        };

        let gs = super::rectangle_gs(&vs).unwrap();
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            for c in [&vs, &gs, &ps] {
                v.validate(&mut c.clone()).unwrap();
            }
        }

        let vs_c = Container::parse(&vs).unwrap();
        let gs_c = Container::parse(&gs).unwrap();
        assert_eq!(
            gs_c.part(FourCC::OSG1).unwrap().data,
            vs_c.part(FourCC::OSG1).unwrap().data
        );
        let psv = gs_c.part(FourCC::PSV0).unwrap().data;
        // Runtime info (4 + 48), 0 bindings (4), string table, index table,
        // then 16-byte PSVSignatureElement records (5 inputs, 5 outputs).
        let str_len = u32::from_le_bytes(psv[56..60].try_into().unwrap()) as usize;
        let idx_at = 60 + str_len;
        let idx_cnt = u32::from_le_bytes(psv[idx_at..idx_at + 4].try_into().unwrap()) as usize;
        let els_at = idx_at + 4 + 4 * idx_cnt + 4;
        let interps: Vec<u8> = (0..10).map(|i| psv[els_at + 16 * i + 13]).collect();
        assert_eq!(interps, [0, 0, 0, 0, 0, 4, 2, 2, 1, 1]);

        #[rustfmt::skip]
        let vertices = [
            -0.5,  0.5, 0.5, 1.0, -0.5,  0.5, 0.25, 3.0,
            -0.5, -0.5, 0.5, 1.0, -0.5, -0.5, 0.50, 5.0,
             0.5, -0.5, 0.5, 1.0,  0.5, -0.5, 0.75, 9.0,
        ];
        for warp in [false, true] {
            let Ok(host) = Host::new(warp) else { continue };
            let root = host.root_signature_for(&[]).unwrap();
            let pso = host.pipeline_gs(&root, &vs, &gs, &ps).unwrap();
            let draw = Draw {
                roots: Vec::new(),
                textures: &[],
                samplers: &[],
                vertices: &vertices,
                stride: 32,
                size: (32, 32),
                mesh_groups: 0,
            };
            let texels = host.draw(&root, &pso, &draw).unwrap();
            for (i, t) in texels.iter().enumerate() {
                let (x, y) = ((i % 32) as f32, (i / 32) as f32);
                let (u, v) = ((x + 0.5) / 16.0 - 1.0, 1.0 - (y + 0.5) / 16.0);
                let inside = u.abs() < 0.5 && v.abs() < 0.5;
                let want = if inside { [u, v, 0.25, 3.0] } else { [0.0; 4] };
                let near = t.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-4);
                assert!(near, "warp {warp} pixel ({x}, {y}): {t:?}, want {want:?}");
            }
        }
    }

    #[test]
    fn vertex_shaders_without_written_outputs_are_refused() {
        for outputs in [
            Vec::new(),
            vec![SigElement {
                usage: 0,
                ..element("SV_Position", 3, 4, 0, 4)
            }],
        ] {
            let mut s = Shader::new(false);
            s.outputs = outputs;
            s.m.terminate(s.entry, Term::Ret);
            assert!(super::rectangle_gs(&s.finish()).is_err());
        }
    }
}
