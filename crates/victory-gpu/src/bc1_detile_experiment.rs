//! Synthetic-only one-mip BC1 detile, with no persistent raw GPU resource.
use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};

const X: [u32; 7] = [3, 5, 6, 9, 11, 13, 15];
const Y: [u32; 6] = [4, 7, 8, 10, 12, 14];

fn spread(c: u32, bits: &[u32]) -> u32 {
    bits.iter().enumerate().fold(0, |n, (i, b)| n | ((c >> i & 1) << b))
}
fn raw_offset(blocks_per_row: u32, x: u32, y: u32) -> u32 {
    (((y >> 6) * blocks_per_row + (x >> 7)) << 16)
        | spread(x, &X) | spread(y, &Y)
}
fn layout(width: u32, height: u32) -> Layout {
    Layout::new(&Desc {
        dimension: Dimension::Tex2d, width, height, depth: 1, mips: 1,
        element_bytes: 8, block: 4, samples: 1, swizzle: Swizzle::SW_64KB_D,
    }).unwrap()
}

#[test]
fn bc1_compute_addresses_match_cpu_detile_even_for_partial_tiles() {
    for (width, height) in [(1,1),(513,257),(1023,517),(2051,1025),(193,1024)] {
        let l = layout(width,height);
        let m = l.mips[0];
        assert_eq!(m.offset,0);
        assert!(m.tail.is_none() || (m.tail.unwrap().x, m.tail.unwrap().y)==(0,0));
        assert_eq!(l.block,[128,64,1]);
        let (bx, by) = (m.width, m.height);
        let raw: Vec<u8> = (0..l.size).map(|i| (i*31+i/193) as u8).collect();
        let pitch=(bx*8).next_multiple_of(256) as usize;
        let size=pitch*(by-1) as usize + bx as usize*8;
        let mut a=vec![0xa5;size];
        let mut b=a.clone();
        l.detile(&raw,0,0,&mut a,pitch,pitch*by as usize).unwrap();
        for y in 0..by {
            for x in 0..bx {
                let src=raw_offset(m.pitch/128,x,y) as usize;
                assert_eq!(src as u64,l.offset(0,x,y,0).unwrap());
                let dst=y as usize*pitch+x as usize*8;
                b[dst..dst+8].copy_from_slice(&raw[src..src+8]);
            }
        }
        assert_eq!(a,b,"{width}x{height}");
    }
}

#[cfg(windows)]
mod windows_fixture {
    use super::*;
    use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_BC1_UNORM, DXGI_SAMPLE_DESC};
    use windows::core::{Interface, s};
    use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
    use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

    fn source() -> String {
        let bits = |name: &str, b: &[u32]| {
            b.iter().enumerate().map(|(i,b)| format!("((({name} >> {i}u) & 1u) << {b}u)"))
                .collect::<Vec<_>>().join(" | ")
        };
        format!(r#"
ByteAddressBuffer input_bytes : register(t0);
RWByteAddressBuffer packed : register(u0);
cbuffer Args : register(b0) {{ uint width, height, blocks_per_row, row_pitch; }};
[RootSignature("SRV(t0), UAV(u0), RootConstants(num32BitConstants=4, b0)")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    uint x=id.x, y=id.y;
    if (x >= width || y >= height) return;
    uint src=((((y>>6u)*blocks_per_row)+(x>>7u))<<16u) | {} | {};
    packed.Store2(y*row_pitch+x*8u, input_bytes.Load2(src));
}}
"#,bits("x",&X),bits("y",&Y))
    }

    fn run() -> windows::core::Result<()> {
        let h=crate::host::host().expect("D3D12 hardware or WARP");
        let dev=&h.device;
        let (width,height)=(1023,517);
        let l=layout(width,height);
        let m=l.mips[0];
        let raw:Vec<u8>=(0..l.size).map(|i| (i*31+i/193) as u8).collect();
        let mut footprint=D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut rows=0;
        let mut row_bytes=0;
        let mut size=0;
        let tdesc=D3D12_RESOURCE_DESC {
            Dimension:D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width:u64::from(width), Height:height,
            DepthOrArraySize:1, MipLevels:1, Format:DXGI_FORMAT_BC1_UNORM,
            SampleDesc:DXGI_SAMPLE_DESC{Count:1,Quality:0}, ..Default::default()
        };
        let bdesc=|n,flags| D3D12_RESOURCE_DESC {
            Dimension:D3D12_RESOURCE_DIMENSION_BUFFER, Width:n, Height:1,
            DepthOrArraySize:1, MipLevels:1, Layout:D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            SampleDesc:DXGI_SAMPLE_DESC{Count:1,Quality:0},Flags:flags,..Default::default()
        };
        // SAFETY: the fixture owns resources, mapped data, command lists and
        // fences, and never releases a resource before its queue work ends.
        unsafe {
            dev.GetCopyableFootprints(&tdesc,0,1,0,Some(&mut footprint),Some(&mut rows),
                Some(&mut row_bytes),Some(&mut size));
            assert_eq!(rows,m.height);
            assert_eq!(row_bytes,u64::from(m.width)*8);
            let pitch=footprint.Footprint.RowPitch;
            assert!(size < 8<<20 && l.size < 8<<20);
            let mut expected=vec![0;size as usize];
            l.detile(&raw,0,0,&mut expected,pitch as usize,(pitch*rows) as usize).unwrap();
            let (input,in_ptr)=h.upload_buffer(l.size).expect("test source upload buffer");
            in_ptr.copy_from_nonoverlapping(raw.as_ptr(),raw.len());

            let create=|heap, desc:&D3D12_RESOURCE_DESC, state| -> windows::core::Result<ID3D12Resource> {
                let mut out=None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES{Type:heap,..Default::default()},
                    D3D12_HEAP_FLAG_NONE,desc,state,None,&mut out)?;
                out.ok_or_else(windows::core::Error::empty)
            };
            let scratch=create(D3D12_HEAP_TYPE_DEFAULT,
                &bdesc(size,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;
            let texture=create(D3D12_HEAP_TYPE_DEFAULT,&tdesc,D3D12_RESOURCE_STATE_COPY_DEST)?;
            let readback=create(D3D12_HEAP_TYPE_READBACK,
                &bdesc(size,D3D12_RESOURCE_FLAG_NONE),D3D12_RESOURCE_STATE_COPY_DEST)?;

            let hlsl=source();
            let mut code:Option<ID3DBlob>=None;
            let mut err:Option<ID3DBlob>=None;
            if let Err(e)=D3DCompile(hlsl.as_ptr().cast(),hlsl.len(),s!("bc1_detile"),None,None,
                s!("main"),s!("cs_5_1"),0,0,&mut code,Some(&mut err)) {
                if let Some(b)=err {
                    eprintln!("{}",String::from_utf8_lossy(std::slice::from_raw_parts(
                        b.GetBufferPointer().cast::<u8>(),b.GetBufferSize())));
                }
                return Err(e);
            }
            let blob=code.expect("compiled");
            let bytes=std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(),blob.GetBufferSize());
            let root:ID3D12RootSignature=dev.CreateRootSignature(0,bytes)?;
            let pso:ID3D12PipelineState=dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC{
                pRootSignature:std::mem::transmute_copy(&root),
                CS:D3D12_SHADER_BYTECODE{pShaderBytecode:bytes.as_ptr().cast(),BytecodeLength:bytes.len()},
                ..Default::default()
            })?;
            let queue:ID3D12CommandQueue=dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC{
                Type:D3D12_COMMAND_LIST_TYPE_DIRECT,..Default::default()
            })?;
            let allocator:ID3D12CommandAllocator=dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list:ID3D12GraphicsCommandList=dev.CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,&allocator,&pso)?;
            let fence:ID3D12Fence=dev.CreateFence(0,D3D12_FENCE_FLAG_NONE)?;
            let submit=|tag| -> windows::core::Result<()> {
                list.Close()?;
                let base:ID3D12CommandList=list.cast()?;
                queue.ExecuteCommandLists(&[Some(base)]);
                queue.Signal(&fence,tag)?;
                let event=CreateEventW(std::ptr::null(),0,0,std::ptr::null());
                assert!(!event.is_null());
                let wait=fence.SetEventOnCompletion(tag,windows::Win32::Foundation::HANDLE(event as _));
                let done=if wait.is_ok() { WaitForSingleObject(event,10_000) } else { u32::MAX };
                CloseHandle(event);
                wait?;
                assert_eq!(done,WAIT_OBJECT_0);
                dev.GetDeviceRemovedReason()
            };
            let placed=|r:&ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION{
                pResource:std::mem::transmute_copy(r),
                Type:D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous:D3D12_TEXTURE_COPY_LOCATION_0{PlacedFootprint:footprint},
            };
            let sub=D3D12_TEXTURE_COPY_LOCATION{
                pResource:std::mem::transmute_copy(&texture),
                Type:D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous:D3D12_TEXTURE_COPY_LOCATION_0{SubresourceIndex:0},
            };
            list.SetComputeRootSignature(&root);
            list.SetComputeRootShaderResourceView(0,input.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(1,scratch.GetGPUVirtualAddress());
            let args=[m.width,m.height,m.pitch/128,pitch];
            list.SetComputeRoot32BitConstants(2,4,args.as_ptr().cast(),0);
            list.Dispatch(m.width.div_ceil(8),m.height.div_ceil(8),1);
            list.ResourceBarrier(&[crate::host::transition(&scratch,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            list.CopyTextureRegion(&sub,0,0,0,&placed(&scratch),None);
            list.ResourceBarrier(&[crate::host::transition(&texture,D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            submit(1)?;
            // Only the finished BC1 texture is retained from the first flight.
            drop((input,scratch));
            allocator.Reset()?;
            list.Reset(&allocator,None::<&ID3D12PipelineState>)?;
            list.CopyTextureRegion(&placed(&readback),0,0,0,&sub,None);
            submit(2)?;
            let mut ptr=std::ptr::null_mut();
            readback.Map(0,None,Some(&mut ptr))?;
            let got=std::slice::from_raw_parts(ptr.cast::<u8>(),size as usize);
            for y in 0..rows as usize {
                let start=y*pitch as usize;
                assert_eq!(got[start..start+row_bytes as usize],expected[start..start+row_bytes as usize]);
            }
            readback.Unmap(0,None);
        }
        Ok(())
    }

    #[test]
    #[ignore = "native D3D12 device required; run on WARP or hardware"]
    fn bc1_texture_survives_dropping_raw_upload_and_scratch() {
        run().unwrap();
    }
}
