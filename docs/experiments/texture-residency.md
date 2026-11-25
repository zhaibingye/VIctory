# Texture residency experiment

This branch adds diagnostics and synthetic test fixtures only. It does not
change the production texture allocation or upload path. There is no raw
VRAM reserve and no new default setting.

- With the diagnostics feature, the frame log separates logical texture
  copy payload from CPU staging work. A table copy without a refresh at
  that step is not necessarily redundant. This counter is not a PCIe or
  hardware timing measurement.
- The BC1 test uses synthetic blocks and a single mip of a 2D 64 KiB D
  layout. It verifies a native texture remains correct after a fence and
  release of the raw source upload and temporary detile output. The
  portable test independently checks the address calculation against the
  CPU converter. No proprietary inputs are used.

On a Windows D3D12 backend, to run the explicit GPU fixture:

```powershell
cargo test --locked -p victory-gpu bc1_texture_survives_dropping_raw_upload_and_scratch -- --ignored
```

This branch is a baseline for measuring copies on demand, not a performance
claim. Before enabling any new runtime path, compare the same scene/trace
at equal residency budgets, count how many bytes are requested for copies,
time the GPU work separately, and check that resources are not recycled
before a GPU fence or invalidated across alias/remap generations.
