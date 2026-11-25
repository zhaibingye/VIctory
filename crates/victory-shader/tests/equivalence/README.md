# Wave32 execution equivalence

This native CPU integration test compares all 32 dwords written by a final
`v_mov_b32 v2,v0`. The source must reach that observer once after restoring
full EXEC. Distinct initial values are assigned to v0, v1, and v2. The checker
does not compare intermediate EXEC, VCC, SCC, scalar scratch, branch traces,
or instruction counts. A wave can execute a block that some structured lanes skip.

## Language and independence

Programs concatenate arithmetic, uniform two-arm branches, masked regions,
and counted loops. A loop has one through six trips and a nonempty body that
can contain branches and masked regions. There are no nested loops. Inside a
narrowed EXEC region the only payload is per-lane arithmetic. It can execute
under a guard or follow a forward EXECZ skip. Every region restores its saved
EXEC. Inline integers are in 0..=64. Word addition wraps modulo 2^32. The
unsigned vector compare forms a mask from the lanes satisfying threshold < v0.
A mask can also come from a supplied scalar or a uniform join of those two sources.

- v0 is the accumulator, v1 is input data, and v2 is the output. The vector
  operations set a constant, add a constant, add v1, or xor with v1.
- s0 is the arbitrary 32-bit mask, s1 selects a branch, s2 counts loop trips,
  s3 saves EXEC, and s4 carries a joined mask. EXEC initially includes every
  lane. Scratch SGPRs, VCC and SCC start at zero.
- The source executor interprets typed instructions with absolute branch targets.
  It holds scalar dwords and full arrays of lane values, and updates only active
  lanes. Encoding translates the source instructions to little-endian GFX10
  words with signed relative branch displacements.
- The SSA evaluator consumes `ir::build(bytes, false, 3)`, the decoded fields,
  actual operand bindings, phis, predecessor lists and exits. The structured
  evaluator consumes `structure::structure(&program)`, walking individual lanes
  through the returned exits and guards with the boolean-node and mask-phi
  mappings. Neither transformed evaluator sees the source instructions.
- Shared pieces are the input fixture, observation contract and standard Rust
  integer operations. The generator and reference share the source instruction
  stream. Hand examples check what that stream means. This is differential
  coverage of the named ISA subset, not an independent certification of the ISA.

The executors bound dynamic execution to 1,024 source instructions or 1,024
block visits (per lane for structured execution). Floating point, Wave64, memory,
textures, barriers, derivatives, discards, cross-lane instructions, arbitrary
scalar side effects in divergent regions, mask-as-data uses and emitted host
shader code are outside the grammar.

## Run and replay

From the checkout with its pinned Rust toolchain:

```sh
cargo test -p victory-shader --locked --test equivalence
cargo test -p victory-shader --locked --test equivalence --release search::campaign -- --exact --ignored --nocapture
WAVE_SEED=42 WAVE_COUNT=1 cargo test -p victory-shader --locked --test equivalence --release search::campaign -- --exact --ignored --nocapture
```

Seeds are unsigned decimal integers. Each seed reconstructs the entire program
and its inputs from SplitMix64. The default run starts at zero and executes up to
10,000 consecutive seeds or 30 minutes, checking the deadline between bounded
cases. It prints the first seed, counts and elapsed search time. The final
outcomes distinguish agreement, compiler refusal, checker-unsupported behavior,
invalid evaluation, compiler errors, exhaustion, panic and semantic difference.
The separate SSA count includes programs whose SSA comparison passed before a
later-stage failure. A mismatch prints its seed, stage, first differing lane,
expected and actual dwords, and reduced regions using the original seed's inputs.

Reduction rebuilds branch targets from the grammar after every edit. It removes
regions and masked-body operations and decreases loop bounds while retaining the
same stage, lane, expected dword and actual dword. It produces a local minimum
under those edits. A controlled guard fault tests the reducer even when a campaign
finds no compiler mismatch. Other in-memory mutations exercise the SSA loop
backedge, inactive-lane masking and the structured branch destinations.
