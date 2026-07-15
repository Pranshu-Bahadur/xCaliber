# xCaliber TODO

Live design: [`co-design.md`](co-design.md)

## Now

### Layout-agnostic paneled weights

- [ ] Remove the prepacked W13/S13 + W2/S2 checkpoint assumption.
- [ ] Define one canonical source W4 + scale layout; checkpoint bytes stay in
      that layout.
- [ ] Repack only the next compute panel into the existing lane-native layout:

```text
source W1/W3 + scales -> FF1 panel -> W13/S13 consumer
source W2    + scales -> FF2 panel -> W2/S2 consumer
```

- [ ] Keep storage bounded to panel buffers; do not materialize a second
      kernel-native copy of the full model.
- [ ] Double-buffer panels: transform panel `j+1` while compute consumes panel
      `j`.
- [ ] Try graph-side conversion first; compare with in-CTA register/SMEM
      swizzle only if the extra global panel write loses.
- [ ] Preserve the current consumer packet contract and direct FF2
      `cp.reduce`; layout conversion must not leak indexing into the hot MMA
      loop.

Proof contract:

```text
same model shapes / routes / source weights / outputs
native-prepacked vs layout-agnostic paneled
timing includes source-layout read + repack + FF1 + FF2
no hidden offline conversion
report panel bytes, conversion ms, total ms, DRAM/L2, regs, SMEM, spills
```

- [ ] Quantify how much of the current win is NVFP4 MMA, routing/compaction,
      and native checkpoint layout.
- [ ] Check whether repack can overlap at all while the current path already
      runs near the DRAM roofline.

Pass:

```text
bounded memory
numerically matched
no spills
end-to-end advantage survives the conversion tax
```
