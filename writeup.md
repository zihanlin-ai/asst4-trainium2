## Part 1: Vector Add and Matrix Transpose

### Step 1: Chunking Vectors to Parallelize Across 128 Compute Lanes (6 points)

**Q1.** Run `vector_add_tiled` with *row_chunk = 1* on a vector size of 25600
(`python run_benchmark.py --kernel tiled -n 25600`). What was the execution time
in microseconds (μs)?

**Answer:**

- `row_chunk = 1`, n = 25600 → **38216 μs** (≈ 38.2 ms)

---

**Q2.** Change `vector_add_tiled` to use *row_chunk = 128*. Record the execution
time (μs) on a vector size of 25600. How much faster is *row_chunk = 128* compared
to *row_chunk = 1*? Why do you think it is faster?
*(Hint: think of execution as loading `ROW_CHUNK` elements in parallel from HBM,
then doing a `ROW_CHUNK`-wide vector add in SBUF.)*

**Answer:**

| ROW_CHUNK | Time (μs) on n=25600 |
|---|---|
| 1 | 38216 |
| 128 | 402 |

- Speedup (1 → 128): **≈ 95×** (38216 / 402)
- Why faster: two effects compound, both stemming from the fact that the
  per-operation cost is roughly **fixed regardless of how much data the
  operation touches**:
  1. **Full 128-lane parallelism.** The Vector Engine is a 128-partition SIMD
     unit whose lanes are clocked together, so a `tensor_scalar` add on a
     `(128,1)` tile takes essentially the same time as on a `(1,1)` tile but
     produces 128 results instead of 1. With `ROW_CHUNK=1`, 127 of 128 lanes
     sit idle every instruction.
  2. **128× fewer DMA transfers.** Each `nisa.dma_copy` pays a fixed
     setup/descriptor overhead independent of size. `ROW_CHUNK=1` issues 25600
     tiny DMAs (overhead-dominated); `ROW_CHUNK=128` issues only
     `25600/128 = 200`, amortizing that overhead over 128 elements each.

  This is visible in the per-iteration cost: 38216 μs / 25600 ≈ 1.49 μs per
  iteration vs 402 μs / 200 ≈ 2.0 μs — roughly constant whether an iteration
  processes 1 or 128 elements. So the win comes almost entirely from doing 128×
  fewer iterations. (The speedup is ~95×, short of an ideal 128×, due to
  residual compile/loop-control/bandwidth overheads.)

---

**Q3.** Try running `vector_add_tiled` with *row_chunk = 256* on a vector size of
25600. You should see an error. In one sentence, explain why.

**Answer:** `ROW_CHUNK=256` tries to allocate an SBUF tile with 256 partitions
(`nl.ndarray((ROW_CHUNK, 1), ..., buffer=nl.sbuf)`), but a NeuronCore's SBUF has a
hard architectural limit of 128 partitions on the first (partition) dimension, so
allocation fails with `ValueError: number of partitions 256 exceed architecture
limitation of 128`.

---

### Step 2a: Improved Data Streaming (4 points)

**Q1.** Run `vector_add_stream` with *FREE_DIM = 2*. How many μs did it take on a
vector size of 25600? How much faster is this than `vector_add_tiled` with
*row_chunk = 128* from Step 1?

**Answer:**

- `stream`, FREE_DIM = 2, n = 25600 → **196 μs**
- Speedup vs `tiled` (row_chunk=128, 402 μs): **≈ 2.05×** (402 / 196)

---

**Q2.** Change *FREE_DIM* to reduce the number of DMA transfers as much as possible
on a vector of size 25600. What value of *FREE_DIM* did you choose, and what was the
execution time (μs)? How much faster is your chosen *FREE_DIM* than *FREE_DIM = 2*?
How much faster than `vector_add_tiled` with *row_chunk = 128*?

**Answer:**

- Chosen FREE_DIM: **200**. Reasoning: the vector is reshaped to
  `(PARTITION_DIM, M/PARTITION_DIM) = (128, 25600/128) = (128, 200)`, so the free
  dimension has 200 columns. Setting `FREE_DIM = 200` makes the loop run
  `25600 / (128 * 200) = 1` iteration — the whole `(128, 200)` tile is loaded in a
  single DMA per input, which is the minimum possible number of DMA transfers.
  (`FREE_DIM` must divide 200, so the largest legal value is 200 itself.)
- Time (μs): **15 μs**
- Speedup vs FREE_DIM = 2 (196 μs): **≈ 13×** (196 / 15)
- Speedup vs `tiled` row_chunk=128 (402 μs): **≈ 27×** (402 / 15)

**Supporting sweep** (powers of two + the divisor 200; `FREE_DIM` must divide 200,
so 16/32/64/128 are skipped — they drop the uncovered tail columns and fail
correctness):

| FREE_DIM | loop iters | time (μs) |
|---|---|---|
| 1 | 200 | 388 |
| 2 | 100 | 196 |
| 4 | 50 | 103 |
| 8 | 25 | 57 |
| 200 | 1 | 15 |

The trend has two regimes: in the overhead-dominated region (FREE_DIM 1→8) the
time roughly halves each time FREE_DIM doubles, i.e. time ∝ iterations ∝
1/FREE_DIM, because runtime is dominated by per-DMA/per-instruction fixed overhead
paid once per iteration. Past that, returns diminish (8→200 cuts iterations 25×
but time only ~3.8×) as the cost floor shifts to the actual data movement and
vector-add work rather than fixed overhead.

---

### Step 2b: Learning to Use Neuron-Profile (5 points)

**Q1.** Generate profiles for *FREE_DIM = 2000* and *FREE_DIM = 1000* on a vector
size of 256000, saving `.neff` / `.ntff` files (`stream_256k_fd2k`,
`stream_256k_fd1k`). *(No written answer required — this step produces the trace
files used below.)*

**Answer:** Four trace files generated:

- `stream_256k_fd2k.neff`, `stream_256k_fd2k.ntff` (FREE_DIM = 2000)
- `stream_256k_fd1k.neff`, `stream_256k_fd1k.ntff` (FREE_DIM = 1000)

Benchmark wall-clock (from `run_benchmark.py`): FREE_DIM=2000 → 25 μs, FREE_DIM=1000 → 23 μs.

---

**Q2.** Using `neuron-profile view --output-format summary-text ...`, report:

**Answer:**

| FREE_DIM | total_time (s) | dma_transfer_count |
|---|---|---|
| 2000 | 2.8340789e-05 (≈ 28.3 μs) | 3 |
| 1000 | 2.6665792e-05 (≈ 26.7 μs) | 6 |

The key observation: **FREE_DIM = 1000 is faster (26.7 μs) despite issuing twice as
many DMA transfers (6 vs 3).** DMA counts match the loop structure — FREE_DIM=2000
runs 1 iteration (2 loads + 1 store = 3 DMAs); FREE_DIM=1000 runs 2 iterations
(4 loads + 2 stores = 6 DMAs).

---

**Q3.** Launch the GUI analytics for both *FREE_DIM = 2000* and *FREE_DIM = 1000*
and inspect the engine timeline. *(No written answer required — observations feed
into Q4. Optional: attach screenshots.)*

**Answer:** _(optional observations / screenshots)_

---

**Q4.** Briefly explain why *FREE_DIM = 1000* has a faster execution time than
*FREE_DIM = 2000* even though it required more DMA transfers. *(Hint: pipelining.)*

**Answer:** _(TODO — overlap of DMA and Vector Engine; smaller tiles let the next
DMA start while the current tile is being added, vs one giant non-overlapped
transfer. Reference the Pending DMA Count / DMA Throughput / VectorE lanes.)_

---

### Step 3: Matrix Transpose (15 points = 10 coding + 5 writeup, +1 EC)

> Code deliverable: implement `matrix_transpose` in `part1/kernels.py` using only
> `nisa.nc_transpose` for compute (memory ops like `dma_copy` / `tensor_copy` are
> allowed). Tiles are ≤ 128×128; M and N are both divisible by 128.

**Q1.** Test on a 1024×1024 matrix (`python run_benchmark.py --kernel transpose -n 1024`)
and record the execution time (μs).

**Answer:** _(TODO)_

- `transpose`, 1024×1024 → **___ μs**

---

**Q2.** Without using a profiler, do you think your kernel is memory-bound or
compute-bound? Explain. Then confirm by profiling (as with `vector_add_stream`).
You may include a screenshot, but provide a written description of how it validates
your answer.

**Answer:** _(TODO)_

- Prediction (before profiling): **memory-bound / compute-bound** because _(TODO)_
- Profiling evidence: _(TODO — engine utilization, DMA throughput vs Tensor Engine
  occupancy; describe what the trace shows.)_

---

**Q3 (Extra credit, 1 pt).** Optimize to minimize latency; target **< 700 μs on a
4096×4096 transpose**. Measure *without* `--profile_name`. Submit a brief writeup
explaining how you identified bottlenecks and addressed them.

**Answer:** _(TODO if attempted)_

- 4096×4096 latency (no profiler): **___ μs**
- Bottlenecks identified: _(TODO)_
- Optimizations applied: _(TODO — tiling strategy, DMA batching, double-buffering,
  alternative APIs, …)_

---

## Part 2: Fused Convolution + Max Pool Layer

> Code deliverable: implement `fused_conv2d_maxpool` in `part2/conv2d.py`.
> Test with `python3 test_harness.py` (add `--test_maxpool` for the fused maxpool
> cases). Correctness must pass on small (32×16) and large (224×224) images, with
> bias, and with maxpool. Performance is tested in float16 and float32.

### Implementation Description

**Answer:** _(TODO — describe the algorithm and how convolution is mapped to matrix
multiplication; how the loops are structured (filter h/w, tiled matmul, batching);
which output dimension is tiled and the resulting input tile shapes; where bias and
maxpool fuse in; any reshape/transpose of inputs and why — recall NKI's matmul
transposes its first ("stationary") input.)_

### Optimization

**Answer:** _(TODO — describe how you optimized: loop ordering for PSUM/SBUF
locality, keeping intermediates in PSUM until a tile completes, double-buffering,
DMA amortization, profiler-guided changes to fill Tensor Engine idle gaps, etc.)_

### Profiling — Achieved MFU

Report the achieved MFU (Model FLOPs Utilization) for both data types. Capture a
trace with `--profile <name>`, open it with
`neuron-profile view -n <name>.neff -s <name>.ntff`, and read the **Cumulative
Utilization** at the very end of the Estimated MFU section.

**Answer:** _(TODO)_

| Data type | Achieved MFU |
|---|---|
| float16 | ___ % |
| float32 | ___ % |

### Extra Credit — Smaller Images

Run `neuron-profile` again on smaller images. Is there a difference in MFU between
smaller and larger images? If so, how would you optimize the fused convolution layer
for smaller images? *(Hint: `nisa.nc_matmul` can accept a >2D tensor as the `moving`
argument, as long as PSUM hardware constraints are respected.)*

**Answer:** _(TODO if attempted)_

- MFU on smaller vs larger images: _(TODO)_
- Approach for smaller images: _(TODO)_

---

## Appendix: Environment Notes

- Instance type: `trn2.3xlarge`, region `ap-southeast-4` (Melbourne).
- All timings recorded with the install-script virtualenv active.
- Performance numbers (Step 3 EC, Part 2) measured **without** `--profile` /
  `--profile_name`, since profiling perturbs execution time.
