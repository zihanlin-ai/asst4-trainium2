# Assignment 4: Programming a Machine Learning Accelerator — Writeup

> Submission note: the final hand-in must be `writeup.pdf`. Draft answers here in
> markdown, then export to PDF before submitting (e.g. via VS Code, pandoc, or a
> browser print-to-PDF). Code deliverables are `part1/kernels.py` (transpose) and
> `part2/conv2d.py` (fused conv+maxpool).
>
> Point budget for the writeup: **Part 1 Questions = 20 pts, Part 2 Questions = 10 pts.**

---

## Part 1: Vector Add and Matrix Transpose

### Step 1: Chunking Vectors to Parallelize Across 128 Compute Lanes (6 points)

**Q1.** Run `vector_add_tiled` with *row_chunk = 1* on a vector size of 25600
(`python run_benchmark.py --kernel tiled -n 25600`). What was the execution time
in microseconds (μs)?

**Answer:** _(TODO)_

- `row_chunk = 1`, n = 25600 → **___ μs**

---

**Q2.** Change `vector_add_tiled` to use *row_chunk = 128*. Record the execution
time (μs) on a vector size of 25600. How much faster is *row_chunk = 128* compared
to *row_chunk = 1*? Why do you think it is faster?
*(Hint: think of execution as loading `ROW_CHUNK` elements in parallel from HBM,
then doing a `ROW_CHUNK`-wide vector add in SBUF.)*

**Answer:** _(TODO)_

| ROW_CHUNK | Time (μs) on n=25600 |
|---|---|
| 1 | ___ |
| 128 | ___ |

- Speedup (1 → 128): **___×**
- Why faster: _(TODO — partitions / 128 parallel lanes / fewer iterations / DMA amortization …)_

---

**Q3.** Try running `vector_add_tiled` with *row_chunk = 256* on a vector size of
25600. You should see an error. In one sentence, explain why.

**Answer:** _(TODO — relate to the 128 maximum partition dimension on a NeuronCore.)_

---

### Step 2a: Improved Data Streaming (4 points)

**Q1.** Run `vector_add_stream` with *FREE_DIM = 2*. How many μs did it take on a
vector size of 25600? How much faster is this than `vector_add_tiled` with
*row_chunk = 128* from Step 1?

**Answer:** _(TODO)_

- `stream`, FREE_DIM = 2, n = 25600 → **___ μs**
- Speedup vs `tiled` (row_chunk=128): **___×**

---

**Q2.** Change *FREE_DIM* to reduce the number of DMA transfers as much as possible
on a vector of size 25600. What value of *FREE_DIM* did you choose, and what was the
execution time (μs)? How much faster is your chosen *FREE_DIM* than *FREE_DIM = 2*?
How much faster than `vector_add_tiled` with *row_chunk = 128*?

**Answer:** _(TODO)_

- Chosen FREE_DIM: **___** (reasoning: n=25600 / PARTITION_DIM=128 = 200 columns ⇒ FREE_DIM = ___ )
- Time (μs): **___**
- Speedup vs FREE_DIM = 2: **___×**
- Speedup vs `tiled` row_chunk=128: **___×**

---

### Step 2b: Learning to Use Neuron-Profile (5 points)

**Q1.** Generate profiles for *FREE_DIM = 2000* and *FREE_DIM = 1000* on a vector
size of 256000, saving `.neff` / `.ntff` files (`stream_256k_fd2k`,
`stream_256k_fd1k`). *(No written answer required — this step produces the trace
files used below.)*

**Answer:** _(files generated — list filenames as a sanity check)_

- _(TODO: stream_256k_fd2k.neff / .ntff, stream_256k_fd1k.neff / .ntff)_

---

**Q2.** Using `neuron-profile view --output-format summary-text ...`, report:

**Answer:** _(TODO)_

| FREE_DIM | total_time (s) | dma_transfer_count |
|---|---|---|
| 2000 | ___ | ___ |
| 1000 | ___ | ___ |

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
