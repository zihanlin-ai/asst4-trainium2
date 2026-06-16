import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax            # 128: partition limit
    n_tiles_c_in = in_channels // c_in_pmax  # number of 128-wide input-channel tiles
    n_tiles_c_out = out_channels // c_in_pmax # number of 128-wide output-channel tiles

    # ------------------------------------------------------------------
    # STEP D: any image size, with bias, with fused maxpool (pool_size 1 or 2).
    # pool_size == 1: write each conv row directly (Step C behaviour).
    # pool_size == 2: compute two conv rows, max them (vertical), then max
    #   adjacent column pairs (horizontal) -> one pooled row, all on-chip.
    # ------------------------------------------------------------------

    # --- Preprocess weights into one SBUF tensor (reused by all b/rows) ---
    # W is (out_ch, in_ch, i, j). nc_matmul computes lhsT.T @ rhs, and we want
    #   out(out_ch, px) = W2d(out_ch, in_ch) @ X(in_ch, px),
    # so lhsT must be W2d.T = (in_ch, out_ch). We transpose each 128x128 block.
    # Stored as one SBUF tensor; the (i,j,oc_t,ic_t) block is a (128 in, 128 out)
    # slice, indexed with the (symbolic) loop vars below.
    w_lhsT = nl.ndarray(
        (c_in_pmax, filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, c_in_pmax),
        dtype=W.dtype, buffer=nl.sbuf,
    )
    for i in nl.affine_range(filter_height):
        for j in nl.affine_range(filter_width):
            for oc_t in nl.affine_range(n_tiles_c_out):
                for ic_t in nl.affine_range(n_tiles_c_in):
                    # load W[oc_block, ic_block, i, j] -> (128 out, 128 in)
                    w_sb = nl.ndarray((c_in_pmax, c_in_pmax), dtype=W.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        src=W[oc_t * c_in_pmax:(oc_t + 1) * c_in_pmax,
                              ic_t * c_in_pmax:(ic_t + 1) * c_in_pmax, i, j],
                        dst=w_sb,
                    )
                    # transpose -> (128 in, 128 out) in PSUM, then copy to SBUF slice
                    w_t = nisa.nc_transpose(w_sb)
                    w_lhsT[:, i, j, oc_t, ic_t, :] = nl.copy(w_t, dtype=W.dtype)

    # --- Preprocess bias: one (128, 1) per output-channel tile, loaded once ---
    # bias[oc] is a per-output-channel scalar; shape (128, 1) so tensor_scalar
    # broadcasts it across all pixels of that channel.
    bias_r = bias.reshape((out_channels, 1))
    bias_sb = nl.ndarray((c_in_pmax, n_tiles_c_out, 1), dtype=bias.dtype, buffer=nl.sbuf)
    for oc_t in nl.affine_range(n_tiles_c_out):
        nisa.dma_copy(
            src=bias_r[oc_t * c_in_pmax:(oc_t + 1) * c_in_pmax, :],
            dst=bias_sb[:, oc_t, :],
        )

    # Unified structure: always process TWO conv rows per iteration. One matmul
    # computes both rows at once via a >2D moving operand (128, 2, out_width).
    # Branch only at the output: pool_size==1 writes both rows; ==2 maxpools them.
    assert out_height % 2 == 0, "this kernel processes two conv rows at a time"
    ROWS = 2                          # conv rows per iteration (= pool window height)
    band_rows = filter_height + 1     # input rows needed for two consecutive conv rows
    n_row_pairs = out_height // ROWS

    for b in nl.affine_range(batch_size):
        for rp in nl.affine_range(n_row_pairs):
            # band: input rows [ROWS*rp, ROWS*rp + band_rows) across all channels
            band = nl.ndarray(
                (c_in_pmax, n_tiles_c_in, band_rows, input_width),
                dtype=X.dtype, buffer=nl.sbuf,
            )
            for ic_t in nl.affine_range(n_tiles_c_in):
                nisa.dma_copy(
                    src=X[b, ic_t * c_in_pmax:(ic_t + 1) * c_in_pmax,
                          ROWS * rp:ROWS * rp + band_rows, :],
                    dst=band[:, ic_t, :, :],
                )

            for oc_t in nl.affine_range(n_tiles_c_out):
                # one PSUM holding BOTH conv rows: (128 out_ch, 2, out_width)
                psum = nl.zeros((c_in_pmax, ROWS, out_width), dtype=nl.float32, buffer=nl.psum)
                for ic_t in nl.affine_range(n_tiles_c_in):
                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            lhsT = w_lhsT[:, i, j, oc_t, ic_t, :]
                            # >2D moving: band rows [i, i+2) -> conv rows 2rp and 2rp+1
                            moving = band[:, ic_t, i:i + ROWS, j:j + out_width]
                            psum += nisa.nc_matmul(lhsT, moving)   # (128, 2, out_width)
                both = nl.copy(psum, dtype=X.dtype)                # (128, 2, out_width)

                if pool_size == 1:
                    # write the two conv rows directly (+bias each)
                    rowA = nisa.tensor_scalar(both[:, 0, :], nl.add, bias_sb[:, oc_t, :])
                    rowB = nisa.tensor_scalar(both[:, 1, :], nl.add, bias_sb[:, oc_t, :])
                    nisa.dma_copy(src=rowA,
                                  dst=X_out[b, oc_t * c_in_pmax:(oc_t + 1) * c_in_pmax, ROWS * rp, :])
                    nisa.dma_copy(src=rowB,
                                  dst=X_out[b, oc_t * c_in_pmax:(oc_t + 1) * c_in_pmax, ROWS * rp + 1, :])
                else:
                    # 2x2 maxpool: vertical max (the two rows), then horizontal (col pairs)
                    vmax = nisa.tensor_tensor(both[:, 0, :], both[:, 1, :], op=nl.maximum)
                    vmax3 = vmax.reshape((c_in_pmax, out_pool_width, pool_size))
                    pooled = nisa.tensor_tensor(vmax3[:, :, 0], vmax3[:, :, 1], op=nl.maximum)
                    pooled = nisa.tensor_scalar(pooled, nl.add, bias_sb[:, oc_t, :])
                    nisa.dma_copy(src=pooled,
                                  dst=X_out[b, oc_t * c_in_pmax:(oc_t + 1) * c_in_pmax, rp, :])

    return X_out

