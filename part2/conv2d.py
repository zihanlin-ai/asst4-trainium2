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
    # STEP A: small image, no bias, no maxpool (pool_size == 1).
    # The whole image fits in SBUF, so no row-banding yet.
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

    # --- Main loop: one output row at a time ---
    for b in nl.affine_range(batch_size):
        # load the whole image X[b] into SBUF: (128 in_ch, n_ic_tiles, H, W_in)
        x_sb = nl.ndarray(
            (c_in_pmax, n_tiles_c_in, input_height, input_width),
            dtype=X.dtype, buffer=nl.sbuf,
        )
        for ic_t in nl.affine_range(n_tiles_c_in):
            nisa.dma_copy(
                src=X[b, ic_t * c_in_pmax:(ic_t + 1) * c_in_pmax, :, :],
                dst=x_sb[:, ic_t, :, :],
            )

        for oc_t in nl.affine_range(n_tiles_c_out):
            for out_row in nl.affine_range(out_height):
                # accumulator for one output row: (128 out_ch, out_width) in PSUM
                psum = nl.zeros((c_in_pmax, out_width), dtype=nl.float32, buffer=nl.psum)
                # accumulate over input-channel tiles and all filter positions
                for ic_t in nl.affine_range(n_tiles_c_in):
                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            # shifted input slice: (128 in_ch, out_width)
                            rhs = x_sb[:, ic_t, out_row + i, j:j + out_width]
                            psum += nisa.nc_matmul(w_lhsT[:, i, j, oc_t, ic_t, :], rhs)
                # PSUM -> SBUF (cast to output dtype), then store this row to HBM
                out_sb = nl.copy(psum, dtype=X.dtype)
                nisa.dma_copy(
                    src=out_sb,
                    dst=X_out[b, oc_t * c_in_pmax:(oc_t + 1) * c_in_pmax, out_row, :],
                )

    return X_out

