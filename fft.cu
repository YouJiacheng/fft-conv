#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mma.h>

#include "fft.h"


namespace cg = cooperative_groups;

template <typename Use, typename Layout = void>
using fragment16 = nvcuda::wmma::fragment<Use, 16, 16, 16, half, Layout>;

using nvcuda::wmma::matrix_a;
using nvcuda::wmma::matrix_b;

using nvcuda::wmma::accumulator;

using nvcuda::wmma::row_major;

using nvcuda::wmma::col_major;

using nvcuda::wmma::layout_t;

template <typename Layout>
__device__ inline void complex_matmul(
    fragment16<matrix_a, row_major> &a_re,
    fragment16<matrix_a, row_major> &a_im,
    fragment16<matrix_b, Layout> &b_re,
    fragment16<matrix_b, Layout> &b_im,
    fragment16<accumulator> &out_re,
    fragment16<accumulator> &out_im)
{
    // no need to use nvcuda::wmma::fill_fragment because ADL
    fill_fragment(out_re, 0.0);
    fill_fragment(out_im, 0.0);

    // out_re = - a_im @ b_im + a_re @ b_re
    mma_sync(out_re, a_im, b_im, out_re);
    for (int i = 0; i < out_re.num_elements; i++)
        out_re.x[i] = -out_re.x[i];
    mma_sync(out_re, a_re, b_re, out_re);

    // out_im = a_re @ b_im + a_im @ b_re
    mma_sync(out_im, a_re, b_im, out_im);
    mma_sync(out_im, a_im, b_re, out_im);
}

__device__ inline half2 complex_mul(half2 a, half2 b)
{
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

constexpr auto pi = 3.141592653589793115997963468544185161590576171875f;

__global__ void digit_reversed_cooley_tukey_16_16_16(
    half *gmem_x_re,
    half *gmem_x_im,
    const half *gmem_F_re,
    const half *gmem_F_im,
    const half *gmem_idx_row,
    const half *gmem_idx_col)
{
    constexpr unsigned int NUM_WARPS = 16;

    constexpr unsigned int N = 4096;
    constexpr unsigned int R = 16;
    constexpr unsigned int PAD1 = 1;
    constexpr unsigned int PAD2 = 8;

    auto block = cg::this_thread_block();
    assert(block.size() / 32 == NUM_WARPS);
    auto warp_id = block.thread_rank() / 32;

    __shared__ half x_re[R][R + PAD1][R + PAD2];
    __shared__ half x_im[R][R + PAD1][R + PAD2];

    fragment16<matrix_a, row_major> F_re;
    fragment16<matrix_a, row_major> F_im;
    load_matrix_sync(F_re, gmem_F_re, R);
    load_matrix_sync(F_im, gmem_F_im, R);

    fragment16<accumulator> idx_row;
    fragment16<accumulator> idx_col;
    load_matrix_sync(idx_row, gmem_idx_row, R, layout_t::mem_row_major);
    load_matrix_sync(idx_col, gmem_idx_col, R, layout_t::mem_row_major);

    {
        // x = einsum('ij,jbc->ibc', F.T, x)
        unsigned int b = warp_id;
        // perform x[:, b, :] = einsum('ij,jc->ic', F.T, x[:, b, :])
        // sum along col
        // cols are independent
        fragment16<matrix_b, row_major> frag_x_re;
        fragment16<matrix_b, row_major> frag_x_im;
        fragment16<accumulator> frag_out_re;
        fragment16<accumulator> frag_out_im;

        load_matrix_sync(frag_x_re, gmem_x_re + b * R, R * R);
        load_matrix_sync(frag_x_im, gmem_x_im + b * R, R * R);
        complex_matmul(F_re, F_im, frag_x_re, frag_x_im, frag_out_re, frag_out_im);

        constexpr auto Ns = N / R;
        constexpr auto unit = 2 * pi / (R * Ns);
#pragma unroll
        for (int i = 0; i < frag_out_re.num_elements; i++)
        {
            auto row = static_cast<unsigned int>(idx_row.x[i]);
            auto col = static_cast<unsigned int>(idx_col.x[i]);
            unsigned int idx = (row * R * R) + (b * R) + col; // [:, b, :]
            // idx = idx % (R * Ns); // idx < N, no-op
            auto t = static_cast<float>((idx / Ns) * (idx % Ns)) * unit;
            auto v = complex_mul({frag_out_re.x[i], frag_out_im.x[i]}, {__cosf(t), -__sinf(t)});
            frag_out_re.x[i] = v.x;
            frag_out_im.x[i] = v.y;
        }

        store_matrix_sync(&x_re[0][b][0], frag_out_re, (R + PAD1) * (R + PAD2), layout_t::mem_row_major);
        store_matrix_sync(&x_im[0][b][0], frag_out_im, (R + PAD1) * (R + PAD2), layout_t::mem_row_major);
    }

    block.sync();

    {
        // x = einsum('ij,bjc->bic', F.T, x)
        unsigned int b = warp_id;
        // perform x[b, :, :] = einsum('ij,jc->ic', F.T, x[b, :, :])
        // sum along col
        // cols are independent
        fragment16<matrix_b, row_major> frag_x_re;
        fragment16<matrix_b, row_major> frag_x_im;
        fragment16<accumulator> frag_out_re;
        fragment16<accumulator> frag_out_im;

        load_matrix_sync(frag_x_re, &x_re[b][0][0], R + PAD2);
        load_matrix_sync(frag_x_im, &x_im[b][0][0], R + PAD2);
        complex_matmul(F_re, F_im, frag_x_re, frag_x_im, frag_out_re, frag_out_im);

        constexpr auto Ns = N / R / R;
        constexpr auto unit = 2 * pi / (R * Ns);
#pragma unroll
        for (int i = 0; i < frag_out_re.num_elements; i++)
        {
            auto row = static_cast<unsigned int>(idx_row.x[i]);
            auto col = static_cast<unsigned int>(idx_col.x[i]);
            unsigned int idx = (b * R * R) + (row * R) + col; // [b, :, :]
            idx = idx % (R * Ns);
            auto t = static_cast<float>((idx / Ns) * (idx % Ns)) * unit;
            auto v = complex_mul({frag_out_re.x[i], frag_out_im.x[i]}, {__cosf(t), -__sinf(t)});
            frag_out_re.x[i] = v.x;
            frag_out_im.x[i] = v.y;
        }

        store_matrix_sync(&x_re[b][0][0], frag_out_re, R + PAD2, layout_t::mem_row_major);
        store_matrix_sync(&x_im[b][0][0], frag_out_im, R + PAD2, layout_t::mem_row_major);
    }

    // block.sync(); // data are warp local

    {
        // x = einsum('ij,bcj->bci', F.T, x)
        unsigned int b = warp_id;
        // perform x[b, :, :] = einsum('ij,cj->ci', F.T, x[b, :, :])
        // sum along row
        // rows are independent
        // !!! col_major
        fragment16<matrix_b, col_major> frag_x_re;
        fragment16<matrix_b, col_major> frag_x_im;
        fragment16<accumulator> frag_out_re;
        fragment16<accumulator> frag_out_im;

        load_matrix_sync(frag_x_re, &x_re[b][0][0], R + PAD2);
        load_matrix_sync(frag_x_im, &x_im[b][0][0], R + PAD2);
        // load in col_major: cj -> jc
        complex_matmul(F_re, F_im, frag_x_re, frag_x_im, frag_out_re, frag_out_im);
        // ij,jc->ic
        // store in col_major: ic -> ci
        store_matrix_sync(gmem_x_re + b * R * R, frag_out_re, R, layout_t::mem_col_major);
        store_matrix_sync(gmem_x_im + b * R * R, frag_out_im, R, layout_t::mem_col_major);
    }
}

__global__ void digit_reversed_cooley_tukey_16_16(
    half *gmem_x_re,
    half *gmem_x_im,
    const half *gmem_F_re,
    const half *gmem_F_im,
    const half *gmem_idx_row,
    const half *gmem_idx_col)
{
    constexpr unsigned int N = 256;
    constexpr unsigned int R = 16;
    constexpr unsigned int PAD = 8;

    __shared__ half x_re[R][R + PAD];
    __shared__ half x_im[R][R + PAD];

    fragment16<matrix_a, row_major> F_re;
    fragment16<matrix_a, row_major> F_im;
    load_matrix_sync(F_re, gmem_F_re, R);
    load_matrix_sync(F_im, gmem_F_im, R);

    fragment16<accumulator> idx_row;
    fragment16<accumulator> idx_col;
    load_matrix_sync(idx_row, gmem_idx_row, R, layout_t::mem_row_major);
    load_matrix_sync(idx_col, gmem_idx_col, R, layout_t::mem_row_major);

    {
        // x = einsum('ij,jb->ib', F.T, x)
        // sum along col
        // cols are independent
        fragment16<matrix_b, row_major> frag_x_re;
        fragment16<matrix_b, row_major> frag_x_im;
        fragment16<accumulator> frag_out_re;
        fragment16<accumulator> frag_out_im;

        load_matrix_sync(frag_x_re, gmem_x_re, R);
        load_matrix_sync(frag_x_im, gmem_x_im, R);
        complex_matmul(F_re, F_im, frag_x_re, frag_x_im, frag_out_re, frag_out_im);

        constexpr auto Ns = N / R;
        constexpr auto unit = 2 * pi / (R * Ns);
#pragma unroll
        for (int i = 0; i < frag_out_re.num_elements; i++)
        {
            auto row = static_cast<unsigned int>(idx_row.x[i]);
            auto col = static_cast<unsigned int>(idx_col.x[i]);
            unsigned int idx = (row * R) + col;
            auto t = static_cast<float>((idx / Ns) * (idx % Ns)) * unit;
            auto v = complex_mul({frag_out_re.x[i], frag_out_im.x[i]}, {__cosf(t), -__sinf(t)});
            frag_out_re.x[i] = v.x;
            frag_out_im.x[i] = v.y;
        }

        store_matrix_sync(&x_re[0][0], frag_out_re, R + PAD, layout_t::mem_row_major);
        store_matrix_sync(&x_im[0][0], frag_out_im, R + PAD, layout_t::mem_row_major);
    }

    {
        // x = einsum('ij,bj->bi', F.T, x)
        // sum along row
        // rows are independent
        // !!! col_major
        fragment16<matrix_b, col_major> frag_x_re;
        fragment16<matrix_b, col_major> frag_x_im;
        fragment16<accumulator> frag_out_re;
        fragment16<accumulator> frag_out_im;

        load_matrix_sync(frag_x_re, &x_re[0][0], R + PAD);
        load_matrix_sync(frag_x_im, &x_im[0][0], R + PAD);
        // load in col_major: bj -> jb
        complex_matmul(F_re, F_im, frag_x_re, frag_x_im, frag_out_re, frag_out_im);
        // ij,jb-> ib
        // store in col_major: ib -> bi
        store_matrix_sync(gmem_x_re, frag_out_re, R, layout_t::mem_col_major);
        store_matrix_sync(gmem_x_im, frag_out_im, R, layout_t::mem_col_major);
    }
}
