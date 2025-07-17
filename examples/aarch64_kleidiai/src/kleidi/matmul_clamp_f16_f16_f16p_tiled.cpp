//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Example usage for matrix multiplication of two half precision floating-point (FP16) matrices and the accumulation of
// the result into an FP16 destination matrix.
//
// The activations and the weights, stored in the LHS and RHS matrices respectively, are both non-transposed matrices.
// The matrix multiplication computation is performed using floating-point fused multiply-add to accumulator (FMLA)
// vector instructions present in the FEAT_FP16 Arm® architecture feature.
//

#define TILED 0
#define ABFT 1
#define EPSILON 0.1f
#define PROFILING 0

#define __aarch64__
#define __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
#define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || \
    !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16.
#else
#include <arm_neon.h>

#include <algorithm>
#include <cfloat>
// #include <cmath>
#include <cstddef>
#include <cstring>
// #include <iomanip>
// #include <iostream>

// Include micro-kernel variants
#include "kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h"
#include "kai_matmul_clamp_f16_f16_f16p_interface.h"
#include "kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h"

// Include convolution filters data
#include "conv_filters.hpp"

// Results structure to store execution results
struct ExecutionResults
{
    uint8_t abft_triggered;   // Flag indicating if ABFT was triggered
    uint8_t error_corrected;  // Flag indicating if error was corrected
    uint8_t weights_tampered; // Flag indicating if weights were tampered
    uint32_t output_hash;     // Pearson hash value of the output data matrix
    uint64_t weights_hash;
};

// Global results vector
ExecutionResults execution_results = {0, 0, 0, 0, 0};

namespace
{

    extern "C" int strcmp(const char *s1, const char *s2)
    {
        while (*s1 && (*s1 == *s2))
        {
            s1++;
            s2++;
        }
        return *(const unsigned char *)s1 - *(const unsigned char *)s2;
    }

    /// Micro-kernel interface
    constexpr kai_matmul_clamp_f16_f16_f16p_ukernel ukernel{
        kai_get_m_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_n_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_nr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_kr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_sr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_lhs_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_dst_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_dst_size_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla};

    /// Fills the matrix with zero values
    void fill_zero_matrix(size_t num_rows, size_t num_cols, float16_t *dst)
    {
        for (size_t i = 0; i < num_rows * num_cols; i++)
        {
            dst[i] = float16_t(0);
        }
    }

#if defined(ABFT) && ABFT == 1
    float16_t neon_oc_f16_f16(
        size_t num_rows, size_t num_cols, size_t row_stride, const float16_t *src, float16_t *output_checksum_row,
        float16_t *output_checksum_col)
    {
        // Initialize output checksums
        for (size_t col = 0; col < num_cols; ++col)
        {
            output_checksum_row[col] = 0.0f;
        }
        for (size_t row = 0; row < num_rows; ++row)
        {
            output_checksum_col[row] = 0.0f;
        }

        float16x8_t total_sum_vec = vdupq_n_f16(0.0f);
        float16_t scalar_sum_tail = 0.0f;

        const size_t vec_width = 8;
        const size_t col_unroll = num_cols & ~(vec_width - 1);

        for (size_t row = 0; row < num_rows; ++row)
        {
            float16x8_t row_sum_vec = vdupq_n_f16(0.0f);
            float16_t row_scalar_sum = 0.0f;

            const float16_t *row_ptr = src + row * row_stride;

            for (size_t col = 0; col < col_unroll; col += vec_width)
            {
                float16x8_t data = vld1q_f16(row_ptr + col);

                // Update column checksums
                float16x8_t col_sum = vld1q_f16(output_checksum_row + col);
                col_sum = vaddq_f16(col_sum, data);
                vst1q_f16(output_checksum_row + col, col_sum);

                // Accumulate into total and row vectors
                total_sum_vec = vaddq_f16(total_sum_vec, data);
                row_sum_vec = vaddq_f16(row_sum_vec, data);
            }

            // Reduce row_sum_vec to scalar
            float16x4_t row_low = vget_low_f16(row_sum_vec);
            float16x4_t row_high = vget_high_f16(row_sum_vec);
            float16x4_t row_sum_4 = vadd_f16(row_low, row_high);
            float16_t row_vec_sum = vget_lane_f16(row_sum_4, 0) + vget_lane_f16(row_sum_4, 1) +
                                    vget_lane_f16(row_sum_4, 2) + vget_lane_f16(row_sum_4, 3);

            row_scalar_sum += row_vec_sum;

            // Tail processing
            for (size_t col = col_unroll; col < num_cols; ++col)
            {
                float16_t val = row_ptr[col];
                output_checksum_row[col] += val;
                row_scalar_sum += val;
                scalar_sum_tail += val;
            }

            output_checksum_col[row] = row_scalar_sum;
        }

        // Reduce total_sum_vec to scalar
        float16x4_t total_low = vget_low_f16(total_sum_vec);
        float16x4_t total_high = vget_high_f16(total_sum_vec);
        float16x4_t total_4 = vadd_f16(total_low, total_high);
        float16_t total_vec_sum =
            vget_lane_f16(total_4, 0) + vget_lane_f16(total_4, 1) + vget_lane_f16(total_4, 2) + vget_lane_f16(total_4, 3);

        return total_vec_sum + scalar_sum_tail;
    }

    static void neon_ic_f16_f16(const float16_t *lhs, float16_t *ic, size_t m, size_t k)
    {
        // Zero-initialize ic vector
        for (size_t k_idx = 0; k_idx < k; ++k_idx)
        {
            ic[k_idx] = 0;
        }

        size_t k_unroll = k & ~7; // unroll in chunks of 8

        for (size_t k_idx = 0; k_idx < k_unroll; k_idx += 8)
        {
            float16x8_t acc = vdupq_n_f16(0.0f);

            for (size_t m_idx = 0; m_idx < m; ++m_idx)
            {
                const float16_t *row_ptr = lhs + m_idx * k + k_idx;
                float16x8_t data = vld1q_f16(row_ptr);
                acc = vaddq_f16(acc, data);
            }

            vst1q_f16(ic + k_idx, acc);
        }

        // Handle tail elements
        for (size_t k_idx = k_unroll; k_idx < k; ++k_idx)
        {
            for (size_t m_idx = 0; m_idx < m; ++m_idx)
            {
                ic[k_idx] += lhs[m_idx * k + k_idx];
            }
        }
    }

    static void neon_ic_f16_f32(const float16_t *rhs, float32_t *ic, size_t n, size_t k)
    {
        /*for (size_t i = 0; i < k; ++i) {
           ic[i] = 0.0f;
       }*/

        size_t i = 0;
        for (; i + 3 < k; i += 4)
        {
            vst1q_f32(ic + i, vdupq_n_f32(0.0f));
        }
        for (; i < k; i++)
        {
            ic[i] = 0.0f;
        }

        for (size_t k_idx = 0; k_idx < k; ++k_idx)
        {
            float32x4_t acc = vdupq_n_f32(0.0f);
            size_t n_idx = 0;

            // Vectorized sum over n dimension (in steps of 4)
            for (; n_idx + 3 < n; n_idx += 4)
            {
                const float16_t *ptr = rhs + k_idx * n + n_idx;

                float16x4_t v_half = vld1_f16(ptr);   // Load 4 half-floats
                float32x4_t v = vcvt_f32_f16(v_half); // Convert to float32
                acc = vaddq_f32(acc, v);              // Accumulate
            }

            // Horizontal add 4-wide vector
            float32_t sum =
                vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1) + vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3);

            // Handle tail
            for (; n_idx < n; ++n_idx)
            {
                sum += (float)rhs[k_idx * n + n_idx];
            }

            ic[k_idx] = sum;
        }
    }

    void neon_checksum_row_f16(const float16_t *ic, const float16_t *rhs, float16_t *checksum_row, size_t K, size_t N)
    {
        const size_t vec_width = 8;
        const size_t K_vec = K & ~(vec_width - 1);

        for (size_t col = 0; col < N; ++col)
        {
            float16x8_t acc_vec = vdupq_n_f16(0.0f);
            float16_t acc_tail = 0.0f;

            const float16_t *rhs_col_ptr = rhs + col; // Start of column `col` in column-major access

            for (size_t row = 0; row < K_vec; row += vec_width)
            {
                float16x8_t ic_vec = vld1q_f16(ic + row);

                // Load 8 elements from column `col` of rhs (non-contiguous access)
                float16_t rhs_vals[8];
                for (int i = 0; i < 8; ++i)
                {
                    rhs_vals[i] = rhs_col_ptr[(row + i) * N];
                }
                float16x8_t rhs_vec = vld1q_f16(rhs_vals);

                acc_vec = vfmaq_f16(acc_vec, ic_vec, rhs_vec);
            }

            // Horizontal reduction of acc_vec
            float16x4_t acc_low = vget_low_f16(acc_vec);
            float16x4_t acc_high = vget_high_f16(acc_vec);
            float16x4_t sum_4 = vadd_f16(acc_low, acc_high);
            float16_t acc_sum =
                vget_lane_f16(sum_4, 0) + vget_lane_f16(sum_4, 1) + vget_lane_f16(sum_4, 2) + vget_lane_f16(sum_4, 3);

            // Tail loop
            for (size_t row = K_vec; row < K; ++row)
            {
                acc_tail += ic[row] * rhs[row * N + col];
            }

            checksum_row[col] = acc_sum + acc_tail;
        }
    }

    void neon_checksum_col_f16(const float16_t *lhs, const float16_t *fc, float16_t *checksum_col, size_t M, size_t K)
    {
        const size_t vec_width = 8;
        const size_t K_vec = K & ~(vec_width - 1);

        for (size_t row = 0; row < M; ++row)
        {
            float16x8_t acc_vec = vdupq_n_f16(0.0f);
            float16_t acc_tail = 0.0f;

            const float16_t *lhs_row = lhs + row * K;

            for (size_t col = 0; col < K_vec; col += vec_width)
            {
                float16x8_t lhs_vec = vld1q_f16(lhs_row + col);
                float16x8_t fc_vec = vld1q_f16(fc + col);
                acc_vec = vfmaq_f16(acc_vec, lhs_vec, fc_vec);
            }

            // Horizontal sum of acc_vec
            float16x4_t acc_low = vget_low_f16(acc_vec);
            float16x4_t acc_high = vget_high_f16(acc_vec);
            float16x4_t sum_4 = vadd_f16(acc_low, acc_high);
            float16_t acc_sum =
                vget_lane_f16(sum_4, 0) + vget_lane_f16(sum_4, 1) + vget_lane_f16(sum_4, 2) + vget_lane_f16(sum_4, 3);

            // Handle tail
            for (size_t col = K_vec; col < K; ++col)
            {
                acc_tail += lhs_row[col] * fc[col];
            }

            checksum_col[row] = acc_sum + acc_tail;
        }
    }
#endif // ABFT
} // namespace

float16_t bias[1024] = {0};
float16_t ic[1024] = {0};
float16_t rhs_packed[75000] = {0};
float16_t output_checksum_row[1024] = {0.0f};
float16_t output_checksum_col[1024] = {0.0f};
float16_t checksum_row[1024] = {0.0f};
float16_t checksum_col[1024] = {0.0f};

// Function to run a single test case
int run_test_case(const TestCase &test_case)
{
    int ret = 0;

    const size_t M = test_case.M;
    const size_t N = test_case.N;
    const size_t K = test_case.K;

    const float16_t *lhs = test_case.lhs_data;
    const float16_t *rhs = test_case.rhs_data;

    // Check the Hash of the Weights
    execution_results.weights_hash = calculate_hash_f16(rhs, K * N);
    if (execution_results.weights_hash != test_case.gemm_hash)
    {
        execution_results.weights_tampered = 1;
    }

    // Allocate bias array and initialize to zero
    const size_t bias_size = N;
    // fill_zero_matrix(1, N, bias);

    const size_t dst_size = M * N;

#if defined(ABFT) && ABFT == 1

    // Use predefined FC checksum from test case instead of calculating
    const float16_t *fc = test_case.fc_checksum;

    // Calculate IC checksum for the RHS matrix
    // float16_t* ic = new float16_t[K];
    neon_ic_f16_f16(lhs, ic, M, K);

#endif // ABFT

    //----------- MICRO-KERNELS TESTS
    //------------------------------------
    //------------------------------------


    
    const size_t nr = ukernel.get_nr();
    const size_t kr = ukernel.get_kr();
    const size_t sr = ukernel.get_sr();

    // In a single row, we pack nr bias values followed by K rows of nr RHS values
    const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(N, K);
    const size_t rhs_packed_cols = nr + K * nr;
    const size_t rhs_packed_rows = rhs_packed_size / (rhs_packed_cols * sizeof(float16_t));

    const size_t lhs_stride = K * sizeof(float16_t);
    const size_t rhs_stride = N * sizeof(float16_t);
    const size_t dst_stride_row = N * sizeof(float16_t);
    const size_t dst_stride_col = sizeof(float16_t);

    // Packing only needs to be performed once if the contents of the bias and RHS matrices are expected to be constant.
    kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(
        1, N, K, nr, kr, sr, // Packing arguments
        rhs_stride,          // RHS stride
        rhs,                 // RHS
        bias,                // Bias
        NULL,                // Scale
        rhs_packed,          // RHS packed
        0, NULL);

    // Framework scheduling params
    const size_t m_step = ukernel.get_m_step(); // Scheduling along M
    const size_t n_step = ukernel.get_n_step(); // Scheduling along N
    const float16_t *dst = test_case.dst_data;

     asm volatile("nop" ::: "memory");
    asm volatile("nop" ::: "memory");
    asm volatile("nop" ::: "memory");

    for (size_t i_m_step = 0; i_m_step < M; i_m_step += m_step)
    {
        for (size_t i_n_step = 0; i_n_step < N; i_n_step += n_step)
        {
            // Support functions return offset in bytes
            const uint8_t *lhs_ptr =
                (const uint8_t *)lhs + (ukernel.get_lhs_packed_offset(i_m_step, K * sizeof(uint16_t)));
            const uint8_t *rhs_ptr = (const uint8_t *)rhs_packed + (ukernel.get_rhs_packed_offset(i_n_step, K));
            uint8_t *dst_ptr = (uint8_t *)dst + (ukernel.get_dst_offset(i_m_step, i_n_step, N * sizeof(uint16_t)));

            const size_t actual_m = std::min(M - i_m_step, m_step);
            const size_t actual_n = std::min(N - i_n_step, n_step);

            ukernel.run_matmul(
                actual_m, actual_n, K, // Dimensions
                lhs_ptr,               // LHS
                lhs_stride,            // LHS stride
                rhs_ptr,               // RHS packed
                dst_ptr,               // DST
                dst_stride_row,        // DST stride (row)
                dst_stride_col,        // DST stride (col)
                -FLT_MAX, FLT_MAX      // Min and max for the clamp operation
            );
        }
    }

    asm volatile("nop" ::: "memory");
    asm volatile("nop" ::: "memory");
    asm volatile("nop" ::: "memory");

#if defined(ABFT) && ABFT == 1
    float16_t oc = neon_oc_f16_f16(M, N, N, dst, output_checksum_row, output_checksum_col);
    float16_t dot = 0;
    for (size_t i = 0; i < K; i++)
    {
        dot += fc[i] * ic[i];
    }

    neon_checksum_col_f16(lhs, fc, checksum_col, M, K);
    neon_checksum_row_f16(ic, rhs, checksum_row, K, N);

    bool col_error_detected = false;
    bool row_error_detected = false;

    for (size_t row = 0; row < M; row++)
    {
        if (__builtin_fabsf(output_checksum_col[row] - checksum_col[row]) > EPSILON)
        {
            col_error_detected = true;
            // printf("Column Checksum Mismatch at %zu (Delta: %f)\n", row, output_checksum_col[row] - checksum_col[row]);
        }
    }

    for (size_t col = 0; col < N; col++)
    {
        if (__builtin_fabsf(output_checksum_row[col] - checksum_row[col]) > EPSILON)
        {
            row_error_detected = true;
            // printf("Row Checksum Mismatch at %zu (Delta: %f)\n", col, output_checksum_row[col] - checksum_row[col]);
        }
    }

    // Check overall checksum
    bool overall_error_detected = false;
    if (__builtin_fabsf(dot - oc) > EPSILON)
    {
        overall_error_detected = true;
        ret = 1;
    }

    // Update execution results
    if (col_error_detected || row_error_detected || overall_error_detected)
    {
        execution_results.abft_triggered = 1;

        // Do error correction here if possible
        execution_results.error_corrected = 0; // Set to 1 if error correction was successful
    }

    // Calculate hash of output matrix
    execution_results.output_hash = calculate_hash_f16(dst, M * N);
#endif // ABFT
    return ret;
}

int main(int argc, char **argv)
{
    // Initialize execution results
    execution_results.abft_triggered = 0;
    execution_results.error_corrected = 0;
    execution_results.output_hash = 0;
    execution_results.weights_hash = 0;

    int total_ret = run_test_case(test_cases[0]);

    // Results are now available in execution_results structure
    // execution_results.abft_triggered - indicates if ABFT was triggered
    // execution_results.error_corrected - indicates if error was corrected
    // execution_results.output_hash - hash of the output matrix

    return total_ret;
}
#endif // Architectural features check.
