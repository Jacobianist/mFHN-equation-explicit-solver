#pragma once

#include <cuda_runtime.h>

__device__ inline void reaction_mfhn(float& du, float& dv, float& dw, const float u, const float v, const float w, const ModelParams& p);

__global__ void gpu_explicit_1d(float* u_out, float* v_out, float* w_out, const float* u_cur, const float* v_cur, const float* w_cur, const SimParams params, int N, float dt, float r_u, float r_v, float r_w);

__global__ void gpu_explicit_2d(float* u_out, float* v_out, float* w_out, const float* u_cur, const float* v_cur, const float* w_cur, const SimParams params, int N, float dt, float r_u, float r_v, float r_w);
