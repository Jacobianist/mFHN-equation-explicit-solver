#include "params.h"
#include "solver_explicit.cuh"
#include <cuda_runtime.h>

__device__ void reaction_mfhn(float &du, float &dv, float &dw, const float u,
                              const float v, const float w,
                              const ModelParams &p) {
  du = p.phi * (p.a * u - p.alpha * u * u * u - p.b * v - p.c * w);
  dv = p.phi * p.eps2 * (u - v);
  dw = p.phi * p.eps3 * (u - w);
}

__global__ void gpu_rk4_1d(float *u_out, float *v_out, float *w_out,
                                        const float *u_cur, const float *v_cur, const float *w_cur,
                                        const SimParams params, int N, float dt) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  float u0 = u_cur[i], v0 = v_cur[i], w0 = w_cur[i];
  float k1_u, k1_v, k1_w;
  float k2_u, k2_v, k2_w;
  float k3_u, k3_v, k3_w;
  float k4_u, k4_v, k4_w;

  reaction_mfhn(k1_u, k1_v, k1_w, u0, v0, w0, params.model);

  float u_k2 = u0 + 0.5f * dt * k1_u;
  float v_k2 = v0 + 0.5f * dt * k1_v;
  float w_k2 = w0 + 0.5f * dt * k1_w;
  reaction_mfhn(k2_u, k2_v, k2_w, u_k2, v_k2, w_k2, params.model);

  float u_k3 = u0 + 0.5f * dt * k2_u;
  float v_k3 = v0 + 0.5f * dt * k2_v;
  float w_k3 = w0 + 0.5f * dt * k2_w;
  reaction_mfhn(k3_u, k3_v, k3_w, u_k3, v_k3, w_k3, params.model);

  float u_k4 = u0 + dt * k3_u;
  float v_k4 = v0 + dt * k3_v;
  float w_k4 = w0 + dt * k3_w;
  reaction_mfhn(k4_u, k4_v, k4_w, u_k4, v_k4, w_k4, params.model);

  u_out[i] =u0 +(dt / 6.0f) * (k1_u + 2.0f * k2_u + 2.0f * k3_u + k4_u);
  v_out[i] =v0 +(dt / 6.0f) * (k1_v + 2.0f * k2_v + 2.0f * k3_v + k4_v);
  w_out[i] =w0 +(dt / 6.0f) * (k1_w + 2.0f * k2_w + 2.0f * k3_w + k4_w);
}

__global__ void gpu_rk4_2d(float *u_out, float *v_out, float *w_out,
                            const float *u_cur, const float *v_cur, const float *w_cur,
                            const SimParams params, int N, float dt) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= N || iy >= N) return;

    int idx = iy * N + ix;

  float u0 = u_cur[idx], v0 = v_cur[idx], w0 = w_cur[idx];
  float k1_u, k1_v, k1_w;
  float k2_u, k2_v, k2_w;
  float k3_u, k3_v, k3_w;
  float k4_u, k4_v, k4_w;

  reaction_mfhn(k1_u, k1_v, k1_w, u0, v0, w0, params.model);

  float u_k2 = u0 + 0.5f * dt * k1_u;
  float v_k2 = v0 + 0.5f * dt * k1_v;
  float w_k2 = w0 + 0.5f * dt * k1_w;
  reaction_mfhn(k2_u, k2_v, k2_w, u_k2, v_k2, w_k2, params.model);

  float u_k3 = u0 + 0.5f * dt * k2_u;
  float v_k3 = v0 + 0.5f * dt * k2_v;
  float w_k3 = w0 + 0.5f * dt * k2_w;
  reaction_mfhn(k3_u, k3_v, k3_w, u_k3, v_k3, w_k3, params.model);

  float u_k4 = u0 + dt * k3_u;
  float v_k4 = v0 + dt * k3_v;
  float w_k4 = w0 + dt * k3_w;
  reaction_mfhn(k4_u, k4_v, k4_w, u_k4, v_k4, w_k4, params.model);

  u_out[idx] = u0 + (dt / 6.0f) * (k1_u + 2.0f * k2_u + 2.0f * k3_u + k4_u);
  v_out[idx] = v0 + (dt / 6.0f) * (k1_v + 2.0f * k2_v + 2.0f * k3_v + k4_v);
  w_out[idx] = w0 + (dt / 6.0f) * (k1_w + 2.0f * k2_w + 2.0f * k3_w + k4_w);
}


__global__ void diffusion_1d_explicit(float *u_next, const float *u_cur, int N, float r)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float u_left, u_right;

    if (i == 0) {
        u_left = u_cur[1];
        u_right = u_cur[1];
    } else if (i == N - 1) {
        u_left = u_cur[N - 2];
        u_right = u_cur[N - 2];
    } else {
        u_left = u_cur[i - 1];
        u_right = u_cur[i + 1];
    }

    float laplacian = (u_left - 2.0f * u_cur[i] + u_right) ;

    u_next[i] = u_cur[i] + r * laplacian;
}



__global__ void diffusion_2d_explicit(
    float *u_next, const float *u_cur,
    int N, float r)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= N || iy >= N) return;

    int idx = iy * N + ix;

    int left_idx, right_idx;

    if (ix == 0) {
        left_idx = iy * N + 1;
        right_idx = iy * N + 1;
    } else if (ix == N - 1) {
        left_idx = iy * N + (N - 2);
        right_idx = iy * N + (N - 2);
    } else {
        left_idx = idx - 1;
        right_idx = idx + 1;
    }

    int bottom_idx, top_idx;

    if (iy == 0) {
        bottom_idx = 1 * N + ix;
        top_idx = 1 * N + ix;
    } else if (iy == N - 1) {
        bottom_idx = (N - 2) * N + ix;
        top_idx = (N - 2) * N + ix;
    } else {
        bottom_idx = idx - N;
        top_idx = idx + N;
    }

    float u_left = u_cur[left_idx];
    float u_right = u_cur[right_idx];
    float u_bottom = u_cur[bottom_idx];
    float u_top = u_cur[top_idx];

    float laplacian = (u_left - 4.0f * u_cur[idx] + u_right + u_bottom + u_top) ;

    u_next[idx] = u_cur[idx] + r * laplacian;
}
