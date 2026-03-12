#include <cuda_runtime.h>

#include "params.h"
#include "solver_explicit.cuh"

__device__ void reaction_mfhn(float& du, float& dv, float& dw, const float u, const float v, const float w, const ModelParams& p)
{
    du = p.phi * (p.a * u - p.alpha * u * u * u - p.b * v - p.c * w);
    dv = p.phi * p.eps2 * (u - v);
    dw = p.phi * p.eps3 * (u - w);
}

__global__ void gpu_explicit_1d(float* u_out, float* v_out, float* w_out, const float* u_cur, const float* v_cur, const float* w_cur, const SimParams params, int N, float dt, float r_u, float r_v, float r_w)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

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

    float u_left, u_right;
    float v_left, v_right;
    float w_left, w_right;

    if (i == 0) {
        u_left = u_cur[1];
        u_right = u_cur[1];
        v_left = v_cur[1];
        v_right = v_cur[1];
        w_left = w_cur[1];
        w_right = w_cur[1];
    } else if (i == N - 1) {
        u_left = u_cur[N - 2];
        u_right = u_cur[N - 2];
        v_left = v_cur[N - 2];
        v_right = v_cur[N - 2];
        w_left = w_cur[N - 2];
        w_right = w_cur[N - 2];
    } else {
        u_left = u_cur[i - 1];
        u_right = u_cur[i + 1];
        v_left = v_cur[i - 1];
        v_right = v_cur[i + 1];
        w_left = w_cur[i - 1];
        w_right = w_cur[i + 1];
    }

    u_out[i] = u0 + r_u * (u_left - 2.0f * u0 + u_right) + (dt / 6.0f) * (k1_u + 2.0f * k2_u + 2.0f * k3_u + k4_u);
    v_out[i] = v0 + r_v * (v_left - 2.0f * v0 + v_right) + (dt / 6.0f) * (k1_v + 2.0f * k2_v + 2.0f * k3_v + k4_v);
    w_out[i] = w0 + r_w * (w_left - 2.0f * w0 + w_right) + (dt / 6.0f) * (k1_w + 2.0f * k2_w + 2.0f * k3_w + k4_w);
}

__global__ void gpu_explicit_2d(float* u_out, float* v_out, float* w_out, const float* u_cur, const float* v_cur, const float* w_cur, const SimParams params, int N, float dt, float r_u, float r_v, float r_w)
{
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

    u_out[idx] = u0 + r_u * (u_cur[left_idx] + u_cur[right_idx] + u_cur[bottom_idx] + u_cur[top_idx] - 4.0f * u0) + (dt / 6.0f) * (k1_u + 2.0f * k2_u + 2.0f * k3_u + k4_u);
    v_out[idx] = v0 + r_v * (v_cur[left_idx] + v_cur[right_idx] + v_cur[bottom_idx] + v_cur[top_idx] - 4.0f * v0) + (dt / 6.0f) * (k1_v + 2.0f * k2_v + 2.0f * k3_v + k4_v);
    w_out[idx] = w0 + r_w * (w_cur[left_idx] + w_cur[right_idx] + w_cur[bottom_idx] + w_cur[top_idx] - 4.0f * w0) + (dt / 6.0f) * (k1_w + 2.0f * k2_w + 2.0f * k3_w + k4_w);
}
