#ifndef REACTION_CUH
#define REACTION_CUH

__device__ inline void reaction_mfhn(float &du, float &dv, float &dw,
                                     const float u, const float v,
                                     const float w, const ModelParams &p);

__global__ void gpu_rk4_1d(float *u_out, float *v_out, float *w_out,
                           const float *u_cur, const float *v_cur,
                           const float *w_cur, const SimParams params, int N,
                           float dt);

__global__ void gpu_rk4_2d(float *u_out, float *v_out, float *w_out,
                           const float *u_cur, const float *v_cur,
                           const float *w_cur, const SimParams params, int N,
                           float dt);

__global__ void diffusion_1d_explicit(float *u_next, const float *u_cur, int N, float r);
__global__ void diffusion_2d_explicit(float *u_next, const float *u_cur, int N, float r);











#endif
