#pragma once

#include <cuda_runtime.h>

#include "params.h"

/**
 * @brief Computes the reaction kinetics for the modified FitzHugh-Nagumo (mFHN) system.
 *
 * Device function that calculates the reaction terms for the three-component system:
 * - du/dt = phi * (a*u - alpha*u^3 - b*v - c*w)
 * - dv/dt = phi * eps2 * (u - v)
 * - dw/dt = phi * eps3 * (u - w)
 *
 * @param du Output: reaction term for activator u
 * @param dv Output: reaction term for inhibitor v
 * @param dw Output: reaction term for inhibitor w
 * @param u Current value of activator
 * @param v Current value of inhibitor v
 * @param w Current value of inhibitor w
 * @param p Model parameters
 */
__device__ inline void reaction_mfhn(float& du, float& dv, float& dw, const float u, const float v, const float w, const ModelParams& p);

/**
 * @brief CUDA kernel for 1D explicit time integration of the mFHN system.
 *
 * Solves the three-component reaction-diffusion system in 1D using:
 * - RK4 (Runge-Kutta 4th order) for reaction terms
 * - Central finite differences for diffusion (Laplacian)
 * - Neumann (zero-flux) boundary conditions
 *
 * Each thread computes one grid point. Uses two-array swapping (current/next).
 *
 * @param u_out Output array for activator u at next time step
 * @param v_out Output array for inhibitor v at next time step
 * @param w_out Output array for inhibitor w at next time step
 * @param u_cur Input array for activator u at current time step
 * @param v_cur Input array for inhibitor v at current time step
 * @param w_cur Input array for inhibitor w at current time step
 * @param params Simulation parameters (includes model parameters)
 * @param N Number of grid points
 * @param dt Time step size
 * @param r_u Diffusion ratio for u: dt * D1 / (2 * dx^2)
 * @param r_v Diffusion ratio for v: dt * D2 / (2 * dx^2)
 * @param r_w Diffusion ratio for w: dt * D3 / (2 * dx^2)
 */
__global__ void gpu_explicit_1d(float* u_out, float* v_out, float* w_out, const float* u_cur, const float* v_cur, const float* w_cur, const SimParams params, int N, float dt, float r_u, float r_v, float r_w);

/**
 * @brief CUDA kernel for 2D explicit time integration of the mFHN system.
 *
 * Solves the three-component reaction-diffusion system in 2D using:
 * - RK4 (Runge-Kutta 4th order) for reaction terms
 * - 5-point stencil for 2D Laplacian (central differences)
 * - Neumann (zero-flux) boundary conditions on all edges
 *
 * Each thread computes one grid point (ix, iy). Uses two-array swapping (current/next).
 * Thread block configuration: 16x16 threads per block.
 *
 * @param u_out Output array for activator u at next time step
 * @param v_out Output array for inhibitor v at next time step
 * @param w_out Output array for inhibitor w at next time step
 * @param u_cur Input array for activator u at current time step
 * @param v_cur Input array for inhibitor v at current time step
 * @param w_cur Input array for inhibitor w at current time step
 * @param params Simulation parameters (includes model parameters)
 * @param N Number of grid points per dimension (N x N grid)
 * @param dt Time step size
 * @param r_u Diffusion ratio for u: dt * D1 / (2 * dx^2)
 * @param r_v Diffusion ratio for v: dt * D2 / (2 * dx^2)
 * @param r_w Diffusion ratio for w: dt * D3 / (2 * dx^2)
 */
__global__ void gpu_explicit_2d(float* u_out, float* v_out, float* w_out, const float* u_cur, const float* v_cur, const float* w_cur, const SimParams params, int N, float dt, float r_u, float r_v, float r_w);
