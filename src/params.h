#pragma once
#include <algorithm>
#include <string>
#include <vector>

/**
 * @brief Model parameters for the modified FitzHugh-Nagumo (mFHN) system.
 * 
 * These parameters define the kinetics of the three-component reaction-diffusion system:
 * - u: activator variable (e.g., membrane potential)
 * - v, w: inhibitor variables (recovery variables)
 */
struct ModelParams {
    float a;      ///< Reaction rate coefficient for activator u
    float b;      ///< Coupling coefficient for inhibitor v
    float c;      ///< Coupling coefficient for inhibitor w
    float alpha;  ///< Cubic nonlinearity coefficient for activator
    float phi;    ///< Scaling factor for reaction kinetics
    float eps2;   ///< Time scale parameter for inhibitor v
    float eps3;   ///< Time scale parameter for inhibitor w
    float D1;     ///< Diffusion coefficient for activator u
    float D2;     ///< Diffusion coefficient for inhibitor v
    float D3;     ///< Diffusion coefficient for inhibitor w
};

/**
 * @brief Simulation parameters for the mFHN explicit solver.
 * 
 * Contains all configuration parameters needed for the CUDA simulation,
 * including spatial/temporal discretization, grid size, and model parameters.
 */
struct SimParams {
    int DIMENSION;            ///< Spatial dimension: 1 for 1D, 2 for 2D simulation
    int N;                    ///< Grid size per dimension (N for 1D, N×N for 2D)
    float dx;                 ///< Spatial step size (grid spacing)
    float dt;                 ///< Time step size
    int steps;                ///< Total number of time steps to simulate
    int num_snapshots;        ///< Number of output snapshots to save
    std::string init_file;    ///< Path to HDF5 file containing initial conditions

    ModelParams model;        ///< Model kinetic parameters
    std::string output_dir = "./results";   ///< Output directory for results
    std::string name = "mfhn_simulation";   ///< Simulation name identifier

    /**
     * @brief Returns the total number of grid points for 1D simulation.
     * @return Number of grid points (N)
     */
    size_t size_1d() const { return N; }

    /**
     * @brief Returns the total number of grid points for 2D simulation.
     * @return Number of grid points (N × N)
     */
    size_t size_2d() const { return N * N; }

    /**
     * @brief Returns the total number of grid points based on dimension.
     * @return N for 1D, N² for 2D
     */
    size_t size_total() const { return (DIMENSION == 1) ? size_1d() : size_2d(); }

    /**
     * @brief Returns the maximum diffusion coefficient among all variables.
     * @return Maximum of D1, D2, D3
     */
    float get_max_diffusion() const { return std::max({model.D1, model.D2, model.D3}); }

    /**
     * @brief Computes the diffusion ratio for activator u.
     * @return dt * D1 / (2 * dx²)
     */
    float r_u() const { return dt * model.D1 / (2.0f * dx * dx); }

    /**
     * @brief Computes the diffusion ratio for inhibitor v.
     * @return dt * D2 / (2 * dx²)
     */
    float r_v() const { return dt * model.D2 / (2.0f * dx * dx); }

    /**
     * @brief Computes the diffusion ratio for inhibitor w.
     * @return dt * D3 / (2 * dx²)
     */
    float r_w() const { return dt * model.D3 / (2.0f * dx * dx); }

    /**
     * @brief Checks if the time step satisfies the CFL stability condition.
     * 
     * For explicit schemes, the time step must satisfy:
     * - 1D: dt ≤ dx² / (2 * D_max)
     * - 2D: dt ≤ dx² / (4 * D_max)
     * 
     * A 10% safety margin is applied.
     * 
     * @return true if stable, false otherwise
     */
    bool is_stable() const
    {
        float Dmax = get_max_diffusion();
        if (Dmax == 0.0f) return true;
        float dt_max = (DIMENSION == 1) ? (dx * dx) / (2.0f * Dmax) : (dx * dx) / (4.0f * Dmax);
        return dt <= 0.9f * dt_max;  // 10% safety margin
    }
};
