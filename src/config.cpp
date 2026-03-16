#include "config.h"

// #include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

// namespace fs = std::filesystem;
using json = nlohmann::json;

/**
 * @brief Loads simulation parameters from a JSON configuration file.
 *
 * Expected JSON structure:
 * {
 *   "simulation": {
 *     "N": <grid_size>,
 *     "dim": <1 or 2>,
 *     "dx": <spatial_step>,
 *     "dt": <time_step>,
 *     "steps": <num_steps>,
 *     "num_snapshots": <num_outputs>,
 *     "init_file": "<path_to_initial_conditions>"
 *   },
 *   "model": {
 *     "a", "b", "c", "alpha", "phi", "eps2", "eps3", "D1", "D2", "D3"
 *   }
 * }
 *
 * @param filename Path to the JSON configuration file
 * @return SimParams structure with loaded and validated parameters
 * @throws std::runtime_error if file cannot be opened or contains invalid/missing fields
 */
SimParams ConfigLoader::load(const std::string& filename)
{
    std::ifstream f(filename);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }

    json j;
    f >> j;

    SimParams params;

    // Load main simulation parameters
    auto& fhnparams = j.at("simulation");
    params.N = fhnparams.at("N").get<int>();
    params.DIMENSION = fhnparams.at("dim").get<int>();
    params.dx = fhnparams.at("dx").get<float>();
    params.dt = fhnparams.at("dt").get<float>();
    params.steps = fhnparams.at("steps").get<int>();
    params.num_snapshots = fhnparams.at("num_snapshots").get<int>();
    params.init_file = fhnparams.at("init_file").get<std::string>();
    // params.output_dir = fhnparams.value("output_dir", "./results");
    params.name = fhnparams.value("name", "fhn_simulation");

    // Load model kinetic parameters
    auto& model = j.at("model");
    params.model.a = model.at("a").get<float>();
    params.model.b = model.at("b").get<float>();
    params.model.c = model.at("c").get<float>();
    params.model.phi = model.at("phi").get<float>();
    params.model.eps2 = model.at("eps2").get<float>();
    params.model.eps3 = model.at("eps3").get<float>();
    params.model.alpha = model.at("alpha").get<float>();
    params.model.D1 = model.at("D1").get<float>();
    params.model.D2 = model.at("D2").get<float>();
    params.model.D3 = model.at("D3").get<float>();

    // Validate loaded parameters
    validate(params);
    return params;
}

/**
 * @brief Validates simulation parameters for consistency and numerical stability.
 *
 * Performs the following checks:
 * 1. Dimension must be 1 or 2
 * 2. Grid size N must be at least 3
 * 3. dx and dt must be positive
 * 4. steps and num_snapshots must be positive
 * 5. Diffusion coefficients must be non-negative
 * 6. CFL stability condition (with warning if potentially unstable)
 *
 * @param params Parameters to validate
 * @throws std::runtime_error if critical validation fails
 */
void ConfigLoader::validate(const SimParams& params)
{
    // Check dimension validity
    if (params.DIMENSION != 1 && params.DIMENSION != 2) {
        throw std::runtime_error("dim must be 1 or 2");
    }
    // Check minimum grid size
    if (params.N < 3) {
        throw std::runtime_error("N must be >= 3");
    }
    // Check positive step sizes
    if (params.dx <= 0.0f || params.dt <= 0.0f) {
        throw std::runtime_error("dx and dt must be positive");
    }
    // Check positive simulation parameters
    if (params.steps < 1 || params.num_snapshots < 1) {
        throw std::runtime_error("steps and num_snapshots must be positive");
    }
    // Check non-negative diffusion coefficients
    if (params.model.D1 < 0.0f || params.model.D2 < 0.0f || params.model.D3 < 0.0f) {
        throw std::runtime_error("Diffusion coefficients must be non-negative");
    }

    // Check CFL stability condition
    if (!params.is_stable()) {
        float Dmax = params.get_max_diffusion();
        float dt_max = (params.DIMENSION == 1) ? (params.dx * params.dx) / (2.0f * Dmax) : (params.dx * params.dx) / (4.0f * Dmax);
        std::cerr << "[WARNING] dt may violate stability condition!" << std::endl;
        std::cerr << "  Current dt: " << params.dt << std::endl;
        std::cerr << "  Max stable dt: " << dt_max << std::endl;
        std::cerr << "  Consider reducing dt or using implicit scheme." << std::endl;
    }

    // mkdir
    // fs::create_directories(params.output_dir);
}
