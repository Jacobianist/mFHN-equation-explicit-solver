#include "config.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

SimParams ConfigLoader::load(const std::string& filename)
{
    std::ifstream f(filename);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }

    json j;
    f >> j;

    SimParams params;

    // main parameters
    auto& fhnparams = j.at("simulation");
    params.N = fhnparams.at("N").get<int>();
    params.DIMENSION = fhnparams.at("dim").get<int>();
    params.dx = fhnparams.at("dx").get<float>();
    params.dt = fhnparams.at("dt").get<float>();
    params.steps = fhnparams.at("steps").get<int>();
    params.num_snapshots = fhnparams.at("num_snapshots").get<int>();
    params.init_file = fhnparams.at("init_file").get<std::string>();
    params.output_dir = fhnparams.value("output_dir", "./results");
    params.name = fhnparams.value("name", "fhn_simulation");

    // model parameters
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

    validate(params);
    return params;
}
void ConfigLoader::validate(const SimParams& params)
{
    if (params.DIMENSION != 1 && params.DIMENSION != 2) {
        throw std::runtime_error("dim must be 1 or 2");
    }
    if (params.N < 3) {
        throw std::runtime_error("N must be >= 3");
    }
    if (params.dx <= 0.0f || params.dt <= 0.0f) {
        throw std::runtime_error("dx and dt must be positive");
    }
    if (params.steps < 1 || params.num_snapshots < 1) {
        throw std::runtime_error("steps and num_snapshots must be positive");
    }
    if (params.model.D1 < 0.0f || params.model.D2 < 0.0f || params.model.D3 < 0.0f) {
        throw std::runtime_error("Diffusion coefficients must be non-negative");
    }

    // checking
    if (!params.is_stable()) {
        float Dmax = params.get_max_diffusion();
        float dt_max = (params.DIMENSION == 1) ? (params.dx * params.dx) / (2.0f * Dmax) : (params.dx * params.dx) / (4.0f * Dmax);
        std::cerr << "[WARNING] dt may violate stability condition!" << std::endl;
        std::cerr << "  Current dt: " << params.dt << std::endl;
        std::cerr << "  Max stable dt: " << dt_max << std::endl;
        std::cerr << "  Consider reducing dt or using implicit scheme." << std::endl;
    }

    // mkdir
    fs::create_directories(params.output_dir);
}
