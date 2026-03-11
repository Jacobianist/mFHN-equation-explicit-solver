#pragma once
#include <algorithm>  // для std::max
#include <string>
#include <vector>

struct ModelParams {
    float a, b, c;
    float alpha, phi;
    float eps2, eps3;
    float D1, D2, D3;
};

struct SimParams {
    int DIMENSION;
    int N;
    float dx;
    float dt;
    int steps;
    int num_snapshots;
    std::string init_file;

    // Вспомогательные методы
    ModelParams model;
    std::string output_dir = "./results";
    std::string name = "mfhn_simulation";
    size_t size_1d() const { return N; }
    size_t size_2d() const { return N * N; }
    size_t size_total() const { return (DIMENSION == 1) ? size_1d() : size_2d(); }
    float get_max_diffusion() const { return std::max({model.D1, model.D2, model.D3}); }
    float r_u() const { return dt * model.D1 / (2.0f * dx * dx); }
    float r_v() const { return dt * model.D2 / (2.0f * dx * dx); }
    float r_w() const { return dt * model.D3 / (2.0f * dx * dx); }

    bool is_stable() const {
        float Dmax = get_max_diffusion();
        if (Dmax == 0.0f) return true;
        float dt_max = (DIMENSION == 1) ? (dx * dx) / (2.0f * Dmax) : (dx * dx) / (4.0f * Dmax);
        return dt <= 0.9f * dt_max;  // 10% safety margin
    }
};
