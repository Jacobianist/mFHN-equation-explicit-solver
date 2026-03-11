#include <cuda_runtime.h>
#include <hdf5.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <ostream>
#include <random>
#include <string>

#include "config.h"
#include "h5_exporter.h"
#include "solver_explicit.cuh"

void generate_1d(std::vector<float>& u, std::vector<float>& v, std::vector<float>& w, const SimParams& params) {
    float center = params.dx * params.N / 2.0f;
    float radius = 8.0f * params.dx;  // радиус пятна

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> noise(-0.1f, 0.1f);

    for (int i = 0; i < params.N; ++i) {
        float x = i * params.dx;
        float dist = std::abs(x - center);

        if (dist <= radius) {
            u[i] = 1.0f + noise(gen);
            v[i] = 0.0f + 0.1f * noise(gen);
            w[i] = 0.0f + 0.1f * noise(gen);
        } else {
            u[i] = 0.0f + 0.01f * noise(gen);
            v[i] = 0.0f;
            w[i] = 0.0f;
        }
    }
}

void generate_2d(std::vector<float>& u, std::vector<float>& v, std::vector<float>& w, const SimParams& params) {
    float center = params.dx * params.N / 2.0f;
    float radius = 16.0f * params.dx;  // радиус пятна

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> noise(-0.1f, 0.1f);

    for (int iy = 0; iy < params.N; ++iy) {
        for (int ix = 0; ix < params.N; ++ix) {
            size_t idx = iy * params.N + ix;
            float x = ix * params.dx;
            float y = iy * params.dx;
            float dist = std::hypotf(x - center, y - center);

            if (dist <= radius) {
                u[idx] = 1.0f + noise(gen);
                v[idx] = 0.0f + 0.1f * noise(gen);
                w[idx] = 0.0f + 0.1f * noise(gen);
            } else {
                u[idx] = 0.0f + 0.02f * noise(gen);
                v[idx] = 0.0f;
                w[idx] = 0.0f;
            }
        }
    }
}

int main() {
    auto t_start = std::chrono::high_resolution_clock::now();
    auto params = ConfigLoader::load("config.json");

    std::cout << "Config loaded. N=" << params.N << ", dt=" << params.dt << ", dx=" << params.dx << ", alpha=" << params.model.alpha << ", ksi=" << params.dt / (4 * params.dx * params.dx) << std::endl;

    std::vector<float> h_u(params.size_total()), h_v(params.size_total()), h_w(params.size_total());
    std::cout << "ONE  " << params.DIMENSION << std::endl;

    if (params.DIMENSION == 1) {
        generate_1d(h_u, h_v, h_w, params);
    } else {
        generate_2d(h_u, h_v, h_w, params);
    }

    // {  //! REWORK
    //     std::string dataset_name = "u";
    //     hid_t file_id = H5Fopen(params.init_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    //     hid_t dataset_id = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);
    //     hid_t space_id = H5Dget_space(dataset_id);
    //     int ndims = H5Sget_simple_extent_ndims(space_id);
    //     hsize_t dims[2];
    //     H5Sget_simple_extent_dims(space_id, dims, nullptr);
    //     hid_t mem_space_id = H5Screate_simple(2, dims, nullptr);
    //     herr_t status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, mem_space_id, space_id, H5P_DEFAULT, h_u.data());

    //     H5Sclose(mem_space_id);
    //     H5Sclose(space_id);
    //     H5Dclose(dataset_id);
    //     H5Fclose(file_id);
    // }  //! REWORK HDF5 loader

    float *d_u, *d_v, *d_w;
    float *d_u_next, *d_v_next, *d_w_next;
    cudaMalloc(&d_u, params.size_total() * sizeof(float));
    cudaMalloc(&d_u_next, params.size_total() * sizeof(float));
    cudaMalloc(&d_v, params.size_total() * sizeof(float));
    cudaMalloc(&d_v_next, params.size_total() * sizeof(float));
    cudaMalloc(&d_w, params.size_total() * sizeof(float));
    cudaMalloc(&d_w_next, params.size_total() * sizeof(float));

    cudaMemcpy(d_u, h_u.data(), params.size_total() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), params.size_total() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), params.size_total() * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    // HDF5Exporter exporter(params.output_dir + "/results.h5", params.num_snapshots, (params.DIMENSION == 1) ? params.N : params.N,
    //                       (params.DIMENSION == 1) ? 0 : params.N,  // Ny=0 для 1D игнорируется
    //                       params.dx, params.dt);
    int save_interval = params.steps / params.num_snapshots;
    if (save_interval < 1) save_interval = 1;
    int snapshot_idx = 0;
    // exporter.save_step(snapshot_idx++, d_u, d_v, d_w, 0.0f);
    // exporter.save_step(snapshot_idx++, d_u, d_v, d_w, 0.0f);

    // for (int ITERATION_STEP = 1; ITERATION_STEP <= params.steps;
    // ++ITERATION_STEP) { run_simulation_step(d_u_next, d_v_next, d_w_next, d_u,
    // d_v, d_w, &solver_res, params.N, params.dx, params.dt, model.D1, model.D2,
    // model.D3); if (ITERATION_STEP % save_interval == 0 || ITERATION_STEP ==
    // params.steps) {
    //     exporter.save_step(snapshot_idx++, d_u, d_v, d_w);
    // }
    // }

    if (params.DIMENSION == 1) {
        int blocks = (params.N + 255) / 256;
        for (int ITERATION_STEP = 1; ITERATION_STEP <= params.steps; ++ITERATION_STEP) {
            gpu_explicit_1d<<<blocks, 256>>>(d_u_next, d_v_next, d_w_next, d_u, d_v, d_w, params, params.N, params.dt, params.r_u(), params.r_v(), params.r_w());
            std::swap(d_u, d_u_next);
            std::swap(d_v, d_v_next);
            std::swap(d_w, d_w_next);

            // if (params.model.D1 > 0) {
            //     diffusion_1d_explicit<<<blocks, 256>>>(d_u, d_u_next, params.N, params.r_u());
            // } else {
            //     std::swap(d_u, d_u_next);
            // }
            // if (params.model.D2 > 0) {
            //     diffusion_1d_explicit<<<blocks, 256>>>(d_v, d_v_next, params.N, params.r_v());
            // } else {
            //     std::swap(d_v, d_v_next);
            // }
            // if (params.model.D3 > 0) {
            //     diffusion_1d_explicit<<<blocks, 256>>>(d_w, d_w_next, params.N, params.r_w());
            // } else {
            //     std::swap(d_w, d_w_next);
            // }
            cudaDeviceSynchronize();
        }
    } else {
        dim3 threads(16, 16);
        dim3 blocks((params.N + 15) / 16, (params.N + 15) / 16);

        for (int ITERATION_STEP = 1; ITERATION_STEP <= params.steps; ++ITERATION_STEP) {
            gpu_explicit_2d<<<blocks, threads>>>(d_u_next, d_v_next, d_w_next, d_u, d_v, d_w, params, params.N, params.dt, params.r_u(), params.r_v(), params.r_w());
            std::swap(d_u, d_u_next);
            std::swap(d_v, d_v_next);
            std::swap(d_w, d_w_next);
            // if (params.model.D1 > 0) {
            //     diffusion_2d_explicit<<<blocks, threads>>>(d_u, d_u_next, params.N, params.r_u());
            // } else {
            //     std::swap(d_u, d_u_next);
            // }
            // if (params.model.D2 > 0) {
            //     diffusion_2d_explicit<<<blocks, threads>>>(d_v, d_v_next, params.N, params.r_v());
            // } else {
            //     std::swap(d_v, d_v_next);
            // }
            // if (params.model.D3 > 0) {
            //     diffusion_2d_explicit<<<blocks, threads>>>(d_w, d_w_next, params.N, params.r_w());
            // } else {
            //     std::swap(d_w, d_w_next);
            // }
            cudaDeviceSynchronize();
        }
    }

    // cudaDeviceSynchronize();

    cudaMemcpy(h_u.data(), d_u, params.size_total() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_u_next);
    cudaFree(d_v_next);
    cudaFree(d_w_next);

    // OUTPUT
    std::ofstream outFile("result_u1.csv");
    if (params.DIMENSION == 2) {
        for (int j = 0; j < params.N; ++j) {
            for (int i = 0; i < params.N; ++i) {
                int idx = i + j * params.N;
                outFile << h_u[idx];
                if (i < params.N - 1) {
                    outFile << ",";
                }
            }
            outFile << "\n";
        }
    } else {
        for (int j = 0; j < params.N - 1; ++j) {
            outFile << h_u[j] << ",";
        }
        outFile << h_u[params.N - 1] << "\n";
    }
    outFile.close();
    std::cout << "Результат сохранен в result_u1.csv" << std::endl;

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << std::to_string(elapsed_sec) << std::endl;
}
