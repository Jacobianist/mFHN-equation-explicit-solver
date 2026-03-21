#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "config.h"
#include "cuda_utils.h"
#include "h5_exporter.h"
#include "logger.h"
#include "solver_explicit.cuh"

/**
 * @file main.cu
 * @brief Main entry point for the mFHN explicit solver
 */

// ============================================================================
// Initial Condition Generators
// ============================================================================

void generate_1d(std::vector<float>& u, std::vector<float>& v, std::vector<float>& w, const SimParams& params)
{
    float center = params.dx * params.N / 2.0f;
    float radius = 8.0f * params.dx;

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

void generate_2d(std::vector<float>& u, std::vector<float>& v, std::vector<float>& w, const SimParams& params)
{
    float center = params.dx * params.N / 2.0f;
    float radius = 16.0f * params.dx;

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

bool load_hdf5_data(const std::string& filename, const std::string& dataset_name, int target_dim, int slice_index, std::vector<float>& out_data)
{
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) return false;

    hid_t dataset_id = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        H5Fclose(file_id);
        return false;
    }

    hid_t file_space_id = H5Dget_space(dataset_id);
    int ndims = H5Sget_simple_extent_ndims(file_space_id);

    if (ndims != 2) {
        H5Sclose(file_space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    hsize_t dims[3] = {0, 0, 0};
    H5Sget_simple_extent_dims(file_space_id, dims, nullptr);

    herr_t status = -1;
    if (target_dim == 2 && ndims == 2) {
        hsize_t total_size = dims[0] * dims[1];
        out_data.resize(total_size);
        hid_t mem_space_id = H5Screate_simple(2, dims, nullptr);
        status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, mem_space_id, file_space_id, H5P_DEFAULT, out_data.data());
        H5Sclose(mem_space_id);
    } else if (target_dim == 1) {
        if (ndims == 1) {
            out_data.resize(dims[0]);
            hid_t mem_space_id = H5Screate_simple(1, dims, nullptr);
            status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, mem_space_id, file_space_id, H5P_DEFAULT, out_data.data());
            H5Sclose(mem_space_id);
        } else if (ndims == 2 && slice_index >= 0 && slice_index < (int)dims[0]) {
            hsize_t size_1d = dims[1];
            out_data.resize(size_1d);
            hsize_t offset[2] = {static_cast<hsize_t>(slice_index), 0};
            hsize_t count[2] = {1, size_1d};
            H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET, offset, nullptr, count, nullptr);
            hid_t mem_space_id = H5Screate_simple(1, &size_1d, nullptr);
            status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, mem_space_id, file_space_id, H5P_DEFAULT, out_data.data());
            H5Sclose(mem_space_id);
        }
    }

    H5Sclose(file_space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return (status >= 0);
}

// ============================================================================
// Simulation Runner
// ============================================================================

template <int DIM>
void run_simulation(CudaBuffer<float>& d_u, CudaBuffer<float>& d_v, CudaBuffer<float>& d_w, std::vector<float>& h_u, std::vector<float>& h_v, std::vector<float>& h_w, const SimParams& params, int saveInterval, HDF5Writer& writer, Logger& logger)
{
    const int total_size = params.size_total();

    CudaBuffer<float> d_u_next(total_size);
    CudaBuffer<float> d_v_next(total_size);
    CudaBuffer<float> d_w_next(total_size);

    CudaTimer step_timer;
    int snapshot_idx = 1;

    if constexpr (DIM == 1) {
        int blocks = (params.N + 255) / 256;
        logger.log("Running 1D simulation: " + std::to_string(params.steps) + " steps, " + std::to_string(blocks) + " blocks");

        for (int step = 1; step < params.steps; ++step) {
            step_timer.start();

            gpu_explicit_1d<<<blocks, 256>>>(d_u_next.get(), d_v_next.get(), d_w_next.get(), d_u.get(), d_v.get(), d_w.get(), params, params.N, params.dt, params.r_u(), params.r_v(), params.r_w());
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaMemcpy(d_u.get(), d_u_next.get(), total_size * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_v.get(), d_v_next.get(), total_size * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_w.get(), d_w_next.get(), total_size * sizeof(float), cudaMemcpyDeviceToDevice));

            if (step % saveInterval == 0 || step == params.steps - 1) {
                CUDA_CHECK(cudaMemcpy(h_u.data(), d_u.get(), total_size * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_v.data(), d_v.get(), total_size * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_w.data(), d_w.get(), total_size * sizeof(float), cudaMemcpyDeviceToHost));
                writer.writeStep(snapshot_idx++, h_u.data(), h_v.data(), h_w.data());
                step_timer.stop();
                logger.log("Step " + std::to_string(step) + " (" + std::to_string(step_timer.elapsed_ms()) + " ms)");
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }

    } else if constexpr (DIM == 2) {
        dim3 threads(16, 16);
        dim3 blocks((params.N + 15) / 16, (params.N + 15) / 16);
        logger.log("Running 2D simulation: " + std::to_string(params.steps) + " steps, " + std::to_string(blocks.x) + "x" + std::to_string(blocks.y) + " blocks");

        for (int step = 1; step <= params.steps; ++step) {
            step_timer.start();

            gpu_explicit_2d<<<blocks, threads>>>(d_u_next.get(), d_v_next.get(), d_w_next.get(), d_u.get(), d_v.get(), d_w.get(), params, params.N, params.dt, params.r_u(), params.r_v(), params.r_w());
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaMemcpy(d_u.get(), d_u_next.get(), total_size * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_v.get(), d_v_next.get(), total_size * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_w.get(), d_w_next.get(), total_size * sizeof(float), cudaMemcpyDeviceToDevice));

            if (step % saveInterval == 0 || step == params.steps) {
                CUDA_CHECK(cudaMemcpy(h_u.data(), d_u.get(), total_size * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_v.data(), d_v.get(), total_size * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_w.data(), d_w.get(), total_size * sizeof(float), cudaMemcpyDeviceToHost));
                writer.writeStep(snapshot_idx++, h_u.data(), h_v.data(), h_w.data());
                step_timer.stop();
                logger.log("Step " + std::to_string(step) + " (" + std::to_string(step_timer.elapsed_ms()) + " ms)");
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char* argv[])
{
    try {
        // Parse config file from arguments (default: config.json)
        std::string config_file = "config.json";
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg[0] != '-') {
                config_file = arg;
                break;
            }
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        // Load configuration
        auto params = ConfigLoader::load(config_file);

        // Print startup info
        std::cout << "mFHN Solver\n";
        std::cout << "CUDA Device: " << get_cuda_device_info() << "\n";
        print_cuda_memory_info();
        std::cout << "Config: N=" << params.N << ", dt=" << params.dt << ", dx=" << params.dx << ", steps=" << params.steps << "\n";

        // Setup output directory
        std::stringstream ss_date, ss_time;
        ss_date << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d");
        if (!std::filesystem::exists("results")) {
            std::filesystem::create_directory("results");
        }

        std::string tags = "_dim" + std::to_string(params.DIMENSION) + "_N" + std::to_string(params.N) + "_a" + std::to_string(params.model.a) + "_e2" + std::to_string(params.model.eps2) + "_e3" + std::to_string(params.model.eps3);
        ss_time << std::put_time(std::localtime(&in_time_t), "%H%M%S");
        std::string run_dir_name = ss_time.str() + tags;
        std::filesystem::path full_path = std::filesystem::path("results") / ss_date.str() / run_dir_name;

        std::filesystem::create_directories(full_path);
        try {
            std::filesystem::copy_file(config_file, full_path / "config.json");
        } catch (...) {
            // Ignore copy errors
        }

        // Initialize logger
        Logger logger(full_path.string() + "/simulation.log");
        logger.log("mFHN Solver");
        logger.log("Simulation started");
        logger.log("Config: N=" + std::to_string(params.N) + ", dt=" + std::to_string(params.dt) + ", dx=" + std::to_string(params.dx) + ", alpha=" + std::to_string(params.model.alpha) + ", eps3=" + std::to_string(params.model.eps3));
        logger.log("Stability check: " + std::string(params.is_stable() ? "PASSED" : "FAILED"));

        // Initialize host arrays
        std::vector<float> h_u(params.size_total()), h_v(params.size_total()), h_w(params.size_total());

        // Load or generate initial conditions
        bool loaded = false;
        if (!params.init_file.empty() && std::filesystem::exists(params.init_file.c_str())) {
            int slice_row = params.DIMENSION == 1 ? (params.N / 2) : 0;
            loaded = load_hdf5_data(params.init_file, "u", params.DIMENSION, slice_row, h_u);
        }

        if (!loaded) {
            if (params.DIMENSION == 1) {
                generate_1d(h_u, h_v, h_w, params);
            } else {
                generate_2d(h_u, h_v, h_w, params);
            }
        }

        logger.logMemoryUsage(params.N);

        // Allocate device memory
        CudaBuffer<float> d_u(params.size_total());
        CudaBuffer<float> d_v(params.size_total());
        CudaBuffer<float> d_w(params.size_total());

        CUDA_CHECK(cudaMemcpy(d_u.get(), h_u.data(), params.size_total() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v.get(), h_v.data(), params.size_total() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w.get(), h_w.data(), params.size_total() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Setup HDF5 writer
        int saveInterval = params.steps / params.num_snapshots;
        HDF5Writer writer(full_path.string() + "/result.h5", params.N, params.num_snapshots + 1, params.DIMENSION);
        writer.writeMetadata(params);

        int snapshot_idx = 0;
        writer.writeStep(snapshot_idx++, h_u.data(), h_v.data(), h_w.data());
        logger.log("Save interval: " + std::to_string(saveInterval) + " steps");

        // Run simulation
        if (params.DIMENSION == 1) {
            run_simulation<1>(d_u, d_v, d_w, h_u, h_v, h_w, params, saveInterval, writer, logger);
        } else if (params.DIMENSION == 2) {
            run_simulation<2>(d_u, d_v, d_w, h_u, h_v, h_w, params, saveInterval, writer, logger);
        } else {
            logger.log("ERROR: Unknown dimension!");
            std::cerr << "Error: Dimension must be 1 or 2, got " << params.DIMENSION << std::endl;
            return 1;
        }

        // Report final statistics
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
        double avg_step_ms = (elapsed_sec * 1000.0) / params.steps;

        logger.log("Results saved in `result.h5` in " + full_path.string());
        logger.log("Total time: " + std::to_string(elapsed_sec) + " seconds");
        logger.log("Average step time: " + std::to_string(avg_step_ms) + " ms");

        // Print summary
        std::cout << "\n=== Simulation Complete ===\n";
        std::cout << "Total time: " << elapsed_sec << " s\n";
        std::cout << "Average step: " << avg_step_ms << " ms\n";
        std::cout << "Results: " << full_path.string() << "/result.h5\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
