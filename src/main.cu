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
 * @brief Main entry point for the mFHN explicit solver with improved error handling
 */

// ============================================================================
// Initial Condition Generators
// ============================================================================

/**
 * @brief Generates 1D initial conditions for the simulation.
 *
 * Creates a localized perturbation (pulse) in the center of the domain:
 * - Inside radius: u ≈ 1.0 (activator), v ≈ 0.0, w ≈ 0.0 (inhibitors)
 * - Outside radius: u ≈ 0.0, v ≈ 0.0, w ≈ 0.0 (rest state)
 * Small random noise is added to mimic perturbations.
 *
 * @param u Output array for activator variable
 * @param v Output array for inhibitor v variable
 * @param w Output array for inhibitor w variable
 * @param params Simulation parameters (N, dx)
 */
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

/**
 * @brief Generates 2D initial conditions for the simulation.
 *
 * Creates a circular localized perturbation (spot) in the center of the domain:
 * - Inside radius: u ≈ 1.0 (activator), v ≈ 0.0, w ≈ 0.0 (inhibitors)
 * - Outside radius: u ≈ 0.0, v ≈ 0.0, w ≈ 0.0 (rest state)
 * Small random noise is added to mimic perturbations.
 *
 * @param u Output array for activator variable (size N×N)
 * @param v Output array for inhibitor v variable
 * @param w Output array for inhibitor w variable
 * @param params Simulation parameters (N, dx)
 */
void generate_2d(std::vector<float>& u, std::vector<float>& v, std::vector<float>& w, const SimParams& params)
{
    float center = params.dx * params.N / 2.0f;
    float radius = 16.0f * params.dx;  // Spot radius

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

/**
 * @brief Loads initial condition data from an HDF5 file.
 *
 * Reads a dataset from an HDF5 file and extracts either:
 * - 1D data: a row slice from a 2D dataset, or the full 1D dataset
 * - 2D data: the complete 2D dataset
 *
 * @param filename Path to the HDF5 file
 * @param dataset_name Name of the dataset to read (e.g., "u", "v", "w")
 * @param target_dim Target dimension (1 for 1D simulation, 2 for 2D)
 * @param slice_index Row index to extract for 1D from 2D data
 * @param out_data Output vector to store the loaded data
 * @return true if loading succeeded, false otherwise
 */
bool load_hdf5_data(const std::string& filename, const std::string& dataset_name, int target_dim, int slice_index, std::vector<float>& out_data)
{
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Warning: Cannot open file " << filename << std::endl;
        return false;
    }

    hid_t dataset_id = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        H5Fclose(file_id);
        return false;
    }

    hid_t file_space_id = H5Dget_space(dataset_id);
    int ndims = H5Sget_simple_extent_ndims(file_space_id);

    if (ndims != 2) {
        std::cerr << "Error: Dataset is not 2D!" << std::endl;
        H5Sclose(file_space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return false;
    }

    hsize_t dims[3] = {0, 0, 0};  // {Rows, Cols}
    H5Sget_simple_extent_dims(file_space_id, dims, nullptr);

    herr_t status = -1;
    if (target_dim == 2) {
        if (ndims == 2) {
            hsize_t total_size = dims[0] * dims[1];
            out_data.resize(total_size);
            hid_t mem_space_id = H5Screate_simple(2, dims, nullptr);
            status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, mem_space_id, file_space_id, H5P_DEFAULT, out_data.data());
            H5Sclose(mem_space_id);
        } else {
            std::cerr << "Error: Expecting 2D data in file, but found " << ndims << "D." << std::endl;
        }
    } else if (target_dim == 1) {
        if (ndims == 1) {
            out_data.resize(dims[0]);
            hid_t mem_space_id = H5Screate_simple(1, dims, nullptr);
            status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, mem_space_id, file_space_id, H5P_DEFAULT, out_data.data());
            H5Sclose(mem_space_id);
        } else if (ndims == 2) {
            if (slice_index < 0 || slice_index >= (int)dims[0]) {
                std::cerr << "Error: slice_index " << slice_index << " is out of bounds [0, " << dims[0] << "]" << std::endl;
            } else {
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
    }

    H5Sclose(file_space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return (status >= 0);
}

// ============================================================================
// Print Usage Information
// ============================================================================

void print_usage(const char* program_name)
{
    std::cout << "Usage: " << program_name << " [config_file]\n"
              << "\n"
              << "Arguments:\n"
              << "  config_file    Path to JSON configuration file (default: config.json)\n"
              << "\n"
              << "Example:\n"
              << "  " << program_name << " config.json\n"
              << "  " << program_name << " my_simulation.json\n"
              << std::endl;
}

// ============================================================================
// Main Entry Point
// ============================================================================

/**
 * @brief Main entry point for the mFHN explicit solver.
 *
 * Execution flow:
 * 1. Load configuration from config.json or json file from passed argument
 * 2. Create output directory structure (results/YYYY-MM-DD/HHMMSS_...)
 * 3. Initialize or load initial conditions
 * 4. Allocate CUDA device memory and transfer data
 * 5. Run time-stepping loop with RK4 + finite differences
 * 6. Save snapshots at regular intervals to HDF5
 * 7. Clean up resources and report timing
 *
 * Output:
 * - results/YYYY-MM-DD/HHMMSS_dimN_N.../result.h5 - simulation data with metadata
 * - results/YYYY-MM-DD/HHMMSS_dimN_N.../simulation.log - log file
 * - results/YYYY-MM-DD/HHMMSS_dimN_N.../config.json - copied config
 */
int main(int argc, char* argv[])
{
    try {
        // Parse command-line arguments
        std::string config_file = "config.json";
        if (argc > 1) {
            if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
                print_usage(argv[0]);
                return 0;
            }
            config_file = argv[1];
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        // Load configuration
        auto params = ConfigLoader::load(config_file);

        // Print CUDA device info
        std::cout << "mFHN Solver - CUDA Device: " << get_cuda_device_info() << "\n";
        print_cuda_memory_info();

        // Setup output directory
        std::stringstream ss_date, ss_time;
        ss_date << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d");
        if (!std::filesystem::exists("results")) {
            std::filesystem::create_directory("results");
        }

        std::string tags = "_dim" + std::to_string(params.DIMENSION) +
                          "_N" + std::to_string(params.N) +
                          "_a" + std::to_string(params.model.a) +
                          "_e2" + std::to_string(params.model.eps2) +
                          "_e3" + std::to_string(params.model.eps3);
        ss_time << std::put_time(std::localtime(&in_time_t), "%H%M%S");
        std::string run_dir_name = ss_time.str() + tags;
        std::filesystem::path full_path = std::filesystem::path("results") / ss_date.str() / run_dir_name;

        std::filesystem::create_directories(full_path);
        try {
            std::filesystem::copy_file(config_file, full_path / "config.json");
        } catch (...) {
            std::cerr << "Warning: Could not copy config.json" << std::endl;
        }

        Logger logger(full_path.string() + "/simulation.log");
        logger.log("Simulation started");
        logger.log("Config: N=" + std::to_string(params.N) +
                  ", dt=" + std::to_string(params.dt) +
                  ", dx=" + std::to_string(params.dx) +
                  ", alpha=" + std::to_string(params.model.alpha));
        logger.log("Stability check: " + std::string(params.is_stable() ? "PASSED" : "FAILED"));

        std::cout << "Config loaded. N=" << params.N
                  << ", dt=" << params.dt
                  << ", dx=" << params.dx
                  << ", alpha=" << params.model.alpha
                  << ", ksi=" << params.dt / (2 * params.dx * params.dx) << std::endl;

        // Initialize host arrays
        std::vector<float> h_u(params.size_total()), h_v(params.size_total()), h_w(params.size_total());

        // Load or generate initial conditions
        bool loaded = false;
        if (std::filesystem::exists(params.init_file.c_str())) {
            std::cout << "Loading initial conditions from " << params.init_file << "..." << std::endl;
            int slice_row = params.DIMENSION == 1 ? (params.N / 2) : 0;
            loaded = load_hdf5_data(params.init_file, "u", params.DIMENSION, slice_row, h_u);
        }

        if (!loaded) {
            std::cout << "Generating default initial conditions..." << std::endl;
            if (params.DIMENSION == 1) {
                generate_1d(h_u, h_v, h_w, params);
            } else {
                generate_2d(h_u, h_v, h_w, params);
            }
        }

        logger.logMemoryUsage(params.N);

        // Allocate device memory using RAII wrappers
        CudaBuffer<float> d_u(params.size_total());
        CudaBuffer<float> d_u_next(params.size_total());
        CudaBuffer<float> d_v(params.size_total());
        CudaBuffer<float> d_v_next(params.size_total());
        CudaBuffer<float> d_w(params.size_total());
        CudaBuffer<float> d_w_next(params.size_total());

        // Transfer initial conditions to device
        CUDA_CHECK(cudaMemcpy(d_u.get(), h_u.data(), params.size_total() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v.get(), h_v.data(), params.size_total() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w.get(), h_w.data(), params.size_total() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Setup HDF5 writer and save metadata
        int saveInterval = params.steps / params.num_snapshots;
        HDF5Writer writer(full_path.string() + "/result.h5", params.N, params.num_snapshots + 1, params.DIMENSION);
        writer.writeMetadata(params);  // Write simulation metadata

        int snapshot_idx = 0;
        writer.writeStep(snapshot_idx++, h_u.data(), h_v.data(), h_w.data());  // Save initial state
        logger.log("Save interval: " + std::to_string(saveInterval) + " steps.");

        // Create CUDA timer for performance measurement
        CudaTimer step_timer;

        // Main simulation loop
        if (params.DIMENSION == 1) {
            // 1D simulation: 256 threads per block
            int blocks = (params.N + 255) / 256;
            logger.log("Running 1D simulation: " + std::to_string(params.steps) + " steps, " +
                      std::to_string(blocks) + " blocks");

            for (int ITERATION_STEP = 1; ITERATION_STEP < params.steps; ++ITERATION_STEP) {
                step_timer.start();
                gpu_explicit_1d<<<blocks, 256>>>(
                    d_u_next.get(), d_v_next.get(), d_w_next.get(),
                    d_u.get(), d_v.get(), d_w.get(),
                    params, params.N, params.dt,
                    params.r_u(), params.r_v(), params.r_w()
                );
                CUDA_CHECK_KERNEL();

                // Swap current and next arrays (ping-pong buffering)
                std::swap(d_u, d_u_next);
                std::swap(d_v, d_v_next);
                std::swap(d_w, d_w_next);

                if (ITERATION_STEP % saveInterval == 0 || ITERATION_STEP == params.steps - 1) {
                    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u.get(), params.size_total() * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_v.data(), d_v.get(), params.size_total() * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_w.data(), d_w.get(), params.size_total() * sizeof(float), cudaMemcpyDeviceToHost));
                    writer.writeStep(snapshot_idx++, h_u.data(), h_v.data(), h_w.data());
                    step_timer.stop();
                    logger.log("Step " + std::to_string(ITERATION_STEP) +
                              " (" + std::to_string(step_timer.elapsed_ms()) + " ms)");
                }
                CUDA_CHECK(cudaDeviceSynchronize());
            }

        } else if (params.DIMENSION == 2) {
            // 2D simulation: 16x16 threads per block (256 threads)
            dim3 threads(16, 16);
            dim3 blocks((params.N + 15) / 16, (params.N + 15) / 16);
            logger.log("Running 2D simulation: " + std::to_string(params.steps) + " steps, " +
                      std::to_string(blocks.x) + "x" + std::to_string(blocks.y) + " blocks");

            for (int ITERATION_STEP = 1; ITERATION_STEP <= params.steps; ++ITERATION_STEP) {
                step_timer.start();
                gpu_explicit_2d<<<blocks, threads>>>(
                    d_u_next.get(), d_v_next.get(), d_w_next.get(),
                    d_u.get(), d_v.get(), d_w.get(),
                    params, params.N, params.dt,
                    params.r_u(), params.r_v(), params.r_w()
                );
                CUDA_CHECK_KERNEL();

                // Swap current and next arrays
                std::swap(d_u, d_u_next);
                std::swap(d_v, d_v_next);
                std::swap(d_w, d_w_next);

                if (ITERATION_STEP % saveInterval == 0 || ITERATION_STEP == params.steps) {
                    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u.get(), params.size_total() * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_v.data(), d_v.get(), params.size_total() * sizeof(float), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_w.data(), d_w.get(), params.size_total() * sizeof(float), cudaMemcpyDeviceToHost));
                    writer.writeStep(snapshot_idx++, h_u.data(), h_v.data(), h_w.data());
                    step_timer.stop();
                    logger.log("Step " + std::to_string(ITERATION_STEP) +
                              " (" + std::to_string(step_timer.elapsed_ms()) + " ms)");
                }
                CUDA_CHECK(cudaDeviceSynchronize());
            }

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

        std::cout << "\n=== Simulation Complete ===\n"
                  << "Total time: " << elapsed_sec << " s\n"
                  << "Average step: " << avg_step_ms << " ms\n"
                  << "Results: " << full_path.string() << "/result.h5\n"
                  << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
