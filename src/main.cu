#include <cuda_runtime.h>
#include <hdf5.h>

// #include <chrono>
#include <filesystem>
#include <iostream>
// #include <ostream>
#include <random>
#include <sstream>
#include <string>

#include "config.h"
#include "h5_exporter.h"
#include "logger.h"
#include "solver_explicit.cuh"

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
bool load_hdf5_data(const std::string& filename, const std::string& dataset_name, int target_dim, int slice_index, std::vector<float>& out_data)
{
    // 1. Открываем файл и датасет
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
    // 2. Получаем размерности файла (чтобы узнать N)
    hid_t file_space_id = H5Dget_space(dataset_id);
    int ndims = H5Sget_simple_extent_ndims(file_space_id);

    if (ndims != 2) {
        std::cerr << "Error: Dataset is not 2D!" << std::endl;
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
                hsize_t size_1d = dims[1];  // Длина строки
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
    // 6. Закрываем всё
    H5Sclose(file_space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return (status >= 0);
}

int main()
{
    auto t_start = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto params = ConfigLoader::load("config.json");
    std::stringstream ss_date, ss_time;
    ss_date << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d");
    if (!std::filesystem::exists("results")) std::filesystem::create_directory("results");

    std::string tags = "";
    tags += "_dim" + std::to_string(params.DIMENSION);
    tags += "_N" + std::to_string(params.N);
    tags += "_a" + std::to_string(params.model.a);
    tags += "_e2" + std::to_string(params.model.eps2);
    tags += "_e3" + std::to_string(params.model.eps3);
    ss_time << std::put_time(std::localtime(&in_time_t), "%H%M%S");
    std::string run_dir_name = ss_time.str() + tags;
    std::filesystem::path full_path = std::filesystem::path("results") / ss_date.str() / run_dir_name;

    std::filesystem::create_directories(full_path);
    try {
        std::filesystem::copy_file("config.json", full_path / "config.json");
    } catch (...) {
        std::cerr << "Warning: Could not copy config.json" << std::endl;
    }
    Logger logger(full_path.string() + "/simulation.log");
    std::cout << "Config loaded. N=" << params.N << ", dt=" << params.dt << ", dx=" << params.dx << ", alpha=" << params.model.alpha << ", ksi=" << params.dt / (2 * params.dx * params.dx) << std::endl;
    std::vector<float> h_u(params.size_total()), h_v(params.size_total()), h_w(params.size_total());

    bool loaded = false;
    if (std::filesystem::exists(params.init_file.c_str())) {
        std::cout << "Loading initial conditions from " << params.init_file << " (Dim: " << params.DIMENSION << ")..." << std::endl;
        int slice_row = params.DIMENSION == 1 ? (params.N / 2) : 0;  // Например, середина
        loaded = load_hdf5_data(params.init_file, "u", params.DIMENSION, slice_row, h_u);
        // if (loaded) loaded = load_hdf5_data(params.init_file, "v", params.dim, slice_row, h_v);
        // if (loaded) loaded = load_hdf5_data(params.init_file, "w", params.dim, slice_row, h_w);
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

    int saveInterval = params.steps / params.num_snapshots;
    HDF5Writer writer(full_path.string() + "/result.h5", params.N, params.num_snapshots + 1, params.DIMENSION);
    int snapshot_idx = 0;
    writer.writeStep(snapshot_idx++, h_u.data(), h_v.data(), h_w.data());
    logger.log("Save interval: " + std::to_string(saveInterval) + " steps.");

    if (params.DIMENSION == 1) {
        int blocks = (params.N + 255) / 256;
        for (int ITERATION_STEP = 0; ITERATION_STEP < params.steps; ++ITERATION_STEP) {
            gpu_explicit_1d<<<blocks, 256>>>(d_u_next, d_v_next, d_w_next, d_u, d_v, d_w, params, params.N, params.dt, params.r_u(), params.r_v(), params.r_w());
            std::swap(d_u, d_u_next);
            std::swap(d_v, d_v_next);
            std::swap(d_w, d_w_next);
            if (ITERATION_STEP % saveInterval == 0 || ITERATION_STEP == params.steps) {
                cudaMemcpy(h_u.data(), d_u, params.size_total() * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_v.data(), d_v, params.size_total() * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_w.data(), d_w, params.size_total() * sizeof(float), cudaMemcpyDeviceToHost);
                writer.writeStep(snapshot_idx++, h_u.data(), h_v.data(), h_w.data());
                logger.log("Step " + std::to_string(ITERATION_STEP));
            }
            cudaDeviceSynchronize();
        }
    } else if (params.DIMENSION == 2) {
        dim3 threads(16, 16);
        dim3 blocks((params.N + 15) / 16, (params.N + 15) / 16);
        for (int ITERATION_STEP = 1; ITERATION_STEP <= params.steps; ++ITERATION_STEP) {
            gpu_explicit_2d<<<blocks, threads>>>(d_u_next, d_v_next, d_w_next, d_u, d_v, d_w, params, params.N, params.dt, params.r_u(), params.r_v(), params.r_w());
            std::swap(d_u, d_u_next);
            std::swap(d_v, d_v_next);
            std::swap(d_w, d_w_next);
            if (ITERATION_STEP % saveInterval == 0 || ITERATION_STEP == params.steps) {
                cudaMemcpy(h_u.data(), d_u, params.size_total() * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_v.data(), d_v, params.size_total() * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_w.data(), d_w, params.size_total() * sizeof(float), cudaMemcpyDeviceToHost);
                writer.writeStep(snapshot_idx++, h_u.data(), h_v.data(), h_w.data());
                logger.log("Step " + std::to_string(ITERATION_STEP));
            }
            cudaDeviceSynchronize();
        }
    } else {
        logger.log("ERROR: Unknown dimension!");
    }

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_u_next);
    cudaFree(d_v_next);
    cudaFree(d_w_next);

    logger.log("Results saved in `result.h5` in " + full_path.string());
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
    logger.log("Total time: " + std::to_string(elapsed_sec) + " seconds.");
}
