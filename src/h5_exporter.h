#pragma once

#include <cuda_runtime.h>
#include <hdf5.h>

#include <string>
#include <vector>

class HDF5Exporter {
   public:
    HDF5Exporter(const std::string& filename, int total_snapshots, int N, float dx, float dt);
    HDF5Exporter(const std::string& filename, int total_snapshots, int Nx, int Ny, float dx, float dt);
    ~HDF5Exporter();

    void save_step(int step_idx, const float* d_u, const float* d_v, const float* d_w, float time);
    void close();

   private:
    hid_t file_id;
    hid_t dataset_u, dataset_v, dataset_w, dataset_time;
    int rank;
    int N1d;
    int Nx, Ny;
    int total_snapshots;
    float dx, dt;
    std::vector<float> h_buffer;
    bool is_open;

    void create_datasets();
    void write_slice(hid_t dataset, int step_idx, const float* d_data);
};