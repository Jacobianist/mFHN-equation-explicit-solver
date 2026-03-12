#pragma once

#include <hdf5.h>

#include <string>
#include <vector>

class HDF5Writer
{
   public:
    HDF5Writer(const std::string& filename, int N, int total_snapshots, int dim);
    ~HDF5Writer();

    void writeStep(int step_idx, const float* h_u, const float* h_v, const float* h_w);

   private:
    hid_t file_id_;
    hid_t dataset_u_, dataset_v_, dataset_w_;
    hid_t filespace_;
    hid_t memspace_;

    int N_;
    int total_snapshots_;
    int dim_;
};
