#include "h5_exporter.h"

#include <iostream>
#include <stdexcept>

HDF5Writer::HDF5Writer(const std::string& filename, int N, int total_snapshots, int dim) : N_(N), total_snapshots_(total_snapshots), dim_(dim), file_id_(-1)
{
    file_id_ = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id_ < 0) throw std::runtime_error("Failed to create HDF5 file");
    herr_t status;
    if (dim_ == 1) {
        hsize_t dims_file[2] = {static_cast<hsize_t>(total_snapshots_), static_cast<hsize_t>(N_)};
        hsize_t chunk_dims[2] = {1, static_cast<hsize_t>(N_)};
        filespace_ = H5Screate_simple(2, dims_file, nullptr);
        memspace_ = H5Screate_simple(1, dims_file + 1, nullptr);
        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist, 2, chunk_dims);
        H5Pset_deflate(plist, 6);
        dataset_u_ = H5Dcreate(file_id_, "u", H5T_NATIVE_FLOAT, filespace_, H5P_DEFAULT, plist, H5P_DEFAULT);
        dataset_v_ = H5Dcreate(file_id_, "v", H5T_NATIVE_FLOAT, filespace_, H5P_DEFAULT, plist, H5P_DEFAULT);
        dataset_w_ = H5Dcreate(file_id_, "w", H5T_NATIVE_FLOAT, filespace_, H5P_DEFAULT, plist, H5P_DEFAULT);
        H5Pclose(plist);
    } else if (dim_ == 2) {
        hsize_t dims_file[3] = {static_cast<hsize_t>(total_snapshots_), static_cast<hsize_t>(N_), static_cast<hsize_t>(N_)};
        hsize_t chunk_dims[3] = {1, static_cast<hsize_t>(N_), static_cast<hsize_t>(N_)};
        filespace_ = H5Screate_simple(3, dims_file, NULL);
        hsize_t dims_mem[2] = {static_cast<hsize_t>(N_), static_cast<hsize_t>(N_)};
        memspace_ = H5Screate_simple(2, dims_mem, NULL);
        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist, 3, chunk_dims);
        H5Pset_deflate(plist, 6);
        dataset_u_ = H5Dcreate(file_id_, "u", H5T_NATIVE_FLOAT, filespace_, H5P_DEFAULT, plist, H5P_DEFAULT);
        dataset_v_ = H5Dcreate(file_id_, "v", H5T_NATIVE_FLOAT, filespace_, H5P_DEFAULT, plist, H5P_DEFAULT);
        dataset_w_ = H5Dcreate(file_id_, "w", H5T_NATIVE_FLOAT, filespace_, H5P_DEFAULT, plist, H5P_DEFAULT);
        H5Pclose(plist);
    } else {
        throw std::runtime_error("HDF5Writer only supports dim=1 or dim=2");
    }
}

HDF5Writer::~HDF5Writer()
{
    if (file_id_ >= 0) {
        if (dataset_u_ >= 0) H5Dclose(dataset_u_);
        if (dataset_v_ >= 0) H5Dclose(dataset_v_);
        if (dataset_w_ >= 0) H5Dclose(dataset_w_);
        if (filespace_ >= 0) H5Sclose(filespace_);
        if (memspace_ >= 0) H5Sclose(memspace_);
        H5Fclose(file_id_);
    }
}

void HDF5Writer::writeStep(int step_idx, const float* h_u, const float* h_v, const float* h_w)
{
    if (step_idx >= total_snapshots_) return;
    hid_t filespace = H5Dget_space(dataset_u_);
    if (dim_ == 1) {
        hsize_t offset[2] = {static_cast<hsize_t>(step_idx), 0};
        hsize_t count[2] = {1, static_cast<hsize_t>(N_)};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
    } else {  // dim_ == 2
        hsize_t offset[3] = {static_cast<hsize_t>(step_idx), 0, 0};
        hsize_t count[3] = {1, static_cast<hsize_t>(N_), static_cast<hsize_t>(N_)};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
    }
    H5Dwrite(dataset_u_, H5T_NATIVE_FLOAT, memspace_, filespace, H5P_DEFAULT, h_u);
    H5Dwrite(dataset_v_, H5T_NATIVE_FLOAT, memspace_, filespace, H5P_DEFAULT, h_v);
    H5Dwrite(dataset_w_, H5T_NATIVE_FLOAT, memspace_, filespace, H5P_DEFAULT, h_w);

    H5Sclose(filespace);
}
