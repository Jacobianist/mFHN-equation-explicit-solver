#include "h5_exporter.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

/**
 * @brief Write a string attribute to the HDF5 file
 */
void HDF5Writer::writeAttribute(const std::string& name, const std::string& value)
{
    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t attr_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(attr_type, value.size() + 1);
    H5Tset_strpad(attr_type, H5T_STR_NULLTERM);
    
    hid_t attr = H5Acreate2(file_id_, name.c_str(), attr_type, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, attr_type, value.c_str());
    
    H5Aclose(attr);
    H5Sclose(attr_space);
    H5Tclose(attr_type);
}

/**
 * @brief Write a float attribute to the HDF5 file
 */
void HDF5Writer::writeAttribute(const std::string& name, float value)
{
    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(file_id_, name.c_str(), H5T_NATIVE_FLOAT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_FLOAT, &value);
    H5Aclose(attr);
    H5Sclose(attr_space);
}

/**
 * @brief Write an int attribute to the HDF5 file
 */
void HDF5Writer::writeAttribute(const std::string& name, int value)
{
    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(file_id_, name.c_str(), H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, &value);
    H5Aclose(attr);
    H5Sclose(attr_space);
}

/**
 * @brief Write simulation metadata as HDF5 file attributes
 */
void HDF5Writer::writeMetadata(const SimParams& params)
{
    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
    std::string timestamp = ss.str();

    // File-level metadata
    writeAttribute("creation_timestamp", timestamp);
    writeAttribute("dimension", params.DIMENSION);
    writeAttribute("grid_size_N", params.N);
    writeAttribute("dx", params.dx);
    writeAttribute("dt", params.dt);
    writeAttribute("total_steps", params.steps);
    writeAttribute("num_snapshots", params.num_snapshots);
    writeAttribute("output_directory", params.output_dir);
    writeAttribute("simulation_name", params.name);
    writeAttribute("init_file", params.init_file);

    // Model parameters
    writeAttribute("model_a", params.model.a);
    writeAttribute("model_b", params.model.b);
    writeAttribute("model_c", params.model.c);
    writeAttribute("model_alpha", params.model.alpha);
    writeAttribute("model_phi", params.model.phi);
    writeAttribute("model_eps2", params.model.eps2);
    writeAttribute("model_eps3", params.model.eps3);
    writeAttribute("model_D1", params.model.D1);
    writeAttribute("model_D2", params.model.D2);
    writeAttribute("model_D3", params.model.D3);

    // Derived parameters
    writeAttribute("r_u", params.r_u());
    writeAttribute("r_v", params.r_v());
    writeAttribute("r_w", params.r_w());
    writeAttribute("is_stable", params.is_stable() ? 1 : 0);
}

/**
 * @brief Constructs and initializes the HDF5 writer.
 *
 * Creates an HDF5 file with three datasets (u, v, w) for storing simulation results.
 * Uses chunked storage with deflate compression (level 6) for efficient I/O.
 *
 * For 1D: datasets have shape (total_snapshots, N)
 * For 2D: datasets have shape (total_snapshots, N, N)
 *
 * @param filename Path to the output HDF5 file
 * @param N Grid size per dimension
 * @param total_snapshots Total number of snapshots to store
 * @param dim Spatial dimension (must be 1 or 2)
 * @throws std::runtime_error if file creation fails or dim is invalid
 */
HDF5Writer::HDF5Writer(const std::string& filename, int N, int total_snapshots, int dim)
    : N_(N), total_snapshots_(total_snapshots), dim_(dim), file_id_(-1),
      dataset_u_(-1), dataset_v_(-1), dataset_w_(-1), filespace_(-1), memspace_(-1)
{
    file_id_ = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id_ < 0) {
        throw std::runtime_error("Failed to create HDF5 file: " + filename);
    }

    herr_t status;
    if (dim_ == 1) {
        // 1D simulation: create datasets with shape (snapshots, N)
        hsize_t dims_file[2] = {static_cast<hsize_t>(total_snapshots_), static_cast<hsize_t>(N_)};
        hsize_t chunk_dims[2] = {1, static_cast<hsize_t>(N_)};
        filespace_ = H5Screate_simple(2, dims_file, nullptr);
        memspace_ = H5Screate_simple(1, dims_file + 1, nullptr);

        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist, 2, chunk_dims);
        H5Pset_deflate(plist, 6);  // Compression level 6

        dataset_u_ = H5Dcreate(file_id_, "u", H5T_NATIVE_FLOAT, filespace_, H5P_DEFAULT, plist, H5P_DEFAULT);
        dataset_v_ = H5Dcreate(file_id_, "v", H5T_NATIVE_FLOAT, filespace_, H5P_DEFAULT, plist, H5P_DEFAULT);
        dataset_w_ = H5Dcreate(file_id_, "w", H5T_NATIVE_FLOAT, filespace_, H5P_DEFAULT, plist, H5P_DEFAULT);

        H5Pclose(plist);

    } else if (dim_ == 2) {
        // 2D simulation: create datasets with shape (snapshots, N, N)
        hsize_t dims_file[3] = {static_cast<hsize_t>(total_snapshots_), static_cast<hsize_t>(N_), static_cast<hsize_t>(N_)};
        hsize_t chunk_dims[3] = {1, static_cast<hsize_t>(N_), static_cast<hsize_t>(N_)};
        filespace_ = H5Screate_simple(3, dims_file, nullptr);

        hsize_t dims_mem[2] = {static_cast<hsize_t>(N_), static_cast<hsize_t>(N_)};
        memspace_ = H5Screate_simple(2, dims_mem, nullptr);

        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist, 3, chunk_dims);
        H5Pset_deflate(plist, 6);  // Compression level 6

        dataset_u_ = H5Dcreate(file_id_, "u", H5T_NATIVE_FLOAT, filespace_, H5P_DEFAULT, plist, H5P_DEFAULT);
        dataset_v_ = H5Dcreate(file_id_, "v", H5T_NATIVE_FLOAT, filespace_, H5P_DEFAULT, plist, H5P_DEFAULT);
        dataset_w_ = H5Dcreate(file_id_, "w", H5T_NATIVE_FLOAT, filespace_, H5P_DEFAULT, plist, H5P_DEFAULT);

        H5Pclose(plist);

    } else {
        H5Fclose(file_id_);
        throw std::runtime_error("HDF5Writer only supports dim=1 or dim=2");
    }
}

/**
 * @brief Destructor that closes all HDF5 resources.
 *
 * Properly closes datasets, dataspaces, and the file handle.
 */
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

/**
 * @brief Writes a single time step snapshot to the HDF5 file.
 *
 * Uses hyperslab selection to write data at the specified snapshot index.
 * Writes all three variables (u, v, w) sequentially.
 *
 * @param step_idx Snapshot index to write (0-based, must be < total_snapshots)
 * @param h_u Host array containing activator variable data
 * @param h_v Host array containing inhibitor v variable data
 * @param h_w Host array containing inhibitor w variable data
 */
void HDF5Writer::writeStep(int step_idx, const float* h_u, const float* h_v, const float* h_w)
{
    if (step_idx >= total_snapshots_ || step_idx < 0) return;

    hid_t filespace = H5Dget_space(dataset_u_);

    if (dim_ == 1) {
        // 1D: select hyperslab [step_idx, 0] with count [1, N]
        hsize_t offset[2] = {static_cast<hsize_t>(step_idx), 0};
        hsize_t count[2] = {1, static_cast<hsize_t>(N_)};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
    } else {  // dim_ == 2
        // 2D: select hyperslab [step_idx, 0, 0] with count [1, N, N]
        hsize_t offset[3] = {static_cast<hsize_t>(step_idx), 0, 0};
        hsize_t count[3] = {1, static_cast<hsize_t>(N_), static_cast<hsize_t>(N_)};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
    }

    // Write data for all three variables
    H5Dwrite(dataset_u_, H5T_NATIVE_FLOAT, memspace_, filespace, H5P_DEFAULT, h_u);
    H5Dwrite(dataset_v_, H5T_NATIVE_FLOAT, memspace_, filespace, H5P_DEFAULT, h_v);
    H5Dwrite(dataset_w_, H5T_NATIVE_FLOAT, memspace_, filespace, H5P_DEFAULT, h_w);

    H5Sclose(filespace);
}
