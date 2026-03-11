#include "h5_exporter.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

HDF5Exporter::HDF5Exporter(const std::string& filename, int total_snapshots, int N, float dx_, float dt_) : rank(1), N1d(N), Nx(N), Ny(1), total_snapshots(total_snapshots), dx(dx_), dt(dt_), is_open(true) {
    // Создание файла
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Cannot create HDF5 file: " + filename);
    }

    // Сохранение атрибутов
    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t attr;

    attr = H5Acreate2(file_id, "dim", H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, &rank);
    H5Aclose(attr);

    attr = H5Acreate2(file_id, "N", H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, &N1d);
    H5Aclose(attr);

    attr = H5Acreate2(file_id, "dx", H5T_NATIVE_FLOAT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_FLOAT, &dx);
    H5Aclose(attr);

    attr = H5Acreate2(file_id, "dt", H5T_NATIVE_FLOAT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_FLOAT, &dt);
    H5Aclose(attr);

    attr = H5Acreate2(file_id, "num_snapshots", H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, &total_snapshots);
    H5Aclose(attr);

    H5Sclose(attr_space);

    create_datasets();
    std::cout << "[HDF5] Created 1D output file: " << filename << " (" << total_snapshots << " snapshots, N=" << N1d << ")" << std::endl;
}

HDF5Exporter::HDF5Exporter(const std::string& filename, int total_snapshots, int Nx_, int Ny_, float dx_, float dt_) : rank(2), N1d(Nx_ * Ny_), Nx(Nx_), Ny(Ny_), total_snapshots(total_snapshots), dx(dx_), dt(dt_), is_open(true) {
    // Создание файла
    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Cannot create HDF5 file: " + filename);
    }

    // Сохранение атрибутов
    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t attr;

    attr = H5Acreate2(file_id, "dim", H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, &rank);
    H5Aclose(attr);

    attr = H5Acreate2(file_id, "Nx", H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, &Nx);
    H5Aclose(attr);

    attr = H5Acreate2(file_id, "Ny", H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, &Ny);
    H5Aclose(attr);

    attr = H5Acreate2(file_id, "dx", H5T_NATIVE_FLOAT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_FLOAT, &dx);
    H5Aclose(attr);

    attr = H5Acreate2(file_id, "dt", H5T_NATIVE_FLOAT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_FLOAT, &dt);
    H5Aclose(attr);

    attr = H5Acreate2(file_id, "num_snapshots", H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, &total_snapshots);
    H5Aclose(attr);

    H5Sclose(attr_space);

    create_datasets();
    std::cout << "[HDF5] Created 2D output file: " << filename << " (" << total_snapshots << " snapshots, " << Nx << "x" << Ny << ")" << std::endl;
}

void HDF5Exporter::create_datasets() {
    hsize_t dims_u[3], dims_time[1];

    if (rank == 1) {
        // 1D: (num_snapshots, N)
        dims_u[0] = total_snapshots;
        dims_u[1] = N1d;
        dims_u[2] = 1;  // dummy dimension for unified handling
    } else {
        // 2D: (num_snapshots, Ny, Nx)
        dims_u[0] = total_snapshots;
        dims_u[1] = Ny;
        dims_u[2] = Nx;
    }

    dims_time[0] = total_snapshots;

    // Создание пространств
    hid_t filespace_u = H5Screate_simple(rank + 1, dims_u, nullptr);
    hid_t filespace_time = H5Screate_simple(1, dims_time, nullptr);

    // Создание датасетов
    dataset_u = H5Dcreate2(file_id, "u", H5T_NATIVE_FLOAT, filespace_u, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dataset_v = H5Dcreate2(file_id, "v", H5T_NATIVE_FLOAT, filespace_u, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dataset_w = H5Dcreate2(file_id, "w", H5T_NATIVE_FLOAT, filespace_u, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dataset_time = H5Dcreate2(file_id, "time", H5T_NATIVE_FLOAT, filespace_time, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Закрытие пространств
    H5Sclose(filespace_u);
    H5Sclose(filespace_time);

    // Выделение буфера (максимальный размер для 2D)
    h_buffer.resize(Nx * Ny);
}

void HDF5Exporter::write_slice(hid_t dataset, int step_idx, const float* d_data) {
    // Копирование данных с GPU в хост
    size_t size = (rank == 1) ? N1d * sizeof(float) : Nx * Ny * sizeof(float);
    cudaMemcpy(h_buffer.data(), d_data, size, cudaMemcpyDeviceToHost);

    // Определение гиперслаба для записи
    hsize_t start[3], count[3];

    if (rank == 1) {
        start[0] = step_idx;
        start[1] = 0;
        start[2] = 0;
        count[0] = 1;
        count[1] = N1d;
        count[2] = 1;
    } else {
        start[0] = step_idx;
        start[1] = 0;
        start[2] = 0;
        count[0] = 1;
        count[1] = Ny;
        count[2] = Nx;
    }

    hid_t filespace = H5Dget_space(dataset);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);

    // Пространство памяти для буфера
    hsize_t mem_dims[2] = {(hsize_t)Ny, (hsize_t)Nx};
    hid_t memspace = (rank == 1) ? H5Screate_simple(1, &count[1], nullptr) : H5Screate_simple(2, mem_dims, nullptr);

    // Запись данных
    H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, h_buffer.data());

    // Очистка
    H5Sclose(memspace);
    H5Sclose(filespace);
}

void HDF5Exporter::save_step(int step_idx, const float* d_u, const float* d_v, const float* d_w, float time) {
    if (!is_open || step_idx >= total_snapshots) return;

    // Запись временной метки
    {
        hid_t filespace = H5Dget_space(dataset_time);
        hsize_t start[1] = {(hsize_t)step_idx};
        hsize_t count[1] = {1};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);

        hid_t memspace = H5Screate_simple(1, count, nullptr);
        H5Dwrite(dataset_time, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, &time);

        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    // Запись компонент
    write_slice(dataset_u, step_idx, d_u);
    write_slice(dataset_v, step_idx, d_v);
    write_slice(dataset_w, step_idx, d_w);

    // Форсирование записи на диск каждые 10 шагов для надёжности
    if (step_idx % 10 == 0 || step_idx == total_snapshots - 1) {
        H5Fflush(file_id, H5F_SCOPE_GLOBAL);
    }
}

void HDF5Exporter::close() {
    if (!is_open) return;

    H5Dclose(dataset_u);
    H5Dclose(dataset_v);
    H5Dclose(dataset_w);
    H5Dclose(dataset_time);
    H5Fclose(file_id);
    is_open = false;
}

HDF5Exporter::~HDF5Exporter() {
    if (is_open) close();
}