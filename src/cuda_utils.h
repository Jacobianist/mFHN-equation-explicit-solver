#pragma once

#include <cuda_runtime.h>
#include <hdf5.h>

#include <iostream>
#include <stdexcept>
#include <string>

/**
 * @file cuda_utils.h
 * @brief CUDA error checking macros and RAII wrappers
 */

// ============================================================================
// CUDA Error Checking
// ============================================================================

/**
 * @brief Check CUDA API call for errors and throw exception on failure
 *
 * @param call CUDA function call to check
 * @throws std::runtime_error if CUDA error occurs
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - " + cudaGetErrorString(err) \
            ); \
        } \
    } while(0)

/**
 * @brief Check for CUDA kernel launch errors (asynchronous)
 *
 * Call after kernel launches to check for execution errors.
 *
 * @throws std::runtime_error if CUDA error occurs
 */
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA kernel error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - " + cudaGetErrorString(err) \
            ); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA device sync error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                " - " + cudaGetErrorString(err) \
            ); \
        } \
    } while(0)

// ============================================================================
// HDF5 Error Checking
// ============================================================================

/**
 * @brief Check HDF5 API call for errors and throw exception on failure
 *
 * @param call HDF5 function call to check
 * @throws std::runtime_error if HDF5 error occurs
 */
#define HDF5_CHECK(call, msg) \
    do { \
        herr_t result = call; \
        if (result < 0) { \
            H5Eprint(H5E_DEFAULT, stderr); \
            throw std::runtime_error(std::string(msg) + " - HDF5 error at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

// ============================================================================
// RAII Wrappers for CUDA Memory
// ============================================================================

/**
 * @brief RAII wrapper for CUDA device memory
 *
 * Automatically frees device memory when object goes out of scope.
 * Supports move semantics but disables copy to prevent double-free.
 *
 * @tparam T Data type stored in the buffer
 */
template<typename T>
class CudaBuffer {
private:
    T* ptr_;
    size_t size_;  // Number of elements (not bytes)

public:
    /**
     * @brief Allocate device memory
     * @param size Number of elements to allocate
     * @throws std::runtime_error if allocation fails
     */
    explicit CudaBuffer(size_t size) : ptr_(nullptr), size_(size) {
        if (size > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, size * sizeof(T)));
        }
    }

    /**
     * @brief Destructor - frees device memory
     */
    ~CudaBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    // Disable copy constructor and assignment (prevent double-free)
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Enable move semantics
    CudaBuffer(CudaBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Get raw pointer for CUDA API calls
     * @return Device pointer
     */
    T* get() const { return ptr_; }

    /**
     * @brief Get size in elements
     * @return Number of elements
     */
    size_t size() const { return size_; }

    /**
     * @brief Check if buffer is valid (non-null)
     * @return true if memory is allocated
     */
    bool valid() const { return ptr_ != nullptr; }
};

/**
 * @brief RAII wrapper for CUDA unified (managed) memory
 *
 * Uses cudaMallocManaged for automatic memory migration between host and device.
 * Useful for simpler code on newer GPUs (Compute Capability 6.0+).
 *
 * @tparam T Data type stored in the buffer
 */
template<typename T>
class CudaUnifiedBuffer {
private:
    T* ptr_;
    size_t size_;

public:
    explicit CudaUnifiedBuffer(size_t size) : ptr_(nullptr), size_(size) {
        if (size > 0) {
            CUDA_CHECK(cudaMallocManaged(&ptr_, size * sizeof(T)));
        }
    }

    ~CudaUnifiedBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    // Disable copy
    CudaUnifiedBuffer(const CudaUnifiedBuffer&) = delete;
    CudaUnifiedBuffer& operator=(const CudaUnifiedBuffer&) = delete;

    // Enable move
    CudaUnifiedBuffer(CudaUnifiedBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaUnifiedBuffer& operator=(CudaUnifiedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* get() const { return ptr_; }
    size_t size() const { return size_; }
    bool valid() const { return ptr_ != nullptr; }

    // Access operators for convenient host-side use
    T& operator[](size_t idx) { return ptr_[idx]; }
    const T& operator[](size_t idx) const { return ptr_[idx]; }
};

// ============================================================================
// CUDA Event Timer for Performance Measurement
// ============================================================================

/**
 * @brief RAII wrapper for CUDA event-based timing
 *
 * Measures GPU execution time with millisecond precision.
 *
 * @example
 * CudaTimer timer;
 * timer.start();
 * myKernel<<<blocks, threads>>>(...);
 * timer.stop();
 * std::cout << "Kernel took " << timer.elapsed_ms() << " ms\n";
 */
class CudaTimer {
private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    bool started_;
    bool stopped_;

public:
    CudaTimer() : started_(false), stopped_(false) {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    // Disable copy
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    /**
     * @brief Start timing
     */
    void start() {
        CUDA_CHECK(cudaEventRecord(start_, 0));
        started_ = true;
        stopped_ = false;
    }

    /**
     * @brief Stop timing
     */
    void stop() {
        if (started_) {
            CUDA_CHECK(cudaEventRecord(stop_, 0));
            stopped_ = true;
        }
    }

    /**
     * @brief Get elapsed time in milliseconds
     * @return Elapsed time in ms
     */
    float elapsed_ms() const {
        if (!started_ || !stopped_) {
            return 0.0f;
        }
        float ms = 0.0f;
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Get CUDA device information as a string
 * @return Device name and compute capability
 */
inline std::string get_cuda_device_info() {
    int device = 0;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    char buf[256];
    snprintf(buf, sizeof(buf), "%s (Compute %d.%d, %zu MB global memory)",
             prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024));
    return std::string(buf);
}

/**
 * @brief Print CUDA memory info (free/total)
 */
inline void print_cuda_memory_info() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "CUDA Memory: " 
              << (free_mem / (1024 * 1024)) << " MB free / " 
              << (total_mem / (1024 * 1024)) << " MB total\n";
}
