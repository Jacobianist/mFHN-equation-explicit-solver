cmake_minimum_required(VERSION 4.1)
project(mFHN_Simulation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)


find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

add_executable(mfhn_solver
    src/main.cu
    src/config.cpp
    src/solver_explicit.cu
)

target_include_directories(mfhn_solver PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(mfhn_solver
    CUDA::cudart
    nlohmann_json::nlohmann_json
    HDF5::HDF5
)

#set_target_properties(fhn_solver PROPERTIES CUDA_ARCHITECTURES "75;80;86")
