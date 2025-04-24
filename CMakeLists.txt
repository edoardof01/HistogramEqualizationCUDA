cmake_minimum_required(VERSION 3.18)
project(HistogramEqualizationCUDA LANGUAGES CXX CUDA)

# Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

# Architettura (RTX 4080)
set(CMAKE_CUDA_ARCHITECTURES 89)

enable_language(CUDA)

# Rimuovi il runtime statico
# set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# File .cu
set(SOURCES
        main.cu
        histogram_gpu.cu
        histogram_cpu.cpp
        stb_image_wrapper.cpp
        histogram_gpu.hpp
        histogram_cpu.hpp
)

# Executable
add_executable(HistogramEqualizationCUDA ${SOURCES})

target_include_directories(HistogramEqualizationCUDA PRIVATE ${CMAKE_SOURCE_DIR})

# Abilita separazione device code
set_target_properties(HistogramEqualizationCUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

target_compile_options(HistogramEqualizationCUDA PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--relocatable-device-code=true -Xcompiler -fPIC>
)

# Link CUDA
find_package(CUDAToolkit REQUIRED)
target_link_libraries(HistogramEqualizationCUDA PRIVATE CUDA::cudart CUDA::cuda_driver)

target_include_directories(HistogramEqualizationCUDA PRIVATE ${CMAKE_SOURCE_DIR}/include)