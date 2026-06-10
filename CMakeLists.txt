cmake_minimum_required(VERSION 3.28)
project(KairoGPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(KairoGPU)
target_sources(KairoGPU
    PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
        FILES src/KairoGPU.cppm
)

add_executable(KairoGPUSmoke tests/smoke.cpp)
target_link_libraries(KairoGPUSmoke PRIVATE KairoGPU)
