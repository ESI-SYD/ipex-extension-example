cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
#XeTLA need 20
set(CMAKE_CXX_STANDARD 20)

project(TritonBenchmark)

find_package(Python COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_package(IPEX REQUIRED)

# add the XeTLA library.
#add_subdirectory(third_party/xetla)
set(XETLA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xetla/include)

# add the XeTLA kernel.
add_subdirectory(xetla_kernel/softmax)