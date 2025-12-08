cmake_minimum_required(VERSION 3.18)
project(cpp_rl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set your LibTorch path
set(Torch_DIR "C:/random/RLwithcpp/libtorch-2.9.1+cu130/libtorch/share/cmake/Torch") # Windows
# set(Torch_DIR "~/libs/libtorch/share/cmake/Torch") # Linux

find_package(Torch REQUIRED)

add_executable(test_libtorch src/test_libtorch.cpp)
target_link_libraries(test_libtorch "${TORCH_LIBRARIES}")
set_property(TARGET test_libtorch PROPERTY CXX_STANDARD 20)
