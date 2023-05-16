cmake_minimum_required(VERSION 3.13)

project(MatMul LANGUAGES CXX CUDA)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/base.cmake)

add_executable(
  ${PROJECT_NAME}
  ${BASE_SRC_PATH_LIST}
  ${CMAKE_CURRENT_SOURCE_DIR}/0_0_MatMul.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/0_0_MatMul_MT.cpp)
  
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES CXX_STANDARD 20
             CUDA_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_ARCHITECTURES 75
             CUDA_SEPARABLE_COMPILATION ON
             CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(
  ${PROJECT_NAME} PUBLIC ${BASE_HEADER_PATH_LIST}
                            ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} ${CUDART_LIBRARY})
