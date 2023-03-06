cmake_minimum_required(VERSION 3.13)
project(CUDA_STUDY LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include/utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/0.basic/0_0_MatMul.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/0.basic/0_0_MatMul_MT.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/0.basic/0_1_MergeSort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/0.basic/0_1_MergeSort_MT.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/1.image/1_0_Blur.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/1.image/1_0_Blur_MT.cpp
)
                
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 20
    CUDA_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories( ${PROJECT_NAME} PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stb
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

if(MSVC)
    find_package(CUDA)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
else()
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

    target_include_directories(
        ${PROJECT_NAME} 
        PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

    target_link_libraries(
        ${PROJECT_NAME} 
        ${CUDART_LIBRARY}
    )
endif()
