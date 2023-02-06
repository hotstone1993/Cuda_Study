cmake_minimum_required(VERSION 3.13)
project(CUDA_STUDY LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/0.basic/00_MatMul.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/0.basic/00_MatMul_MT.cpp
)
                
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 20
    CUDA_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 75
)

target_include_directories( ${PROJECT_NAME} PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include 
)

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
