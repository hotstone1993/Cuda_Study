set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -src-in-ptx")

find_package(CUDA REQUIRED)

if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

set(BASE_SRC_PATH_LIST ${CMAKE_CURRENT_LIST_DIR}/../src/main.cpp
                        ${CMAKE_CURRENT_LIST_DIR}/../src/include/utils.cu)

set(BASE_HEADER_PATH_LIST ${CMAKE_CURRENT_LIST_DIR}/../src
                            ${CMAKE_CURRENT_LIST_DIR}/../src/include
                            ${CMAKE_CURRENT_LIST_DIR}/../thirdparty/stb
                            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})