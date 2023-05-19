cmake_minimum_required(VERSION 3.13)

project(LockFreeQueueTest LANGUAGES CXX)

find_package(Catch2 3 REQUIRED)

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/UnitTest.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

target_link_libraries(${PROJECT_NAME} PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(${PROJECT_NAME})