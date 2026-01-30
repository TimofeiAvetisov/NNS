cmake_minimum_required(VERSION 3.10)

project(NNS)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${CMAKE_SOURCE_DIR}/Include)

file(GLOB_RECURSE SOURCES "Source/*.cpp" "Source/*.c" "Source/*.cc" "Source/*.cxx")

if(SOURCES)
    add_executable(${PROJECT_NAME} ${SOURCES})
else()
    message(STATUS "No source files found in Source/ directory")
    add_executable(${PROJECT_NAME} "")
endif()

