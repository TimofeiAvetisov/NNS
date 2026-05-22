cmake_minimum_required(VERSION 3.16)

project(NNS LANGUAGES CXX)

include(FetchContent)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json")
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Skip updates for fetched dependencies")

option(NNS_BUILD_EXAMPLE "Build the MNIST training example" OFF)
option(NNS_BUILD_TESTING "Build NNS tests" ON)
option(NNS_ENABLE_FORMAT_TARGETS "Enable clang-format helper targets" ON)

# Dependencies
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_BTL OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable tests in fetched dependencies" FORCE)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR nns-fetchcontent-no-cmakelists
)

FetchContent_MakeAvailable(eigen)

if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}"
    )
endif()

FetchContent_Declare(
    msft_proxy4
    GIT_REPOSITORY https://github.com/ngcpp/proxy.git
    GIT_TAG 4.0.0
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(msft_proxy4)

# Library target
add_library(nns INTERFACE)
add_library(NNS::nns ALIAS nns)

target_compile_features(nns INTERFACE cxx_std_23)
target_include_directories(nns INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(nns INTERFACE
    Eigen3::Eigen
    msft_proxy4::proxy
)

function(nns_enable_project_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4)
        target_compile_options(${target_name} PRIVATE $<$<CONFIG:Release>:/O2>)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(${target_name} PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>)
    else()
        message(WARNING "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}, no extra warnings set")
    endif()
endfunction()

if(NNS_BUILD_EXAMPLE)
    add_executable(${PROJECT_NAME} main.cpp)
    target_link_libraries(${PROJECT_NAME} PRIVATE NNS::nns)
    nns_enable_project_warnings(${PROJECT_NAME})
endif()

# Tests
set(BUILD_TESTING ${NNS_BUILD_TESTING} CACHE BOOL "Build NNS tests" FORCE)
include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Formatting
if(NNS_ENABLE_FORMAT_TARGETS)
    find_program(CLANG_FORMAT_EXE NAMES clang-format)
    if(CLANG_FORMAT_EXE)
        file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hpp"
        )

        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/format_sources.txt" "")
        foreach(src IN LISTS PROJECT_SOURCES)
            file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/format_sources.txt" "${src}\n")
        endforeach()

        add_custom_target(format
            COMMAND ${CMAKE_COMMAND}
                -DCLANG_FORMAT=${CLANG_FORMAT_EXE}
                -DSOURCES_FILE=${CMAKE_CURRENT_BINARY_DIR}/format_sources.txt
                -DMODE=format
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunClangFormat.cmake
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMENT "Running clang-format"
            VERBATIM
        )

        add_custom_target(format-check
            COMMAND ${CMAKE_COMMAND}
                -DCLANG_FORMAT=${CLANG_FORMAT_EXE}
                -DSOURCES_FILE=${CMAKE_CURRENT_BINARY_DIR}/format_sources.txt
                -DMODE=check
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunClangFormat.cmake
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMENT "Checking clang-format compliance"
            VERBATIM
        )
    else()
        message(STATUS "clang-format not found, format targets unavailable")
    endif()
endif()
