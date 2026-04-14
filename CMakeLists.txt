cmake_minimum_required(VERSION 3.16)

project(NNS LANGUAGES C CXX)

include(FetchContent)

# -----------------------------
# Dependencies (Eigen)
# -----------------------------
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "" FORCE)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_BTL OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
    msft_proxy4
    GIT_TAG 4.0.0
    GIT_REPOSITORY https://github.com/ngcpp/proxy.git
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(msft_proxy4)

# -----------------------------
# Main target
# -----------------------------
add_executable(${PROJECT_NAME}
    main.cpp
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)

if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:/O2>)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>)
else()
    message(WARNING "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}, no extra flags set")
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Eigen3::Eigen
    msft_proxy4::proxy
)

# -----------------------------
# clang-format (опциональный таргет)
# -----------------------------
find_program(CLANG_FORMAT_EXE NAMES clang-format)

if (CLANG_FORMAT_EXE)
    file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/include/*.hpp
        ${CMAKE_SOURCE_DIR}/include/*.h
        ${CMAKE_SOURCE_DIR}/*.cpp
    )

    add_custom_target(format
        COMMAND "${CLANG_FORMAT_EXE}" -i --style=file ${PROJECT_SOURCES}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Running clang-format"
        VERBATIM
        COMMAND_EXPAND_LISTS
    )

    add_custom_target(format-check
        COMMAND "${CLANG_FORMAT_EXE}" --dry-run --Werror --style=file ${PROJECT_SOURCES}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Checking clang-format compliance"
        VERBATIM
        COMMAND_EXPAND_LISTS
    )
else()
    message(STATUS "clang-format not found, format targets unavailable")
endif()