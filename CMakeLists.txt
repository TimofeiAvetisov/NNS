cmake_minimum_required(VERSION 3.16)

project(NNS LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

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

# -----------------------------
# Main target
# -----------------------------
add_executable(${PROJECT_NAME}
  main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/utils
)

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)

# -----------------------------
# clang-format only
# -----------------------------

# Путь к clang-format (Windows)
if(WIN32)
  set(LLVM_BIN "C:/Program Files/LLVM/bin")
  set(CLANG_FORMAT_EXE "${LLVM_BIN}/clang-format.exe")
else()
  find_program(CLANG_FORMAT_EXE NAMES clang-format)
endif()

# Собираем только ваш код
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/*.cpp
  ${CMAKE_SOURCE_DIR}/*.hpp
  ${CMAKE_SOURCE_DIR}/*.h
)

# Исключаем build и сторонние директории
list(FILTER PROJECT_SOURCES EXCLUDE REGEX ".*/build/.*")
list(FILTER PROJECT_SOURCES EXCLUDE REGEX ".*/build%-ninja/.*")
list(FILTER PROJECT_SOURCES EXCLUDE REGEX ".*/bin/.*")
list(FILTER PROJECT_SOURCES EXCLUDE REGEX ".*/_deps/.*")
list(FILTER PROJECT_SOURCES EXCLUDE REGEX ".*/cmake%-build%-.*")

if(EXISTS "${CLANG_FORMAT_EXE}")
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
endif()
