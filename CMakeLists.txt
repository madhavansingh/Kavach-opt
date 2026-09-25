cmake_minimum_required(VERSION 3.20)

project(kavach_opt
    VERSION 0.1.0
    DESCRIPTION "Sovereign Indigenous GPU-Accelerated Optimization Solver Core"
    LANGUAGES CXX
)

# Standard C++20 Enforcement
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for clang-tidy, LSP, and tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build Options
option(KAVACH_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(KAVACH_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(KAVACH_BUILD_TESTS "Build CTest test suite" ON)
option(KAVACH_BUILD_CLI "Build kavach-opt CLI executable" ON)
option(KAVACH_ENABLE_CUDA "Enable CUDA backend support (Phase 11)" OFF)

# Determine Git Commit Hash
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
else()
    set(GIT_COMMIT_HASH "unknown")
endif()
if(NOT GIT_COMMIT_HASH)
    set(GIT_COMMIT_HASH "0000000")
endif()

# Compiler Warning Flags
add_library(kavach_compiler_flags INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    target_compile_options(kavach_compiler_flags INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
    )
    if(KAVACH_ENABLE_WARNINGS_AS_ERRORS)
        target_compile_options(kavach_compiler_flags INTERFACE -Werror)
    endif()
elseif(MSVC)
    target_compile_options(kavach_compiler_flags INTERFACE
        /W4
        /permissive-
    )
    if(KAVACH_ENABLE_WARNINGS_AS_ERRORS)
        target_compile_options(kavach_compiler_flags INTERFACE /WX)
    endif()
endif()

# Sanitizer Configuration
if(KAVACH_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
        message(STATUS "Configuring AddressSanitizer and UndefinedBehaviorSanitizer")
        target_compile_options(kavach_compiler_flags INTERFACE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
            -g
        )
        target_link_options(kavach_compiler_flags INTERFACE
            -fsanitize=address,undefined
        )
    else()
        message(WARNING "Sanitizers requested but compiler ${CMAKE_CXX_COMPILER_ID} not configured for sanitizers")
    endif()
endif()

# Generated Version Header
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/core/version.hpp"
    @ONLY
)

# Core Library Target
add_library(kavach_core
    src/core/logger.cpp
    src/core/random.cpp
    src/model/model.cpp
    src/io/parser.cpp
    src/io/mps_parser.cpp
    src/io/lp_parser.cpp
    src/numerics/sparse_matrix.cpp
    src/numerics/model_converter.cpp
    src/numerics/lu_factorization.cpp
    src/numerics/scaling.cpp
    src/presolve/presolve.cpp
    src/algorithms/lp/standard_form.cpp
    src/algorithms/lp/basis.cpp
    src/algorithms/lp/revised_simplex.cpp
    src/algorithms/lp/simplex.cpp
)

target_link_libraries(kavach_core PUBLIC kavach_compiler_flags)

target_include_directories(kavach_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# CLI Binary Target
if(KAVACH_BUILD_CLI)
    add_executable(kavach-opt src/cli/main.cpp)
    target_link_libraries(kavach-opt PRIVATE kavach_core)
endif()

# Testing Target
if(KAVACH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
