add_executable(kavach_unit_tests
    main.cpp
    unit/test_phase0_foundation.cpp
    unit/test_phase1_canonical_model.cpp
    unit/test_phase2_parser.cpp
    unit/test_phase3_numerics.cpp
    unit/test_presolve.cpp
    unit/test_phase5_lp.cpp
    unit/test_phase6_verification.cpp
    unit/test_phase7_milp.cpp
    unit/test_phase8_dual_simplex.cpp
    unit/test_phase9_qp.cpp
    unit/test_phase10_cpu_parallel.cpp
    unit/test_phase11_cuda.cpp
    unit/test_phase12_benchmark.cpp
    unit/test_phase13_mrpl.cpp
    unit/test_phase14_strategy.cpp
    unit/test_cutting_planes.cpp
    unit/test_advanced_search.cpp
)

target_link_libraries(kavach_unit_tests PRIVATE kavach_core)
target_include_directories(kavach_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME kavach_unit_tests COMMAND kavach_unit_tests)

# Add dependency graph integrity check as a CTest test
find_package(Python3 COMPONENTS Interpreter QUIET)
if(NOT Python3_FOUND AND EXISTS "/usr/bin/python3")
    set(Python3_EXECUTABLE "/usr/bin/python3")
    set(Python3_FOUND TRUE)
endif()

if(Python3_FOUND)
    add_test(
        NAME dependency_sovereignty_audit
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/check_dependencies.py
    )
endif()
