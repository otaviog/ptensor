UNIT_TEST := "build/coverage/native/tests/unit_tests_all"

_run_coverage:
    cmake --workflow --preset coverage-build
    - {{ UNIT_TEST }}
    llvm-profdata merge -sparse default.profraw -o edge-ai-coverage.profdata
    rm default.profraw

coverage: _run_coverage
    llvm-cov export -object {{ UNIT_TEST }} -instr-profile=edge-ai-coverage.profdata -format=lcov > lcov.info
    rm edge-ai-coverage.profdata

coverage-html: _run_coverage
    llvm-cov show --ignore-filename-regex=.*/build/.* -format=html -output-dir=build/coverage/coverage-html -instr-profile=edge-ai-coverage.profdata --object {{ UNIT_TEST }}
    rm edge-ai-coverage.profdata

clang-format:
    find native -type f \( -name "*.cpp" -o -name "*.hpp" \) -exec clang-format -i {} +
