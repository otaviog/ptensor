UNIT_TEST := "build/coverage/tests/unit_tests_all"

# On macOS, LLVM tools are accessed via xcrun. On Linux, use the versioned binaries.
LLVM_PROFDATA := if os() == "macos" { "xcrun llvm-profdata" } else { "llvm-profdata-20" }
LLVM_COV := if os() == "macos" { "xcrun llvm-cov" } else { "llvm-cov-20" }

_run_coverage:
    cmake --workflow --preset coverage-build
    - {{ UNIT_TEST }}
    {{ LLVM_PROFDATA }} merge -sparse default.profraw -o ptensor-coverage.profdata
    rm default.profraw

coverage: _run_coverage
    {{ LLVM_COV }} export -object {{ UNIT_TEST }} -instr-profile=ptensor-coverage.profdata -format=lcov > lcov.info
    rm ptensor-coverage.profdata

coverage-html: _run_coverage
    {{ LLVM_COV }} show --ignore-filename-regex=.*/build/.* -format=html -output-dir=build/coverage/coverage-html -instr-profile=ptensor-coverage.profdata --object {{ UNIT_TEST }}
    rm ptensor-coverage.profdata

clang-format:
    find src -type f \( -name "*.cpp" -o -name "*.hpp" \) ! -path "*/lib/*" -exec clang-format -i {} +

CLANG_TIDY_BUILD := "build/clang/debug"
CLANG_TIDY_EXTRA := if os() == "macos" { "--extra-arg=-isysroot" + ` xcrun --show-sdk-path` } else { "" }

# Lint only files that are in the active build (compile_commands.json).
# Skips modules disabled by the preset (infer/recog) and orphans.
_clang_tidy_files:
    @jq -r '.[].file' {{ CLANG_TIDY_BUILD }}/compile_commands.json | grep "/src/" | grep -v "/lib/" | grep -v "imgui_impl" | sort -u

clang-tidy:
    just _clang_tidy_files | xargs -n 1 -P "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" clang-tidy --quiet -p {{ CLANG_TIDY_BUILD }} {{ CLANG_TIDY_EXTRA }}

test:
    ctest --preset clang/debug

run-ci:
    act --job linux-check

DEVCONTAINER_IMAGE := "ptensor-devcontainer"

devcontainer-build:
    docker build -t {{ DEVCONTAINER_IMAGE }} -f .github/docker/Dockerfile .

devcontainer-shell: devcontainer-build
    docker run --rm -it -v "$(pwd)":/workspace -w /workspace {{ DEVCONTAINER_IMAGE }} bash

# Package the VS Code extension into build/ptensor-vscode.vsix.
vscode-package:
    mkdir -p build
    cd src/ptensor-vscode && npx @vscode/vsce package --no-dependencies --out "{{ justfile_directory() }}/build/ptensor-vscode.vsix"
