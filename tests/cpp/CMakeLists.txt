cmake_minimum_required(VERSION 3.16)

# ---------------------------------------------------------------------------
# Fetch Catch2 v3 using CMake's FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.4
)
FetchContent_MakeAvailable(Catch2)

# ---------------------------------------------------------------------------
# Test executable
# ---------------------------------------------------------------------------
add_executable(test_ucminf test_ucminf.cpp)

target_link_libraries(test_ucminf
    PRIVATE
        ucminf_core
        Catch2::Catch2WithMain
)

# ---------------------------------------------------------------------------
# Discover and register tests with CTest
# ---------------------------------------------------------------------------
include(CTest)
include(Catch)
catch_discover_tests(test_ucminf)
