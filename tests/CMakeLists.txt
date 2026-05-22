FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(catch2)

add_executable(nns_tests
    nns_smoke_tests.cpp
)

target_link_libraries(nns_tests PRIVATE
    NNS::nns
    Catch2::Catch2WithMain
)
nns_enable_project_warnings(nns_tests)

include(Catch)
catch_discover_tests(nns_tests)
