#include <type_traits>

#include <catch2/catch_test_macros.hpp>

#include <nns/NNS.hpp>

TEST_CASE("Core aliases expose scalar matrix and vector types", "[core]") {
    STATIC_REQUIRE(std::is_same_v<nns::Scalar, double>);

    nns::Matrix matrix = nns::Matrix::Zero(2, 3);
    nns::Vector vector = nns::Vector::Zero(2);

    REQUIRE(matrix.rows() == 2);
    REQUIRE(matrix.cols() == 3);
    REQUIRE(vector.rows() == 2);
    REQUIRE(vector.cols() == 1);
}

TEST_CASE("StrongType stores values and converts to the underlying type", "[core]") {
    struct Tag {};

    nns::StrongType<Tag> value{1.25};
    nns::StrongType<Tag, int> integer{7};

    REQUIRE(static_cast<nns::Scalar>(value) == 1.25);
    REQUIRE(static_cast<int>(integer) == 7);
}
