#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <nns/NNS.hpp>

using Catch::Approx;

TEST_CASE("Built-in scalar activations return expected values", "[activation]") {
    nns::ReLU relu;
    REQUIRE(relu.forward(-2.0) == Approx(0.0));
    REQUIRE(relu.forward(3.0) == Approx(3.0));
    REQUIRE(relu.derivative(-2.0) == Approx(0.0));
    REQUIRE(relu.derivative(3.0) == Approx(1.0));

    nns::Sigmoid sigmoid;
    REQUIRE(sigmoid.forward(0.0) == Approx(0.5));
    REQUIRE(sigmoid.derivative(0.0) == Approx(0.25));

    nns::Tanh tanh;
    REQUIRE(tanh.forward(0.0) == Approx(0.0));
    REQUIRE(tanh.derivative(0.0) == Approx(1.0));
}

TEST_CASE("AnyScalarActivation dispatches custom activation", "[activation][proxy]") {
    struct Square {
        nns::Scalar forward(nns::Scalar x) const {
            return x * x;
        }

        nns::Scalar derivative(nns::Scalar x) const {
            return nns::Scalar{2.0} * x;
        }
    };

    nns::AnyScalarActivation activation = nns::make_AnyScalarActivation(Square{});

    REQUIRE(activation->forward(3.0) == Approx(9.0));
    REQUIRE(activation->derivative(3.0) == Approx(6.0));
}

TEST_CASE("ActivationLayer applies activation and backward derivative elementwise", "[layers]") {
    nns::ActivationLayer layer(nns::ReLU{});

    nns::Matrix X(2, 2);
    X << -1.0, 2.0, 3.0, -4.0;

    auto [Y, cache] = layer.forward(nns::Matrix{X});
    REQUIRE(Y(0, 0) == Approx(0.0));
    REQUIRE(Y(0, 1) == Approx(2.0));
    REQUIRE(Y(1, 0) == Approx(3.0));
    REQUIRE(Y(1, 1) == Approx(0.0));

    auto [dX, grad] = layer.backward(nns::Matrix::Ones(2, 2), cache);
    REQUIRE(dX(0, 0) == Approx(0.0));
    REQUIRE(dX(0, 1) == Approx(1.0));
    REQUIRE(dX(1, 0) == Approx(1.0));
    REQUIRE(dX(1, 1) == Approx(0.0));
    REQUIRE_FALSE(grad.has_value());
}
