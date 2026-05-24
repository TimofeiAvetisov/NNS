#include <cmath>
#include <stdexcept>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <nns/NNS.hpp>

using Catch::Approx;

TEST_CASE("MSE and MAE losses compute expected values and gradients", "[loss]") {
    nns::Matrix y_hat(2, 2);
    y_hat << 1.0, 2.0, 3.0, 4.0;

    nns::Matrix y(2, 2);
    y << 1.0, 1.0, 3.0, 5.0;

    nns::MSELoss mse;
    REQUIRE(mse.loss(y_hat, y) == Approx(0.5));
    REQUIRE(mse.gradient(y_hat, y)(0, 1) == Approx(0.5));
    REQUIRE(mse.gradient(y_hat, y)(1, 1) == Approx(-0.5));

    nns::MAELoss mae;
    REQUIRE(mae.loss(y_hat, y) == Approx(0.5));
    REQUIRE(mae.gradient(y_hat, y)(0, 1) == Approx(0.25));
    REQUIRE(mae.gradient(y_hat, y)(1, 1) == Approx(-0.25));
}

TEST_CASE("Huber loss switches between quadratic and linear regions", "[loss]") {
    nns::HuberLoss huber;
    huber.delta = nns::Scalar{1.0};

    nns::Matrix y_hat(1, 2);
    y_hat << 0.5, 3.0;

    nns::Matrix y = nns::Matrix::Zero(1, 2);

    REQUIRE(huber.loss(y_hat, y) == Approx((0.125 + 2.5) / 2.0));
    REQUIRE(huber.gradient(y_hat, y)(0, 0) == Approx(0.25));
    REQUIRE(huber.gradient(y_hat, y)(0, 1) == Approx(0.5));

    huber.delta = nns::Scalar{0.0};
    REQUIRE_THROWS_AS(huber.loss(y_hat, y), std::invalid_argument);
}

TEST_CASE("Binary cross entropy losses compute finite values and gradients", "[loss]") {
    nns::Matrix probs(1, 2);
    probs << 0.8, 0.2;

    nns::Matrix labels(1, 2);
    labels << 1.0, 0.0;

    nns::BCELoss bce;
    REQUIRE(bce.loss(probs, labels) == Approx(-std::log(0.8)));
    REQUIRE(bce.gradient(probs, labels)(0, 0) == Approx(-0.625));
    REQUIRE(bce.gradient(probs, labels)(0, 1) == Approx(0.625));

    bce.eps = nns::Scalar{0.5};
    REQUIRE_THROWS_AS(bce.loss(probs, labels), std::invalid_argument);

    nns::BCEWithLogitsLoss logits_loss;
    nns::Matrix logits = nns::Matrix::Zero(1, 2);
    REQUIRE(logits_loss.loss(logits, labels) == Approx(std::log(2.0)));
    REQUIRE(logits_loss.gradient(logits, labels)(0, 0) == Approx(-0.25));
    REQUIRE(logits_loss.gradient(logits, labels)(0, 1) == Approx(0.25));
}

TEST_CASE("Cross entropy applies stable softmax per column", "[loss]") {
    nns::Matrix logits(3, 2);
    logits << 2.0, 1.0, 1.0, 0.0, 0.0, 3.0;

    nns::Matrix labels = nns::Matrix::Zero(3, 2);
    labels(0, 0) = 1.0;
    labels(2, 1) = 1.0;

    const nns::Matrix probs = nns::CrossEntropyLoss::softmax(logits);
    REQUIRE(probs.col(0).sum() == Approx(1.0));
    REQUIRE(probs.col(1).sum() == Approx(1.0));

    nns::CrossEntropyLoss loss;
    REQUIRE(loss.loss(logits, labels) > 0.0);
    REQUIRE(loss.gradient(logits, labels).rows() == logits.rows());
    REQUIRE(loss.gradient(logits, labels).cols() == logits.cols());
    REQUIRE(loss.gradient(logits, labels).col(0).sum() == Approx(0.0).margin(1e-12));
}

TEST_CASE("All built-in losses reject invalid shapes and empty inputs", "[loss]") {
    nns::Matrix a = nns::Matrix::Ones(2, 2);
    nns::Matrix b = nns::Matrix::Ones(1, 2);

    REQUIRE_THROWS_AS(nns::MSELoss{}.gradient(a, b), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::MAELoss{}.gradient(a, b), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::HuberLoss{}.gradient(a, b), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::BCELoss{}.gradient(a, b), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::BCEWithLogitsLoss{}.gradient(a, b), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::CrossEntropyLoss{}.gradient(a, b), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::CrossEntropyLoss::softmax(nns::Matrix{0, 0}), std::invalid_argument);
}

TEST_CASE("AnyLossFunction dispatches custom stateful loss", "[loss][proxy]") {
    struct CountingLoss {
        nns::Scalar loss(const nns::Matrix&, const nns::Matrix&) {
            ++calls;
            return nns::Scalar{3.0};
        }

        nns::Matrix gradient(const nns::Matrix& y_hat, const nns::Matrix&) {
            ++calls;
            return nns::Matrix::Zero(y_hat.rows(), y_hat.cols());
        }

        int calls = 0;
    };

    nns::AnyLossFunction loss = nns::make_AnyLossFunction<CountingLoss>();
    nns::Matrix value = nns::Matrix::Zero(1, 1);

    REQUIRE(loss->loss(value, value) == Approx(3.0));
    REQUIRE(loss->gradient(value, value).rows() == 1);
}
