#include <cmath>
#include <stdexcept>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <nns/NNS.hpp>

using Catch::Approx;

TEST_CASE("ConstantLR and TimeDecayLR expose iteration state", "[lr]") {
    nns::ConstantLR constant(nns::LR{0.1});
    REQUIRE(constant.get_lr() == Approx(0.1));
    REQUIRE(constant.get_iter() == 1);
    constant.iter_step();
    REQUIRE(constant.get_iter() == 2);

    nns::TimeDecayLR decay(nns::LR{1.0}, nns::S0{1.0}, nns::P{0.5});
    REQUIRE(decay.get_lr() == Approx(1.0 / std::sqrt(2.0)));
    decay.iter_step();
    REQUIRE(decay.get_iter() == 2);

    REQUIRE_THROWS_AS(nns::ConstantLR(nns::LR{0.0}), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::TimeDecayLR(nns::LR{0.0}, nns::S0{1.0}, nns::P{0.5}), std::invalid_argument);
}

TEST_CASE("AnyLearningRateScheduler dispatches custom scheduler", "[lr][proxy]") {
    struct StepLR {
        nns::Scalar get_lr() {
            return iter < 2 ? nns::Scalar{0.5} : nns::Scalar{0.25};
        }

        void iter_step() {
            ++iter;
        }

        size_t get_iter() const {
            return iter;
        }

        size_t iter = 1;
    };

    nns::AnyLearningRateScheduler scheduler = nns::make_AnyLearningRateScheduler(StepLR{});
    REQUIRE(scheduler->get_lr() == Approx(0.5));
    scheduler->iter_step();
    REQUIRE(scheduler->get_lr() == Approx(0.25));
}

TEST_CASE("SGD updates matrix and vector parameters", "[optimizer]") {
    nns::SGDOptimizer optimizer(nns::ConstantLR(nns::LR{0.1}));

    nns::Matrix weights(1, 2);
    weights << 1.0, -1.0;
    nns::Matrix gradients(1, 2);
    gradients << 0.5, -0.5;
    optimizer.update_weights(weights, gradients, {});
    REQUIRE(weights(0, 0) == Approx(0.95));
    REQUIRE(weights(0, 1) == Approx(-0.95));

    nns::Vector bias(2);
    bias << 1.0, -1.0;
    nns::Vector bias_grad(2);
    bias_grad << 0.5, -0.5;
    optimizer.update_weights(bias, bias_grad, {});
    REQUIRE(bias(0) == Approx(0.95));
    REQUIRE(bias(1) == Approx(-0.95));

    REQUIRE_THROWS_AS(optimizer.update_weights(weights, nns::Matrix::Zero(2, 2), {}),
                      std::invalid_argument);
}

TEST_CASE("Adam updates parameters and validates hyperparameters", "[optimizer]") {
    nns::Matrix weights = nns::Matrix::Ones(1, 1);
    nns::Matrix gradients = nns::Matrix::Ones(1, 1);

    nns::AdamOptimizer optimizer(nns::ConstantLR(nns::LR{0.01}));
    std::any cache = optimizer.update_weights(weights, gradients, {});
    REQUIRE(cache.has_value());
    REQUIRE(weights(0, 0) < 1.0);

    nns::Vector bias = nns::Vector::Ones(1);
    nns::Vector bias_grad = nns::Vector::Ones(1);
    cache = optimizer.update_weights(bias, bias_grad, {});
    REQUIRE(cache.has_value());
    REQUIRE(bias(0) < 1.0);

    nns::AdamOptimizer bad_beta1(nns::ConstantLR(nns::LR{0.01}), nns::Beta1{1.0});
    REQUIRE_THROWS_AS(bad_beta1.update_weights(weights, gradients, {}), std::invalid_argument);

    nns::AdamOptimizer bad_beta2(nns::ConstantLR(nns::LR{0.01}), nns::Beta1{0.9}, nns::Beta2{1.0});
    REQUIRE_THROWS_AS(bad_beta2.update_weights(weights, gradients, {}), std::invalid_argument);

    nns::AdamOptimizer bad_eps(nns::ConstantLR(nns::LR{0.01}), nns::Beta1{0.9}, nns::Beta2{0.999},
                               nns::Eps{0.0});
    REQUIRE_THROWS_AS(bad_eps.update_weights(weights, gradients, {}), std::invalid_argument);
}

TEST_CASE("AnyOptimizer dispatches custom optimizer", "[optimizer][proxy]") {
    struct ScaleOptimizer {
        std::any update_weights(nns::Matrix& param, const nns::Matrix& grad, std::any&& cache) {
            param -= nns::Scalar{2.0} * grad;
            return std::move(cache);
        }

        std::any update_weights(nns::Vector& param, const nns::Vector& grad, std::any&& cache) {
            param -= nns::Scalar{2.0} * grad;
            return std::move(cache);
        }

        void iter_step() {
            ++steps;
        }

        int steps = 0;
    };

    nns::AnyOptimizer optimizer = nns::make_AnyOptimizer(ScaleOptimizer{});
    nns::Matrix weights = nns::Matrix::Ones(1, 1);
    nns::Matrix gradients = nns::Matrix::Constant(1, 1, 0.25);

    optimizer->update_weights(weights, gradients, {});
    optimizer->iter_step();
    REQUIRE(weights(0, 0) == Approx(0.5));
}
