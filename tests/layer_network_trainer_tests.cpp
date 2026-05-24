#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <nns/NNS.hpp>

using Catch::Approx;

TEST_CASE("LinearLayer validates dimensions and produces expected shapes", "[layers]") {
    nns::RandomGenerator rng(nns::Seed{99});

    REQUIRE_THROWS_AS(nns::LinearLayer(nns::In{0}, nns::Out{1}, rng), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::LinearLayer(nns::In{1}, nns::Out{0}, rng), std::invalid_argument);

    nns::LinearLayer layer(nns::In{2}, nns::Out{1}, rng);
    REQUIRE_THROWS_AS(layer.predict(nns::Matrix::Zero(3, 1)), std::invalid_argument);

    auto [Y, cache] = layer.forward(nns::Matrix::Ones(2, 3));
    REQUIRE(Y.rows() == 1);
    REQUIRE(Y.cols() == 3);

    auto [dX, grads] = layer.backward(nns::Matrix::Ones(1, 3), cache);
    REQUIRE(dX.rows() == 2);
    REQUIRE(dX.cols() == 3);
    REQUIRE(grads.has_value());

    REQUIRE_THROWS_AS(layer.backward(nns::Matrix::Ones(2, 3), cache), std::invalid_argument);
    REQUIRE_THROWS_AS(layer.backward(nns::Matrix::Ones(1, 3), std::any{}), std::invalid_argument);
}

TEST_CASE("LinearLayer update validates gradient shapes", "[layers][optimizer]") {
    nns::RandomGenerator rng(nns::Seed{1});
    nns::LinearLayer layer(nns::In{2}, nns::Out{1}, rng);
    nns::AnyOptimizer optimizer =
        nns::make_AnyOptimizer(nns::SGDOptimizer(nns::ConstantLR(nns::LR{0.01})));

    std::any bad_weight_grad =
        std::pair<nns::Matrix, nns::Vector>{nns::Matrix::Zero(2, 2), nns::Vector::Zero(1)};
    std::any bad_bias_grad =
        std::pair<nns::Matrix, nns::Vector>{nns::Matrix::Zero(1, 2), nns::Vector::Zero(2)};

    REQUIRE_THROWS_AS(layer.update(std::move(bad_weight_grad), optimizer, {}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(layer.update(std::move(bad_bias_grad), optimizer, {}), std::invalid_argument);
}

TEST_CASE("make_AnyLayer wraps linear and activation layers", "[layers][proxy]") {
    nns::RandomGenerator rng(nns::Seed{2});

    nns::AnyLayer linear = nns::make_AnyLayer(nns::LinearLayer(nns::In{1}, nns::Out{1}, rng));
    REQUIRE(linear->predict(nns::Matrix::Ones(1, 1)).rows() == 1);

    nns::AnyLayer activation = nns::make_AnyLayer(nns::ReLU{});
    nns::Matrix X(1, 2);
    X << -1.0, 2.0;
    nns::Matrix Y = activation->predict(X);
    REQUIRE(Y(0, 0) == Approx(0.0));
    REQUIRE(Y(0, 1) == Approx(2.0));
}

TEST_CASE("NeuralNetwork supports predict, forward, backward, and update", "[network]") {
    nns::RandomGenerator rng(nns::Seed{42});

    nns::NeuralNetwork net(
        nns::LinearLayer(nns::In{2}, nns::Out{3}, rng, nns::Distribution::Normal, nns::Gain{0.01}),
        nns::ReLU(),
        nns::LinearLayer(nns::In{3}, nns::Out{1}, rng, nns::Distribution::Normal, nns::Gain{0.01}));

    nns::Matrix X(2, 4);
    X << 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0;

    const nns::Matrix prediction = net.predict(X);
    REQUIRE(prediction.rows() == 1);
    REQUIRE(prediction.cols() == 4);

    auto [forward_output, cache] = net.forward(nns::Matrix{X});
    REQUIRE(forward_output.rows() == 1);
    REQUIRE(forward_output.cols() == 4);
    REQUIRE(cache.has_value());

    auto [dX, gradients] = net.backward(nns::Matrix::Ones(1, 4), cache);
    REQUIRE(dX.rows() == 2);
    REQUIRE(dX.cols() == 4);
    REQUIRE(gradients.has_value());

    nns::AnyOptimizer optimizer =
        nns::make_AnyOptimizer(nns::SGDOptimizer(nns::ConstantLR(nns::LR{0.01})));
    std::any opt_cache = net.update(std::move(gradients), optimizer, {});
    REQUIRE(opt_cache.has_value());

    REQUIRE_THROWS_AS(net.predict(nns::Matrix::Zero(3, 1)), std::invalid_argument);
    REQUIRE_THROWS_AS(net.backward(nns::Matrix::Ones(1, 4), std::any{}), std::invalid_argument);
}

TEST_CASE("NeuralNetwork rejects invalid cache and gradient counts", "[network]") {
    nns::RandomGenerator rng(nns::Seed{3});
    nns::NeuralNetwork net(nns::LinearLayer(nns::In{1}, nns::Out{1}, rng));

    std::vector<std::any> wrong_cache_count{std::any{}, std::any{}};
    REQUIRE_THROWS_AS(net.backward(nns::Matrix::Ones(1, 1), std::any{wrong_cache_count}),
                      std::invalid_argument);

    std::vector<std::any> wrong_grad_count{std::any{}, std::any{}};
    nns::AnyOptimizer optimizer =
        nns::make_AnyOptimizer(nns::SGDOptimizer(nns::ConstantLR(nns::LR{0.01})));
    REQUIRE_THROWS_AS(net.update(std::any{std::move(wrong_grad_count)}, optimizer, {}),
                      std::invalid_argument);
}

TEST_CASE("NeuralNetwork rejects operations without layers", "[network]") {
    nns::NeuralNetwork net;

    REQUIRE_THROWS_AS(net.predict(nns::Matrix::Ones(1, 1)), std::logic_error);
    REQUIRE_THROWS_AS(net.forward(nns::Matrix::Ones(1, 1)), std::logic_error);
}

TEST_CASE("Trainer exposes fit_epoch, fit history, RNG path, and optimizer reset", "[trainer]") {
    nns::RandomGenerator rng(nns::Seed{123});

    nns::Matrix X(1, 4);
    X << 0.0, 1.0, 2.0, 3.0;

    nns::Matrix Y(1, 4);
    Y << 0.0, 1.0, 2.0, 3.0;

    nns::DataLoader loader(X, Y, nns::BatchSize{2}, nns::Shuffle{false});
    nns::NeuralNetwork net(
        nns::LinearLayer(nns::In{1}, nns::Out{2}, rng, nns::Distribution::Normal, nns::Gain{0.01}),
        nns::Tanh(),
        nns::LinearLayer(nns::In{2}, nns::Out{1}, rng, nns::Distribution::Normal, nns::Gain{0.01}));

    nns::AnyOptimizer optimizer =
        nns::make_AnyOptimizer(nns::SGDOptimizer(nns::ConstantLR(nns::LR{0.01})));
    nns::AnyLossFunction loss = nns::make_AnyLossFunction<nns::MSELoss>();

    nns::Trainer trainer(net, optimizer, loss, loader);

    REQUIRE(std::isfinite(trainer.fit_epoch()));

    const std::vector<nns::Scalar> history = trainer.fit(2);
    REQUIRE(history.size() == 2);
    REQUIRE(std::isfinite(history[0]));
    REQUIRE(std::isfinite(history[1]));

    trainer.reset_optimizer_state();
    REQUIRE(std::isfinite(trainer.fit_epoch()));
}

TEST_CASE("Trainer supports shuffled training and rejects loaders without full batches",
          "[trainer]") {
    nns::RandomGenerator rng(nns::Seed{321});

    nns::Matrix X = nns::Matrix::Ones(1, 4);
    nns::Matrix Y = nns::Matrix::Ones(1, 4);
    nns::DataLoader shuffled_loader(X, Y, nns::BatchSize{2}, nns::Shuffle{true});
    nns::NeuralNetwork net(nns::LinearLayer(nns::In{1}, nns::Out{1}, rng));
    nns::AnyOptimizer optimizer =
        nns::make_AnyOptimizer(nns::SGDOptimizer(nns::ConstantLR(nns::LR{0.01})));
    nns::AnyLossFunction loss = nns::make_AnyLossFunction<nns::MSELoss>();
    nns::Trainer trainer(net, optimizer, loss, shuffled_loader);
    REQUIRE(std::isfinite(trainer.fit_epoch(rng)));
    REQUIRE(trainer.fit(2, rng).size() == 2);
    REQUIRE_THROWS_AS(trainer.fit_epoch(), std::logic_error);

    nns::DataLoader no_full_batch(nns::Matrix::Ones(1, 1), nns::Matrix::Ones(1, 1),
                                  nns::BatchSize{2}, nns::Shuffle{false});
    nns::Trainer no_batch_trainer(net, optimizer, loss, no_full_batch);
    REQUIRE_THROWS_AS(no_batch_trainer.fit_epoch(), std::logic_error);
}
