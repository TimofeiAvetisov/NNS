#include <cmath>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <nns/activation/BuiltinActivations.hpp>
#include <nns/core/Types.hpp>
#include <nns/layers/LinearLayers.hpp>
#include <nns/learningrates/BuiltinLearningRates.hpp>
#include <nns/loss/AnyLossFunction.hpp>
#include <nns/loss/BuiltinLoss.hpp>
#include <nns/network/Network.hpp>
#include <nns/optimizer/AnyOptimizer.hpp>
#include <nns/optimizer/BuiltinOptimizers.hpp>
#include <nns/trainer/Trainer.hpp>
#include <nns/utils/DataLoader.hpp>
#include <nns/utils/Random.hpp>

using Catch::Approx;

TEST_CASE("DataLoader splits matrices into full batches", "[utils][dataloader]") {
    nns::Matrix X(2, 4);
    X << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;

    nns::Matrix Y(1, 4);
    Y << 10.0, 20.0, 30.0, 40.0;

    nns::DataLoader loader(X, Y, nns::BatchSize{2}, nns::Shuffle{false});

    REQUIRE(loader.size() == 4);
    REQUIRE(loader.batch_size() == 2);
    REQUIRE(loader.num_batches() == 2);

    const auto first_batch = loader.get_batch(0);
    REQUIRE(first_batch.X.rows() == 2);
    REQUIRE(first_batch.X.cols() == 2);
    REQUIRE(first_batch.Y.rows() == 1);
    REQUIRE(first_batch.Y.cols() == 2);
    REQUIRE(first_batch.X(0, 0) == Approx(X(0, 0)));
    REQUIRE(first_batch.X(1, 1) == Approx(X(1, 1)));
    REQUIRE(first_batch.Y(0, 1) == Approx(Y(0, 1)));
}

TEST_CASE("Built-in losses return expected values and gradients", "[loss]") {
    nns::Matrix y_hat(2, 2);
    y_hat << 1.0, 2.0, 3.0, 4.0;

    nns::Matrix y(2, 2);
    y << 1.0, 1.0, 3.0, 5.0;

    const nns::MSELoss mse;
    REQUIRE(mse.loss(y_hat, y) == Approx(0.5));

    const nns::Matrix mse_grad = mse.gradient(y_hat, y);
    REQUIRE(mse_grad(0, 1) == Approx(0.5));
    REQUIRE(mse_grad(1, 1) == Approx(-0.5));

    nns::Matrix logits(3, 2);
    logits << 2.0, 1.0, 1.0, 0.0, 0.0, 3.0;

    nns::Matrix labels = nns::Matrix::Zero(3, 2);
    labels(0, 0) = 1.0;
    labels(2, 1) = 1.0;

    const nns::CrossEntropyLoss cross_entropy;
    const nns::Matrix probs = nns::CrossEntropyLoss::softmax(logits);
    REQUIRE(probs.col(0).sum() == Approx(1.0));
    REQUIRE(probs.col(1).sum() == Approx(1.0));
    REQUIRE(cross_entropy.loss(logits, labels) > 0.0);

    const nns::Matrix ce_grad = cross_entropy.gradient(logits, labels);
    REQUIRE(ce_grad.rows() == logits.rows());
    REQUIRE(ce_grad.cols() == logits.cols());
    REQUIRE(ce_grad.col(0).sum() == Approx(0.0).margin(1e-12));
}

TEST_CASE("SGD updates matrix weights", "[optimizer]") {
    nns::Matrix weights(1, 2);
    weights << 1.0, -1.0;

    nns::Matrix gradients(1, 2);
    gradients << 0.5, -0.5;

    nns::SGDOptimizer optimizer(nns::ConstantLR(nns::LR{0.1}));
    optimizer.update_weights(weights, gradients, {});

    REQUIRE(weights(0, 0) == Approx(0.95));
    REQUIRE(weights(0, 1) == Approx(-0.95));
}

TEST_CASE("NeuralNetwork supports predict, forward, and backward", "[network]") {
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

    nns::Matrix dY = nns::Matrix::Ones(1, 4);
    auto [dX, gradients] = net.backward(std::move(dY), cache);
    REQUIRE(dX.rows() == 2);
    REQUIRE(dX.cols() == 4);
    REQUIRE(gradients.has_value());
}

TEST_CASE("Trainer exposes fit_epoch and fit returns scalar loss history", "[trainer]") {
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

    const nns::Scalar epoch_loss = trainer.fit_epoch();
    REQUIRE(std::isfinite(epoch_loss));

    const std::vector<nns::Scalar> history = trainer.fit(2);
    REQUIRE(history.size() == 2);
    REQUIRE(std::isfinite(history[0]));
    REQUIRE(std::isfinite(history[1]));
}
