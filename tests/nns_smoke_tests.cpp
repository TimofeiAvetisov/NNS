#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <nns/NNS.hpp>

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

    REQUIRE_THROWS_AS(loader.get_batch(2), std::out_of_range);
}

TEST_CASE("DataLoader rejects invalid inputs", "[utils][dataloader]") {
    nns::Matrix X = nns::Matrix::Zero(2, 3);
    nns::Matrix Y = nns::Matrix::Zero(1, 2);

    REQUIRE_THROWS_AS(nns::DataLoader(X, Y, nns::BatchSize{1}), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::DataLoader(X, nns::Matrix::Zero(1, 3), nns::BatchSize{0}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(nns::DataLoader(nns::Matrix{2, 0}, nns::Matrix{1, 0}, nns::BatchSize{1}),
                      std::invalid_argument);

    nns::DataLoader shuffling_loader(X, nns::Matrix::Zero(1, 3), nns::BatchSize{1},
                                     nns::Shuffle{true});
    REQUIRE_THROWS_AS(shuffling_loader.reset_epoch(), std::logic_error);
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

    REQUIRE_THROWS_AS(mse.loss(y_hat, nns::Matrix::Zero(1, 2)), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::CrossEntropyLoss::softmax(nns::Matrix{0, 0}), std::invalid_argument);
}

TEST_CASE("All built-in losses reject shape mismatches", "[loss]") {
    nns::Matrix a = nns::Matrix::Ones(2, 2);
    nns::Matrix b = nns::Matrix::Ones(1, 2);

    REQUIRE_THROWS_AS(nns::MSELoss{}.gradient(a, b), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::MAELoss{}.gradient(a, b), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::HuberLoss{}.gradient(a, b), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::BCELoss{}.gradient(a, b), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::BCEWithLogitsLoss{}.gradient(a, b), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::CrossEntropyLoss{}.gradient(a, b), std::invalid_argument);
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

    REQUIRE_THROWS_AS(optimizer.update_weights(weights, nns::Matrix::Zero(2, 2), {}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(nns::ConstantLR(nns::LR{0.0}), std::invalid_argument);
}

TEST_CASE("Adam and learning rate schedulers reject invalid parameters", "[optimizer][lr]") {
    nns::Matrix weights = nns::Matrix::Ones(1, 1);
    nns::Matrix gradients = nns::Matrix::Ones(1, 1);

    nns::AdamOptimizer bad_beta1(nns::ConstantLR(nns::LR{0.01}), nns::Beta1{1.0});
    REQUIRE_THROWS_AS(bad_beta1.update_weights(weights, gradients, {}), std::invalid_argument);

    nns::AdamOptimizer bad_beta2(nns::ConstantLR(nns::LR{0.01}), nns::Beta1{0.9}, nns::Beta2{1.0});
    REQUIRE_THROWS_AS(bad_beta2.update_weights(weights, gradients, {}), std::invalid_argument);

    nns::AdamOptimizer bad_eps(nns::ConstantLR(nns::LR{0.01}), nns::Beta1{0.9}, nns::Beta2{0.999},
                               nns::Eps{0.0});
    REQUIRE_THROWS_AS(bad_eps.update_weights(weights, gradients, {}), std::invalid_argument);

    REQUIRE_THROWS_AS(nns::TimeDecayLR(nns::LR{0.0}), std::invalid_argument);
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

    REQUIRE_THROWS_AS(net.predict(nns::Matrix::Zero(3, 1)), std::invalid_argument);
    REQUIRE_THROWS_AS(net.backward(nns::Matrix::Ones(1, 4), std::any{}), std::invalid_argument);
}

TEST_CASE("NeuralNetwork rejects operations without layers", "[network]") {
    nns::NeuralNetwork net;

    REQUIRE_THROWS_AS(net.predict(nns::Matrix::Ones(1, 1)), std::logic_error);
    REQUIRE_THROWS_AS(net.forward(nns::Matrix::Ones(1, 1)), std::logic_error);
}

TEST_CASE("Layers reject invalid dimensions and backward inputs", "[layers]") {
    nns::RandomGenerator rng(nns::Seed{99});

    REQUIRE_THROWS_AS(nns::LinearLayer(nns::In{0}, nns::Out{1}, rng), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::LinearLayer(nns::In{1}, nns::Out{0}, rng), std::invalid_argument);

    nns::LinearLayer layer(nns::In{2}, nns::Out{1}, rng);
    REQUIRE_THROWS_AS(layer.predict(nns::Matrix::Zero(3, 1)), std::invalid_argument);

    auto [Y, cache] = layer.forward(nns::Matrix::Ones(2, 1));
    REQUIRE_THROWS_AS(layer.backward(nns::Matrix::Ones(2, 1), cache), std::invalid_argument);
    REQUIRE_THROWS_AS(layer.backward(nns::Matrix::Ones(1, 1), std::any{}), std::invalid_argument);

    nns::ActivationLayer activation(nns::ReLU{});
    REQUIRE_THROWS_AS(activation.predict(nns::Matrix{0, 0}), std::invalid_argument);
}

TEST_CASE("RandomGenerator rejects invalid initialization input", "[utils][random]") {
    nns::RandomGenerator rng(nns::Seed{1});
    nns::Matrix empty(0, 0);
    nns::Matrix non_empty(1, 1);

    REQUIRE_THROWS_AS(rng.init_matrix(empty), std::invalid_argument);
    REQUIRE_THROWS_AS(rng.init_matrix(non_empty, nns::Distribution::Normal, nns::Gain{0.0}),
                      std::invalid_argument);
}

TEST_CASE("load_csv parses valid files and rejects invalid files", "[utils][csv]") {
    const auto base = std::filesystem::temp_directory_path();
    const auto valid_path = base / "nns_valid.csv";
    const auto inconsistent_path = base / "nns_inconsistent.csv";

    {
        std::ofstream file(valid_path);
        file << "label,x1,x2\n";
        file << "1,0.5,0.2\n";
        file << "0,0.1,0.9\n";
    }

    auto [X, Y] = nns::load_csv(valid_path.string(), {0}, ',', true);
    REQUIRE(X.rows() == 2);
    REQUIRE(X.cols() == 2);
    REQUIRE(Y.rows() == 1);
    REQUIRE(Y.cols() == 2);
    REQUIRE(Y(0, 0) == Approx(1.0));
    REQUIRE(X(0, 1) == Approx(0.1));

    {
        std::ofstream file(inconsistent_path);
        file << "1,2,3\n";
        file << "4,5\n";
    }

    REQUIRE_THROWS_AS(nns::load_csv(valid_path.string(), {}, ',', true), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::load_csv(valid_path.string(), {4}, ',', true), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::load_csv(valid_path.string(), {0, 1, 2}, ',', true),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(nns::load_csv(inconsistent_path.string(), {0}), std::runtime_error);
    REQUIRE_THROWS_AS(nns::load_csv((base / "nns_missing.csv").string(), {0}), std::runtime_error);

    std::filesystem::remove(valid_path);
    std::filesystem::remove(inconsistent_path);
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

    trainer.reset_optimizer_state();
    REQUIRE(std::isfinite(trainer.fit_epoch()));
}

TEST_CASE("Trainer rejects loaders without full batches", "[trainer]") {
    nns::RandomGenerator rng(nns::Seed{321});
    nns::Matrix X = nns::Matrix::Ones(1, 1);
    nns::Matrix Y = nns::Matrix::Ones(1, 1);
    nns::DataLoader loader(X, Y, nns::BatchSize{2}, nns::Shuffle{false});
    nns::NeuralNetwork net(nns::LinearLayer(nns::In{1}, nns::Out{1}, rng));
    nns::AnyOptimizer optimizer =
        nns::make_AnyOptimizer(nns::SGDOptimizer(nns::ConstantLR(nns::LR{0.01})));
    nns::AnyLossFunction loss = nns::make_AnyLossFunction<nns::MSELoss>();
    nns::Trainer trainer(net, optimizer, loss, loader);

    REQUIRE_THROWS_AS(trainer.fit_epoch(), std::logic_error);
}
