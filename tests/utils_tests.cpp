#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <nns/NNS.hpp>

using Catch::Approx;

TEST_CASE("RandomGenerator stores seed, reseeds, shuffles, and samples values", "[utils][random]") {
    nns::RandomGenerator rng(nns::Seed{42});
    REQUIRE(rng.seed() == 42u);

    rng.reseed(7);
    REQUIRE(rng.seed() == 7u);

    std::uniform_real_distribution<nns::Scalar> distribution(0.0, 1.0);
    const nns::Scalar value = rng.get_random_value(distribution);
    REQUIRE(value >= 0.0);
    REQUIRE(value <= 1.0);

    std::vector<int> values{1, 2, 3, 4};
    rng.shuffle(values);
    REQUIRE(values.size() == 4);
}

TEST_CASE("RandomGenerator initializes matrices with supported distributions", "[utils][random]") {
    nns::RandomGenerator rng(nns::Seed{1});

    for (auto dist : {nns::Distribution::Normal, nns::Distribution::XavierNormal,
                      nns::Distribution::HeNormal}) {
        nns::Matrix matrix = nns::Matrix::Zero(3, 2);
        rng.init_matrix(matrix, dist, nns::Gain{1.0});
        REQUIRE(matrix.rows() == 3);
        REQUIRE(matrix.cols() == 2);
        REQUIRE(matrix.array().isFinite().all());
    }

    nns::Matrix empty(0, 0);
    nns::Matrix non_empty(1, 1);
    REQUIRE_THROWS_AS(rng.init_matrix(empty), std::invalid_argument);
    REQUIRE_THROWS_AS(rng.init_matrix(non_empty, nns::Distribution::Normal, nns::Gain{0.0}),
                      std::invalid_argument);
}

TEST_CASE("DataLoader splits matrices into full batches and drops partial batch",
          "[utils][dataloader]") {
    nns::Matrix X(2, 5);
    X << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0;

    nns::Matrix Y(1, 5);
    Y << 10.0, 20.0, 30.0, 40.0, 50.0;

    nns::DataLoader loader(X, Y, nns::BatchSize{2}, nns::Shuffle{false});

    REQUIRE(loader.size() == 5);
    REQUIRE(loader.batch_size() == 2);
    REQUIRE(loader.num_batches() == 2);

    size_t batches = 0;
    for (auto [bX, bY] : loader) {
        REQUIRE(bX.cols() == 2);
        REQUIRE(bY.cols() == 2);
        ++batches;
    }
    REQUIRE(batches == 2);

    const auto first_batch = loader.get_batch(0);
    REQUIRE(first_batch.X(0, 0) == Approx(X(0, 0)));
    REQUIRE(first_batch.Y(0, 1) == Approx(Y(0, 1)));
    REQUIRE_THROWS_AS(loader.get_batch(2), std::out_of_range);
}

TEST_CASE("DataLoader rejects invalid inputs and invalid reset mode", "[utils][dataloader]") {
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

    nns::RandomGenerator rng(nns::Seed{4});
    REQUIRE_NOTHROW(shuffling_loader.reset_epoch(rng));
}

TEST_CASE("load_csv parses valid files and rejects invalid files", "[utils][csv]") {
    const auto base = std::filesystem::temp_directory_path();
    const auto valid_path = base / "nns_valid.csv";
    const auto inconsistent_path = base / "nns_inconsistent.csv";
    const auto empty_path = base / "nns_empty.csv";

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
    {
        std::ofstream file(empty_path);
    }

    REQUIRE_THROWS_AS(nns::load_csv(valid_path.string(), {}, ',', true), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::load_csv(valid_path.string(), {4}, ',', true), std::invalid_argument);
    REQUIRE_THROWS_AS(nns::load_csv(valid_path.string(), {0, 1, 2}, ',', true),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(nns::load_csv(inconsistent_path.string(), {0}), std::runtime_error);
    REQUIRE_THROWS_AS(nns::load_csv(empty_path.string(), {0}), std::runtime_error);
    REQUIRE_THROWS_AS(nns::load_csv((base / "nns_missing.csv").string(), {0}), std::runtime_error);

    std::filesystem::remove(valid_path);
    std::filesystem::remove(inconsistent_path);
    std::filesystem::remove(empty_path);
}
