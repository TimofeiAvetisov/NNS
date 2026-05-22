#include <nns/network/Network.hpp>
#include <nns/loss/AnyLossFunction.hpp>
#include <nns/utils/Random.hpp>
#include <nns/utils/LoadCSV.hpp>
#include <nns/utils/DataLoader.hpp>
#include <nns/trainer/Trainer.hpp>

#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

using namespace nns;
#define M_PI 3.14159265358979323846

double accuracy(NeuralNetwork& net, const Matrix& X, const Matrix& Y, size_t batch_size) {
    if (X.cols() != Y.cols()) {
        throw std::runtime_error("accuracy(): X and Y must have equal number of samples");
    }
    if (batch_size == 0) {
        throw std::runtime_error("accuracy(): batch_size must be > 0");
    }

    const size_t total = static_cast<size_t>(X.cols());
    size_t correct = 0;

    for (size_t start = 0; start < total; start += batch_size) {
        const size_t end = std::min(start + batch_size, total);
        const Index cols = static_cast<Index>(end - start);

        Matrix batch_X = X.middleCols(static_cast<Index>(start), cols);

        auto logits = net.predict(batch_X);

        for (Index col = 0; col < cols; ++col) {
            Index pred_class = 0;
            Index true_class = 0;

            logits.col(col).maxCoeff(&pred_class);
            Y.col(static_cast<Index>(start) + col).maxCoeff(&true_class);

            if (pred_class == true_class) {
                ++correct;
            }
        }
    }

    return static_cast<double>(correct) / static_cast<double>(total);
}

Matrix one_hot_10(const Matrix& mat) {
    Index cols = mat.cols();
    Matrix ret = Matrix::Zero(10, cols);
    for (Index i = 0; i < cols; ++i) {
        ret(static_cast<Index>(mat(0, i)), i) = 1;
    }
    return ret;
}

int main() {
    try {
        const std::string train_path = "../../MNIST/MNIST/mnist_train.csv";
        const std::string test_path = "../../MNIST/MNIST/mnist_test.csv";

        std::cout << "Loading train dataset...\n";
        auto [train_X, train_Y_t] = load_csv(train_path, {0}, (char)44, true);
        auto train_Y = one_hot_10(train_Y_t);
        train_X = train_X.unaryExpr([](double x) { return x / 255.0; });
        std::cout << "Loading test dataset...\n";
        auto [test_X, test_Y_t] = load_csv(test_path, {0}, (char)44, true);
        auto test_Y = one_hot_10(test_Y_t);
        test_X = test_X.unaryExpr([](double x) { return x / 255.0; });

        RandomGenerator rng(Seed{42});

        constexpr size_t batch_size = 100;
        constexpr size_t epochs = 10;

        DataLoader train_loader(std::move(train_X), std::move(train_Y), BatchSize{batch_size}, Shuffle{true});
        auto [xxx, yyy] = train_loader.get_batch(1);
    
        NeuralNetwork net(
            LinearLayer(In{784}, Out{128}, rng, Distribution::Normal, Gain{0.01}),
            ReLU(),
            LinearLayer(In{128}, Out{64}, rng, Distribution::Normal, Gain{0.01}),
            ReLU(),
            LinearLayer(In{64}, Out{10}, rng, Distribution::Normal, Gain{0.01})
        );

        AnyOptimizer opt = make_AnyOptimizer(
            AdamOptimizer(ConstantLR(LR{0.001}))
        );

        AnyLossFunction loss_func = make_AnyLossFunction<CrossEntropyLoss>();

        Trainer trainer(net, opt, loss_func, train_loader);

        std::cout << "Start training...\n";
        //std::vector<double> loss_history = trainer.fit(epochs, rng);

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            std::vector<double> one_epoch_loss = trainer.fit(1, rng);
            const double test_acc = accuracy(net, test_X, test_Y, batch_size);

            std::cout << "Epoch " << (epoch + 1)
                    << "/" << epochs
                    << ", loss = " << one_epoch_loss[0]
                    << ", test accuracy = " << test_acc
                    << '\n';
        }

        const double final_test_acc = accuracy(net, test_X, test_Y, batch_size);
        std::cout << "Final test accuracy: " << final_test_acc << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
