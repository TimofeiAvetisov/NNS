#include <nns/network/Network.hpp>
#include <nns/loss/LossFunctions.hpp>
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

using namespace std;
#define M_PI 3.14159265358979323846
int main() {
    nns::RandomGenerator::instance().reseed(42);

    const int N = 1000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-M_PI, M_PI);

    Matrix X(1, N), y(1, N);
    for (int i = 0; i < N; ++i) {
        double x = dist(rng);
        X(0, i) = x;
        y(0, i) = std::sin(x);
    }

    X /= M_PI;

    nns::NeuralNetwork net(nns::LinearLayer(nns::IN{1}, nns::OUT{32}), nns::Tanh{},
                           nns::LinearLayer(nns::IN{32}, nns::OUT{32}), nns::Tanh{},
                           nns::LinearLayer(nns::IN{32}, nns::OUT{1}));

    nns::MSELoss loss_fn;
    const int epochs = 20000;
    const int print_every = 500;

    for (int epoch = 0; epoch <= epochs; ++epoch) {
        auto [y_hat, cache] = net.forward(X);
        double loss = loss_fn.loss(y_hat, y);
        Matrix dL_dy = loss_fn.gradient(y_hat, y);
        auto grads = net.backward(dL_dy, cache);
        net.update(grads);

        if (epoch % print_every == 0) {
            cout << "epoch " << setw(6) << epoch << "  loss = " << scientific << setprecision(4)
                 << loss << "\n";
        }
    }
    cout << "\nTraining completed.\n";

    // Проверяем на нескольких точках
    cout << "\nCheck:\n";
    cout << setw(10) << "x" << setw(12) << "sin(x)" << setw(12) << "predict" << setw(12)
         << "error\n";

    Matrix test(1, 7);
    test << -M_PI, -M_PI / 2, -M_PI / 4, 0, M_PI / 4, M_PI / 2, M_PI;
    Matrix test_norm = test / M_PI;

    Matrix pred = net.predict(test_norm);
    for (int i = 0; i < test.cols(); ++i) {
        double x = test(0, i);
        double true_val = std::sin(x);
        double pred_val = pred(0, i);
        cout << fixed << setprecision(4) << setw(10) << x << setw(12) << true_val << setw(12)
             << pred_val << setw(12) << std::abs(true_val - pred_val) << "\n";
    }

    while (true) {
        double val;
        cin >> val;
        Matrix input(1, 1);
        input(0, 0) = val / M_PI;

        Matrix output = net.predict(input);
        cout << "Predicted value for x = " << val << ": " << output(0, 0) << "\n";
        cout << "True value: " << std::sin(val) << "\n";
    }

    return 0;
}