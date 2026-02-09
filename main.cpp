#include <nns/network/Network.hpp>
#include <nns/layers/LossLayers.hpp>
#include <iostream>
struct Zero {
    double forward(double x) const {
        return 0;
    }

    double derivative(double x, double y) const{
        return 0;
    }
};

int main() {

    NeuralNetwork net(
        LinearLayer(2, 10, InitScheme::Normal),
        ReLU(),
        LinearLayer(10, 1, InitScheme::Normal),
        Zero()
    );

    Matrix X(2, 1);
    X << 1, 2;
    LossLayers loss(LossType::MSE);
    Matrix Y = net.forward(X);
    auto dl_dy = loss.backward(Y, Y);
    auto R = net.backward(dl_dy);
    for (auto r : R) {
        std::cout << r.dA << ' ' << r.db << '\n';
    }
    std::cout << Y << "\n";

}
