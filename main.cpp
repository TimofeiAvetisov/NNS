#include <nns/network/Network.hpp>
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
    auto rng = std::make_shared<RandomGenerator>();

    NeuralNetwork net(
        LinearLayer(2, 10, rng, InitScheme::Normal),
        ReLU(),
        LinearLayer(10, 1, rng, InitScheme::Normal),
        Zero()
    );

    Matrix X(2, 1);
    X << 1, 2;

    Matrix Y = net.forward(X);
    std::cout << Y << "\n";
}
