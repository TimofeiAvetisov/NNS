#include <nns/network/Network.hpp>
#include <nns/layers/LossLayers.hpp>
#include <iostream>

using namespace nns;

int main() {

    NeuralNetwork net(LinearLayer(IN{2}, OUT{10}, InitScheme::Normal), ReLU(),
                      LinearLayer(IN{10}, OUT{1}, InitScheme::Normal));

    Matrix X(2, 1);
    X << 1, 2;
    LossLayers loss(LossType::MSE);
    auto [Y, cache] = net.forward(X);
    Matrix Y_ = Matrix::Ones(1, 1);
    std::cout << Y << ' ' << Y_ << '\n';
    auto dl_dy = loss.backward(Y, Y_);
    std::cout << "Loss backward dL_dy: " << dl_dy << '\n';
    auto R = std::move(net.backward(dl_dy, cache));
    std::cout << Y << "\n";
}
