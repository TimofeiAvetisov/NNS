#pragma once
#include <nns/core/Types.hpp>
#include <nns/layers/Layers.hpp>
#include <nns/core/Tape.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <nns/activation/BuiltinActivations.hpp>
#include <nns/layers/LossLayers.hpp>
#include <Random.hpp>
#include <vector>
#include <memory>
#include <initializer_list>
#include <utility>
#include <stdexcept>


class NeuralNetwork {
public:
    // initializer list of layers {{linlayersz_1, linlayersz_2, act1}, {linlayersz_2, linlayersz_3,
    // act2}, ...} somehow need to pass LossLayer
    NeuralNetwork(std::initializer_list<std::pair<std::pair<size_t, size_t>, ActivationFactory>>
                      layer_configs) {
        rng_ = std::make_shared<RandomGenerator>();
        grads_.reserve(layer_configs.size() *
                       2);  // each layer pair has LinearLayer and ActivationLayer
        layers_.reserve(layer_configs.size() * 2);
        for (const auto& config : layer_configs) {
            size_t in_dim = config.first.first;
            size_t out_dim = config.first.second;
            ActivationFactory act_factory = config.second;

            layers_.push_back(std::make_unique<AnyLayer>(
                MakeLinearLayer(in_dim, out_dim, rng_, InitScheme::XavierNormal, 1.0)));
            layers_.push_back(std::make_unique<AnyLayer>(ActivationLayer(act_factory())));
            grads_.emplace_back(out_dim, in_dim);
            grads_.emplace_back(0, 0);  // ActivationLayer has no grads
        }
    }

    Matrix predict(const Matrix& X) {
        Matrix out = X;
        for (const auto& layer : layers_) {
            out = (*layer)->predict(out);
        }
        return out;
    }

    double train_step(const Matrix& X, const Matrix& y, double lr) {
        Tape tape;  // need to review it

        // Forward pass
        Matrix out = X;
        for (size_t i = 0; i < layers_.size(); ++i) {
            out = (*layers_[i])->forward(out, tape, &grads_[i]);
        }
        // Compute loss
        const double loss = loss_layer_.forward(out, y);

        // Compute initial gradient from loss
        Matrix dL_dy = loss_layer_.backward(out, y);

        // Backward pass
        tape.backward(dL_dy);

        // SGD step aka update weights
        for (size_t i = 0; i < layers_.size(); ++i) {
            (*layers_[i])->sgd_step(lr, &grads_[i]);
        }
        zero_grads();
        return loss;
    }

    void set_loss(LossType type = LossType::MSE) {  // Need to think about it
        loss_layer_ = LossLayers(type);
    }

    void reseed_rng(uint32_t seed) {
        rng_->reseed(seed);
    }

    void show_rng_seed() {
        rng_->show_seed();
    }

    void zero_grads() {
        for (auto& grad : grads_) {
            grad.set_zero();
        }
    }

private:
    std::shared_ptr<RandomGenerator> rng_;
    std::vector<std::unique_ptr<AnyLayer>> layers_;
    std::vector<LinearGrads> grads_;
    LossLayers loss_layer_{LossType::MSE};
};