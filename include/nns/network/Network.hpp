#pragma once
#include <nns/core/Types.hpp>
#include <nns/layers/Layers.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <nns/activation/BuiltinActivations.hpp>
#include <Random.hpp>
#include <vector>
#include <memory>
#include <initializer_list>
#include <utility>
#include <stdexcept>
#include <variant>

class NeuralNetwork {
public:
    // move-only
    NeuralNetwork() = default;
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;
    NeuralNetwork(NeuralNetwork&&) = default;
    NeuralNetwork& operator=(NeuralNetwork&&) = default;

    template <class... Args>
    explicit NeuralNetwork(Args&&... args) {
        layers_.reserve(sizeof...(Args));

        (push_one(std::forward<Args>(args)), ...);
    }

    Matrix predict(Matrix X) {
        for (size_t i = 0; i < layers_.size(); ++i) {
            X = std::move(layers_[i]->predict(X));
        }
        return X;
    }

    Matrix forward(Matrix X) {
        for (size_t i = 0; i < layers_.size(); ++i) {
            X = std::move(layers_[i]->forward(std::move(X)));
        }
        return X;
    }

    std::vector<LinearGrads> backward(Matrix& dL_dy) {
        if (layers_.empty()) {
            throw std::runtime_error("NeuralNetwork::backward: no layers in the network");
        }

        std::vector<LinearGrads> grads;
        grads.reserve(layers_.size());
        for (size_t i = 0; i < layers_.size(); ++i) {
            grads.push_back(std::move(layers_[i]->form_grads()));
        }

        for (size_t i = 0; i < layers_.size(); ++i) {
            size_t rev_i = layers_.size() - 1 - i;
            dL_dy = std::move(layers_[rev_i]->backward(std::move(dL_dy), grads[rev_i]));
        }
        return grads;
    }

    void update_weights(double lr /*should be optimizer*/, std::vector<LinearGrads> grads) {
        for (size_t i = 0; i < layers_.size(); ++i) {
            layers_[i]->sgd_step(lr /*should be optimizer*/, std::move(grads[i]));
        }
    }

    static void reseed_rng(uint32_t seed) {
        RandomGenerator::instance().reseed(seed);
    }

    static void show_rng_seed() {
        RandomGenerator::instance().show_seed();
    }

private:
    std::vector<AnyLayer> layers_;

    template <class T>
    void push_one(T&& x) {
        static_assert(
            std::is_constructible_v<LinearLayer, T&&> ||
                std::is_constructible_v<AnyScalarActivation, T&&>,
            "NeuralNetwork ctor: Must be constructible as LinearLayer or AnyScalarActivation");
        LayerVariant v{std::forward<T>(x)};
        push_variant(std::move(v));
    }

    using LayerVariant = std::variant<LinearLayer, AnyScalarActivation>;
    void push_variant(LayerVariant&& v) {
        std::visit(
            [&](auto&& obj) {
                using U = std::remove_cvref_t<decltype(obj)>;

                if constexpr (std::is_same_v<U, LinearLayer>) {
                    const auto out_dim = obj.out_dim();
                    const auto in_dim = obj.in_dim();

                    layers_.push_back(AnyLayer(std::move(obj)));
                } else {
                    layers_.push_back(AnyLayer(ActivationLayer(std::move(obj))));
                }
            },
            std::move(v));
    }
};
