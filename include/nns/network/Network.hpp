#pragma once
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nns/activation/BuiltinActivations.hpp>
#include <nns/core/Types.hpp>
#include <nns/layers/ActivationLayers.hpp>
#include <nns/layers/AnyLayer.hpp>
#include <nns/layers/LinearLayers.hpp>
#include <nns/optimizer/AnyOptimizer.hpp>

namespace nns {
class NeuralNetwork {
public:
    template <class... Args>
    explicit NeuralNetwork(Args&&... args) {
        layers_.reserve(sizeof...(Args));
        (layers_.push_back(make_AnyLayer(std::forward<Args>(args))), ...);
    }

    Matrix predict(const Matrix& X) const {
        validate_has_layers("predict");
        Matrix Y = X;
        for (const auto& layer : layers_) {
            Y = layer->predict(Y);
        }
        return Y;
    }

    std::pair<Matrix, std::any> forward(Matrix&& X) {
        validate_has_layers("forward");
        std::vector<std::any> layers_cache;
        layers_cache.reserve(layers_.size());
        for (AnyLayer& layer : layers_) {
            auto [Y, cache] = layer->forward(std::move(X));
            X = std::move(Y);
            layers_cache.push_back(std::move(cache));
        }
        return std::make_pair(std::move(X), std::move(layers_cache));
    }

    std::pair<Matrix, std::any> backward(Matrix&& dL_dy, const std::any& layers_cache) {
        validate_has_layers("backward");
        std::vector<std::any> grads;
        grads.reserve(layers_.size());

        if (!layers_cache.has_value()) {
            throw std::invalid_argument("NeuralNetwork::backward: layers_cache is empty");
        }

        const std::vector<std::any>& cache =
            std::any_cast<const std::vector<std::any>&>(layers_cache);
        if (cache.size() != layers_.size()) {
            throw std::invalid_argument(
                "NeuralNetwork::backward: cache size must match number of layers");
        }

        for (size_t i = 0; i < layers_.size(); ++i) {
            size_t rev_i = layers_.size() - 1 - i;
            auto [dL_dy_new, grad] = layers_[rev_i]->backward(std::move(dL_dy), cache[rev_i]);
            grads.push_back(std::move(grad));
            dL_dy = std::move(dL_dy_new);
        }
        std::reverse(grads.begin(), grads.end());

        return std::make_pair(std::move(dL_dy), std::any(std::move(grads)));
    }

    std::any update(std::any&& layers_gradients, AnyOptimizer& opt, std::any&& opt_cache) {
        validate_has_layers("update");
        std::vector<std::any> grads =
            std::any_cast<std::vector<std::any>&&>(std::move(layers_gradients));
        if (grads.size() != layers_.size()) {
            throw std::invalid_argument(
                "NeuralNetwork::update: gradients size must match number of layers");
        }

        std::vector<std::any> cache;
        if (opt_cache.has_value()) {
            cache = std::any_cast<std::vector<std::any>&&>(std::move(opt_cache));
            if (cache.size() != layers_.size()) {
                throw std::invalid_argument(
                    "NeuralNetwork::update: optimizer cache size must match number of layers");
            }
        } else {
            cache.assign(layers_.size(), std::any{});
        }

        for (size_t i = 0; i < layers_.size(); ++i) {
            cache[i] = layers_[i]->update(std::move(grads[i]), opt, std::move(cache[i]));
        }
        return cache;
    }

private:
    void validate_has_layers(const char* method) const {
        if (layers_.empty()) {
            throw std::logic_error(std::string("NeuralNetwork::") + method +
                                   ": network must contain at least one layer");
        }
    }

    std::vector<AnyLayer> layers_;
};
}  // namespace nns
