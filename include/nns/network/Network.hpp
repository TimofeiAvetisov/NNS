#pragma once
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
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
        Matrix Y;
        for (const auto& layer : layers_) {
            Y = layer->predict(X);
        }
        return Y;
    }

    std::pair<Matrix, std::any> forward(Matrix&& X) {
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
        std::vector<std::any> grads;
        grads.reserve(layers_.size());

        assert(layers_cache.has_value() && "Empty cache passed in Newtorwk::backward");

        const std::vector<std::any>& cache =
            std::any_cast<const std::vector<std::any>&>(layers_cache);
        for (size_t i = 0; i < layers_.size(); ++i) {
            size_t rev_i = layers_.size() - 1 - i;
            auto [dL_dy_new, grad] = layers_[rev_i]->backward(std::move(dL_dy), cache[rev_i]);
            grads.push_back(std::move(grad));
            dL_dy = std::move(dL_dy_new);
        }
        reverse(grads.begin(), grads.end());

        return std::make_pair(std::move(dL_dy), std::any(std::move(grads)));
    }

    std::any update(std::any&& layers_gradients, AnyOptimizer& opt, std::any&& opt_cache) {
        std::vector<std::any> grads =
            std::any_cast<std::vector<std::any>&&>(std::move(layers_gradients));
        std::vector<std::any> cache;
        if (opt_cache.has_value()) {
            cache = std::any_cast<std::vector<std::any>&&>(std::move(opt_cache));
        } else {
            cache.assign(layers_.size(), std::any{});
        }

        for (size_t i = 0; i < layers_.size(); ++i) {
            cache[i] = layers_[i]->update(std::move(grads[i]), opt, std::move(cache[i]));
        }
        return cache;
    }

private:
    std::vector<AnyLayer> layers_;
};
}  // namespace nns
