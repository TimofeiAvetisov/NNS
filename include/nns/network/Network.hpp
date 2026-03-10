#pragma once
#include <vector>
#include <memory>
#include <initializer_list>
#include <utility>
#include <stdexcept>
#include <variant>
#include <iostream>

#include <nns/core/Types.hpp>
#include <nns/layers/Layers.hpp>
#include <nns/layers/ActivationLayers.hpp>
#include <nns/layers/LinearLayers.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <nns/core/Cache.hpp>
#include <nns/activation/BuiltinActivations.hpp>
#include <nns/utils/Random.hpp>
#include <nns/optimizer/Optimizers.hpp>
#include <nns/core/OptCache.hpp>

namespace nns {
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
        (layers_.push_back(make_AnyLayer(std::forward<Args>(args))), ...);
    }

    Matrix predict(Matrix X) {
        for (size_t i = 0; i < layers_.size(); ++i) {
            X = layers_[i]->predict(std::move(X));
        }
        return X;
    }

    std::pair<Matrix, Cache> forward(Matrix X) {
        std::vector<Cache> caches_X;
        caches_X.reserve(layers_.size());
        for (size_t i = 0; i < layers_.size(); ++i) {
            auto [Y, cache_] = std::move(layers_[i]->forward(std::move(X)));
            caches_X.push_back(std::move(cache_));
            X = std::move(Y);
        }
        Cache cache_data(Data(std::move(caches_X)));
        return {X, cache_data};
    }

    LinearGrads backward(Matrix& dL_dy, const Cache& cache) {
        if (layers_.empty()) {
            throw std::runtime_error("NeuralNetwork::backward: no layers in the network");
        }

        std::vector<LinearGrads> grads;
        grads.reserve(layers_.size());
        for (size_t i = 0; i < layers_.size(); ++i) {
            size_t rev_i = layers_.size() - 1 - i;
            std::cout << "processing layer: " << rev_i << "\r";
            auto [dL_dy_new, grad_] = layers_[rev_i]->backward(std::move(dL_dy), cache[rev_i]);
            grads.push_back(std::move(grad_));
            dL_dy = std::move(dL_dy_new);
        }
        reverse(grads.begin(), grads.end());
        return LinearGrads(Data(grads));
    }

    void update(const LinearGrads& grads, const AnyOptimizer& opt, OptCache& opt_cache) {
        for (size_t i = 0; i < layers_.size(); ++i) {
            layers_[i]->update(grads[i], opt, opt_cache);
        }
        opt->step();
    }

    static void reseed_rng(uint32_t seed) {
        RandomGenerator::instance().reseed(seed);
    }

    static void show_rng_seed() {
        RandomGenerator::instance().show_seed();
    }

private:
    std::vector<AnyLayer> layers_;
};
}  // namespace nns
