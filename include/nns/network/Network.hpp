#pragma once
#include <nns/core/Types.hpp>
#include <nns/layers/Layers.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <nns/core/Cache.hpp>
#include <nns/activation/BuiltinActivations.hpp>
#include <Random.hpp>
#include <vector>
#include <memory>
#include <initializer_list>
#include <utility>
#include <stdexcept>
#include <variant>
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

        (push_one(std::forward<Args>(args)), ...);
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
            auto [dL_dy_new, grad_] = layers_[rev_i]->backward(std::move(dL_dy), cache[rev_i]);
            grads.push_back(std::move(grad_));
            dL_dy = std::move(dL_dy_new);
        }
        return LinearGrads(Data(grads));
    }

    void update(const LinearGrads& grads /*, Optimizer opt, OptCache cache*/) {
        for (size_t i = 0; i < layers_.size(); ++i) {
            layers_[i]->update(grads[i] /*, opt, cache*/);
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
                    layers_.push_back(AnyLayer(std::move(obj)));
                } else {
                    layers_.push_back(AnyLayer(ActivationLayer(std::move(obj))));
                }
            },
            std::move(v));
    }
};
}  // namespace nns