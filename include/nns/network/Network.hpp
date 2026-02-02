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
        grads_.reserve(sizeof...(Args));

        (push_one(std::forward<Args>(args)), ...);
    }

    Matrix predict(const Matrix& X) {
        Matrix out = X;
        for (const auto& layer : layers_) {
            out = (*layer)->predict(out);
        }
        return out;
    }

    Matrix forward(const Matrix& X, const Matrix& y) {
        Matrix out = X;
        for (size_t i = 0; i < layers_.size(); ++i) {
            out = (*layers_[i])->forward(out, tape_, &grads_[i]);
        }
        return out;
    }

    void backward(Matrix& dL_dy) {
        tape_.backward(dL_dy);
    }

    void update_weights(double lr/*should be optimizer*/) {
        for (size_t i = 0; i < layers_.size(); ++i) {
            (*layers_[i])->sgd_step(lr/*should be optimizer*/, &grads_[i]);
        }
        zero_grads();
    }

    void reseed_rng(uint32_t seed) {
        rng_->reseed(seed);
    }

    void show_rng_seed() {
        rng_->show_seed();
    }


private:
    std::shared_ptr<RandomGenerator> rng_;
    std::vector<std::unique_ptr<AnyLayer>> layers_;
    std::vector<LinearGrads> grads_;
    Tape tape_;

    template <class T>
    void push_one(T&& x) {
        LayerVariant v{std::forward<T>(x)};
        push_variant(std::move(v));
    }

    using LayerVariant = std::variant<LinearLayer, AnyScalarActivation>;
    void push_variant(LayerVariant&& v) {
        std::visit([&](auto&& obj) {
            using U = std::remove_cvref_t<decltype(obj)>;

            if constexpr (std::is_same_v<U, LinearLayer>) {
                const auto out_dim = obj.out_dim();
                const auto in_dim  = obj.in_dim();

                layers_.push_back(std::make_unique<AnyLayer>(std::move(obj)));
                grads_.emplace_back(out_dim, in_dim);
            } else {
                layers_.push_back(std::make_unique<AnyLayer>(
                    ActivationLayer(std::move(obj))
                ));
                grads_.emplace_back(0, 0);
            }
        }, std::move(v));
    }

    void zero_grads() {
        for (auto& grad : grads_) {
            grad.set_zero();
        }
    }
};