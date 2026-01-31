#pragma once

#include <TypeErasure.hpp>
#include <concepts>
#include <memory>
#include <utility>

#include <nns/core/Types.hpp>
#include <nns/core/Tape.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <nns/layers/LinearLayers.hpp>
#include <nns/layers/ActivationLayers.hpp>

struct ILayer {
    virtual Matrix forward(const Matrix& X, Tape& tape,
                           LinearGrads* grads) = 0;  // ActivationLayer ignores grads
    virtual Matrix predict(const Matrix& X);
    virtual void sgd_step(double lr, LinearGrads* grads) = 0;  // ActivationLayer do nothing
    virtual ~ILayer() = default;
};

template <class T>
concept LayerLike =
    requires(T& t, const Matrix& X, Tape& tape, LinearGrads* g, const double lr, const bool ist) {
        { t.forward(X, tape, g, ist) } -> std::same_as<Matrix>;
        { t.sgd_step(lr, g) } -> std::same_as<void>;
        { t.predict(X) } -> std::same_as<Matrix>;
    };

template <class Base, class TObject>
class CLayerImpl final : public Base {
    static_assert(LayerLike<TObject>,
                  "Layer must provide: "
                  "Matrix forward(const Matrix&, Tape&, LinearGrads*, bool is_training), "
                  "void sgd_step(double, LinearGrads*), "
                  "Matrix predict(const Matrix&)");

public:
    template <class U>
    explicit CLayerImpl(U&& obj) : object_(std::forward<U>(obj)) {
    }

    Matrix forward(const Matrix& X, Tape& tape, LinearGrads* grads,
                   bool is_training = true) override {
        return object_.forward(X, tape, grads, is_training);
    }

    Matrix predict(const Matrix& X) override {
        return object_.predict(X);
    }

    void sgd_step(double lr, LinearGrads* grads) override {
        object_.sgd_step(lr, grads);
    }

private:
    TObject object_;
};

using AnyLayer = CAnyMovable<ILayer, CLayerImpl>;

inline AnyLayer MakeLinearLayer(size_t in_dim, size_t out_dim,
                                InitScheme init_scheme = InitScheme::XavierNormal,
                                double gain = 1.0, std::shared_ptr<RandomGenerator> rng) {
    return AnyLayer(LinearLayer(in_dim, out_dim, init_scheme, gain, rng));
}

inline AnyLayer MakeLinearLayer(const Matrix& A_init, const Vector& b_init) {
    return AnyLayer(LinearLayer(A_init, b_init));
}

inline AnyLayer MakeActivationLayer(AnyScalarActivation activation) {
    return AnyLayer(ActivationLayer(std::move(activation)));
}
