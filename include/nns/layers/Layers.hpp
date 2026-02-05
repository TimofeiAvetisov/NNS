#pragma once

#include <TypeErasure.hpp>
#include <concepts>
#include <memory>
#include <utility>

#include <nns/core/Types.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <nns/layers/LinearLayers.hpp>
#include <nns/layers/ActivationLayers.hpp>

struct ILayer {
    virtual Matrix forward(Matrix X) = 0;
    virtual Matrix predict(const Matrix& X) = 0;
    virtual Matrix backward(Matrix dY, LinearGrads& grads) = 0;
    virtual void sgd_step(double lr, LinearGrads& grads) = 0;  // ActivationLayer do nothing
    virtual LinearGrads form_grads() const = 0;
    virtual ~ILayer() = default;
};

template <class T>
concept LayerLike =
    requires(T& t, const Matrix& X, LinearGrads* g, const double lr) {
        { t.forward(X) } -> std::same_as<Matrix>;
        { t.sgd_step(lr, g) } -> std::same_as<void>;
        { t.predict(X) } -> std::same_as<Matrix>;
        { t.backward(X, g) } -> std::same_as<Matrix>;
        { t.form_grads() } -> std::same_as<LinearGrads>;
    };

template <class Base, class TObject>
class CLayerImpl : public Base {
    static_assert(LayerLike<TObject>,
                  "Layer must provide: "
                  "Matrix forward(const Matrix&), "
                  "void sgd_step(double, LinearGrads&), "
                  "Matrix predict(const Matrix&), "
                  "Matrix backward(const Matrix&, LinearGrads&), "
                  "LinearGrads form_grads() const");

public:
    template <class U>
    explicit CLayerImpl(U&& obj) : object_(std::forward<U>(obj)) {
    }

    Matrix forward(Matrix X) override {
        return object_.forward(X);
    }

    Matrix predict(const Matrix& X) override {
        return object_.predict(X);
    }

    Matrix backward(Matrix dY, LinearGrads& grads) override {

        return object_.backward(dY, grads);
    }

    LinearGrads form_grads() const override {
        return object_.form_grads();
    }

    void sgd_step(double lr, LinearGrads& grads) override {
        object_.sgd_step(lr, grads);
    }

private:
    TObject object_;
};

using AnyLayer = CAnyMovable<ILayer, CLayerImpl>;

inline AnyLayer MakeLinearLayer(size_t in_dim, size_t out_dim,
                                InitScheme init_scheme = InitScheme::XavierNormal,
                                double gain = 1.0) {
    return AnyLayer(LinearLayer(in_dim, out_dim, init_scheme, gain));
}

inline AnyLayer MakeLinearLayer(const Matrix& A_init, const Vector& b_init) {
    return AnyLayer(LinearLayer(A_init, b_init));
}

inline AnyLayer MakeActivationLayer(AnyScalarActivation activation) {
    return AnyLayer(ActivationLayer(std::move(activation)));
}
