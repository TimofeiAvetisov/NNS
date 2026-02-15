#pragma once

#include <TypeErasure.hpp>
#include <concepts>
#include <memory>
#include <utility>

#include <nns/core/Types.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <nns/layers/LinearLayers.hpp>
#include <nns/layers/ActivationLayers.hpp>
namespace nns {
struct ILayer {
    virtual Matrix forward(Matrix X) = 0;
    virtual Matrix predict(Matrix X) = 0;
    virtual Matrix backward(Matrix dY, LinearGrads* grads) = 0;
    virtual void sgd_step(double lr, LinearGrads grads) = 0;  // ActivationLayer do nothing
    virtual LinearGrads form_grads() const = 0;
    virtual ~ILayer() = default;
};

template <class T>
concept LayerLike = requires(T& t, Matrix X, LinearGrads g, double lr, LinearGrads* grads_ptr) {
    { t.forward(X) } -> std::same_as<Matrix>;
    { t.sgd_step(lr, g) } -> std::same_as<void>;
    { t.predict(X) } -> std::same_as<Matrix>;
    { t.backward(X, grads_ptr) } -> std::same_as<Matrix>;
    { t.form_grads() } -> std::same_as<LinearGrads>;
};

template <class Base, class TObject>
class CLayerImpl : public Base {
    static_assert(LayerLike<TObject>,
                  "Layer must provide: "
                  "Matrix forward(Matrix), "
                  "void sgd_step(double, LinearGrads), "
                  "Matrix predict(Matrix), "
                  "Matrix backward(Matrix, LinearGrads*), "
                  "LinearGrads form_grads() const");

public:
    template <class U>
    explicit CLayerImpl(U&& obj) : object_(std::forward<U>(obj)) {
    }

    Matrix forward(Matrix X) override {
        return object_.forward(std::move(X));
    }

    Matrix predict(Matrix X) override {
        return object_.predict(std::move(X));
    }

    Matrix backward(Matrix dY, LinearGrads* grads) override {

        return object_.backward(std::move(dY), grads);
    }

    LinearGrads form_grads() const override {
        return object_.form_grads();
    }

    void sgd_step(double lr, LinearGrads grads) override {
        object_.sgd_step(lr, std::move(grads));
    }

private:
    TObject object_;
};

using AnyLayer = CAnyMovable<ILayer, CLayerImpl>;

inline AnyLayer MakeLinearLayer(IN in_dim, OUT out_dim,
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
}  // namespace nns
