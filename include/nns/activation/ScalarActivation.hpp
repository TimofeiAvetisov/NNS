#pragma once
#include <TypeErasure.hpp>
#include <concepts>
#include <nns/core/Types.hpp>

struct IScalarActivation {
    virtual double forward(double x) const = 0;
    virtual double derivative(double x,
                              double y) const = 0;  // sometimes y is useful (e.g. for sigmoid)
    virtual ~IScalarActivation() = default;
};

template <class T>
concept ScalarActivationLike = requires(const T& t, const double x, const double y) {
    { t.forward(x) } -> std::same_as<double>;
    { t.derivative(x, y) } -> std::same_as<double>;
};

template <class Base, class TObject>
class CScalarActivationImpl : public Base {
    static_assert(ScalarActivationLike<TObject>,
                  "Scalar activation must provide: double forward(double) const and double "
                  "derivative(double, double) const");

public:
    template <class U>
    explicit CScalarActivationImpl(U&& obj) : object_(std::forward<U>(obj)) {
    }

    double forward(double x) const override {
        return object_.forward(x);
    }

    double derivative(double x, double y) const override {
        return object_.derivative(x, y);
    }

private:
    TObject object_;
};

using AnyScalarActivation = CAnyMovable<IScalarActivation, CScalarActivationImpl>;