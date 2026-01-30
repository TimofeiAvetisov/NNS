#pragma once
#include <nns/core/Types.hpp>
#include <nns/activation/BuiltinActivations.hpp>
#include <nns/activation/ScalarActivation.hpp>
#include <nns/nodes/ActivationNode.hpp>
#include <nns/core/Tape.hpp>
#include <stdexcept>

class ActivationLayer {
public:
    explicit ActivationLayer(AnyScalarActivation activation) : act_(std::move(activation)) {
        if (!act_.isDefined()) {
            throw std::invalid_argument("ActivationLayer constructor: activation is not defined");
        }
    }
    ActivationLayer(const ActivationLayer&) = delete;
    ActivationLayer& operator=(const ActivationLayer&) = delete;
    ActivationLayer(ActivationLayer&&) = default;
    ActivationLayer& operator=(ActivationLayer&&) = default;

    Matrix forward(const Matrix& X, Tape& tape) {
        Matrix Y = X.unaryExpr([this](double x) { return act_->forward(x); });
        tape.push(std::make_unique<ActivationNode>(X, Y, act_));
        return Y;
    }

private:
    AnyScalarActivation act_;
};