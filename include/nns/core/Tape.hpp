#pragma once
#include <nns/core/Types.hpp>
#include <vector>
#include <memory>
#include <utility>

struct TapeNode {
    virtual ~TapeNode() = default;
    virtual Matrix backward(const Matrix& grad_output) = 0;
};

struct Tape {
    std::vector<std::unique_ptr<TapeNode>> nodes;

    void push(std::unique_ptr<TapeNode> node) {
        nodes.push_back(std::move(node));
    }

    Matrix backward(Matrix grad) {
        for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
        nodes.clear();
        return grad;
    }

    void clear() {
        nodes.clear();
    }
};

using TapePtr = std::shared_ptr<Tape>;
