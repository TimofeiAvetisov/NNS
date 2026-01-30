#include <Layers.hpp>
#include <stdexcept>

// may be different initialization methods based on activation functions not just by hands
LinearLayer::LinearLayer(size_t in_dim, size_t out_dim,
                         InitScheme init_scheme, double gain,
                         std::shared_ptr<RandomGenerator> rng) {
    if (!rng) {
        throw std::invalid_argument("LinearLayer constructor: rng pointer is null");
    }
    A_ = rng->init_linear_weights(out_dim, in_dim, init_scheme, gain);
    b_ = Vector::Zero(static_cast<int>(out_dim));

    dA_.setZero(A_.rows(), A_.cols());
    db_.setZero(b_.size());
}

LinearLayer::LinearLayer(const Matrix& A_init, const Vector& b_init) {  // idk if needed
    if (A_init.rows() != b_init.size()) {
        throw std::invalid_argument("Dimension mismatch between A_init and b_init");
    }
    A_ = A_init;
    b_ = b_init;
    dA_.setZero(A_.rows(), A_.cols());
    db_.setZero(b_.size());
}

Matrix LinearLayer::forward(const Matrix& X) {
    if (X.rows() != A_.cols()) {
        throw std::invalid_argument("LinearLayer::forward: dimension mismatch between A_ and X");
    }
    last_X_ = X;
    Matrix Y = A_ * X;
    Y.colwise() += b_;
    return Y;
}

Matrix LinearLayer::backward(const Matrix& dY) {
    if (dY.rows() != A_.cols()) {
        throw std::invalid_argument("LinearLayer::backward: dimension mismatch between A_ and dY");
    }
    if (dY.cols() != last_X_.rows()) {
        throw std::invalid_argument("LinearLayer::backward: dimension mismatch between last_X_ and dY");
    }
    dA_ = dY * last_X_.transpose();
    db_ = dY.rowwise().sum();
    return A_.transpose() * dY;
}

void LinearLayer::zero_grad() {
    dA_.setZero(A_.rows(), A_.cols());
    db_.setZero(b_.size());
}

void LinearLayer::step(double lr) {
    A_ -= lr * dA_;
    b_ -= lr * db_;
}

int LinearLayer::in_dim() const {
    return static_cast<int>(A_.cols());
}
int LinearLayer::out_dim() const {
    return static_cast<int>(A_.rows());
}

ActivationLayer::ActivationLayer(AnyScalarActivation activation) : activation_(std::move(activation)) {
    if (!activation_.isDefined()) {
        throw std::invalid_argument("ActivationLayer constructor: activation is not defined");
    }
}

Matrix ActivationLayer::forward(const Matrix& X) {
    last_X_ = X;
    Matrix Y = X.unaryExpr([this](double x)  {return activation_->forward(x);});
    last_Y_ = Y;
    return Y;
}

Matrix ActivationLayer::backward(const Matrix& dY) {
    if ((dY.rows() != last_Y_.rows()) || (dY.cols() != last_Y_.cols())) {
        throw std::invalid_argument("ActivationLayer::backward: dimension mismatch between dY and last_Y_");
    }

    Matrix dX = dY;
    for (int i = 0; i < dY.rows(); ++i) {
        for (int j = 0; j < dY.cols(); ++j) {
            double x = last_X_(i, j);
            double y = last_Y_(i, j);
            dX(i, j) *= activation_->derivative(x, y);
        }
    }
    return dX;
}

// Should make as ActibationLayer above thru type erasure
Loss::Loss(LossType type) : type_(type) {
}

double Loss::forward(const Matrix& y_hat, const Matrix& y) const {
    switch (type_) {
        case LossType::MSE: {
            // MSE loss function
            const double n = static_cast<double>(y_hat.size());
            return (y_hat - y).squaredNorm() / n;
        }
        default:
            throw std::invalid_argument("Invalid loss type");
    }
}

Matrix Loss::backward(const Matrix& y_hat, const Matrix& y) const {
    switch (type_) {
        case LossType::MSE: {
            const double n = static_cast<double>(y_hat.size());
            return (2.0 / n) * (y_hat - y);
        }
        default:
            throw std::invalid_argument("Invalid loss type");
    }
}