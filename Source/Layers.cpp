#include <Layers.hpp>
#include <stdexcept>


LinearLayer::LinearLayer(size_t in_dim, size_t out_dim, double std_dev) {
    A_ = Matrix::Random(static_cast<int>(out_dim), static_cast<int>(in_dim)) * std_dev;
    b_ = Vector::Random(static_cast<int>(out_dim)) * std_dev;

    dA_.setZero(A_.rows(), A_.cols());
    db_.setZero(b_.size());
}

LinearLayer::LinearLayer(const Matrix& A_init, const Vector& b_init) : A_(A_init), b_(b_init) {
    dA_.setZero(A_.rows(), A_.cols());
    db_.setZero(b_.size());
}

Matrix LinearLayer::forward(const Matrix& X) {
    last_X_ = X;
    Matrix Y = A_ * X;
    Y.colwise() += b_;
    return Y;
}

Matrix LinearLayer::backward(const Matrix& dY) {
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


ActivationLayer::ActivationLayer(ActType type) : type_(type) {}

Matrix ActivationLayer::forward(const Matrix& X) {
    last_X_ = X;

    switch (type_) {
        case ActType::ReLU:
            last_Y_ = X.cwiseMax(0.0);
            break;
        case ActType::Sigmoid:
            last_Y_ = (1.0 / (1.0 + (-X.array()).exp())).matrix();
            break;
        case ActType::Tanh:
            last_Y_ = X.array().tanh().matrix();
            break;
        default:
            throw std::invalid_argument("Invalid activation type");
    }
    return last_Y_;
}

Matrix ActivationLayer::backward(const Matrix& dY) {
    switch (type_) {
        case ActType::ReLU: {
            Matrix mask = (last_X_.array() > 0.0).cast<double>().matrix();
            return dY.cwiseProduct(mask);
        }
        case ActType::Sigmoid: {
            return (dY.array() * (last_Y_.array() * (1.0 - last_Y_.array()))).matrix(); // бля че за хуетень
        }
        case ActType::Tanh: {
            return (dY.array() * (1.0 - last_Y_.array().square())).matrix();
        }
        default:
            throw std::invalid_argument("Invalid activation type");
    }
}

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