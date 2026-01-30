#pragma once

#include <Eigen/Dense>
#include <Utils.hpp>
#include <memory>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

class LinearLayer {
public:
    LinearLayer(const Matrix& A_init, const Vector& b_init);
    LinearLayer(size_t in_dim, size_t out_dim, InitScheme init_scheme = InitScheme::XavierNormal,
                double gain = 1.0, std::shared_ptr<RandomGenerator> rng = nullptr);

    /**
     * @brief Forward pass through the linear layer
     * @param X Input matrix of shape (in_dim, batch_size)
     * @return Output matrix of shape (out_dim, batch_size)
     */
    Matrix forward(const Matrix& X);

    /**
     * @brief Backward pass through the linear layer
     * @param dY Gradient of the output matrix of shape (out_dim, batch_size)
     * @return Gradient of the input matrix of shape (in_dim, batch_size)
     */
    Matrix backward(const Matrix& dY);

    /**
     * @brief Zero the gradients
     */
    void zero_grad();

    /**
     * @brief Update the weights and biases
     * @param lr Learning rate
     */
    void step(double lr);

    /**
     * @brief Get the input dimension
     * @return Input dimension
     */
    int in_dim() const;

    /**
     * @brief Get the output dimension
     * @return Output dimension
     */
    int out_dim() const;

private:
    Matrix A_;
    Vector b_;

    Matrix last_X_;

    Matrix dA_;
    Vector db_;
};

enum class ActType { ReLU, Sigmoid, Tanh };

class ActivationLayer {
public:
    explicit ActivationLayer(ActType type);

    Matrix forward(const Matrix& X);
    Matrix backward(const Matrix& dY);

private:
    ActType type_;
    Matrix last_X_;
    Matrix last_Y_;
};

enum class LossType { MSE };

class Loss {
public:
    explicit Loss(LossType type);

    double forward(const Matrix& y_hat, const Matrix& y) const;
    Matrix backward(const Matrix& y_hat, const Matrix& y) const;

private:
    LossType type_;
};
