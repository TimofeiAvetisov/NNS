#include <Eigen/Dense>
#include <iostream>
#include <Layers.hpp>


int main() {
    Eigen::MatrixXd m = Eigen::MatrixXd::Random(2, 2);
    m(0, 0) = 5;
    m(1, 0) = 6;
    m(0, 1) = 7;
    m(1, 1) = 8;
    Eigen::MatrixXd mt = m.transpose();
    std::cout << "Transpose of m:\n" << mt << std::endl;
    // inverse m
    Eigen::MatrixXd mi = m.inverse();
    std::cout << "Inverse of m:\n" << mi << std::endl;
    // determinant of m
    double det = m.determinant();
    std::cout << "Determinant of m:\n" << det << std::endl;
    // trace of m
    double trace = m.trace();
    std::cout << "Trace of m:\n" << trace << std::endl;
    // norm of m
    double norm = m.norm();
    std::cout << "Norm of m:\n" << norm << std::endl;
    // eigenvalues of m (can be complex for a non-symmetric matrix)
    Eigen::VectorXcd eigenvalues = m.eigenvalues();
    std::cout << "Eigenvalues of m:\n" << eigenvalues.transpose() << std::endl;
    // SVD of m
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
    std::cout << "SVD of m:\n"
              << svd.matrixU() << "\n * \n"
              << svd.singularValues().asDiagonal() << "\n * \n"
              << svd.matrixV().transpose() << std::endl;
    // LU decomposition of m
    Eigen::FullPivLU<Eigen::MatrixXd> lu(m);
    std::cout << "LU decomposition of m:\n" << lu.matrixLU() << std::endl;

    return 0;
}