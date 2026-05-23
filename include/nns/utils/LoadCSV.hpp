#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nns/core/Types.hpp>

namespace nns {
inline std::pair<Matrix, Matrix> load_csv(const std::string& path,
                                          const std::unordered_set<size_t>& target_cols,
                                          char delimiter = ',', bool has_header = false) {
    if (target_cols.empty()) {
        throw std::invalid_argument("load_csv: target_cols must not be empty");
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("load_csv: cannot open " + path);
    }

    std::string line;
    if (has_header) {
        std::getline(file, line);
    }

    std::vector<std::vector<Scalar>> x_rows, y_rows;
    size_t expected_cols = 0;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream ss(line);
        std::string token;
        std::vector<Scalar> x_row, y_row;
        size_t col = 0;
        while (std::getline(ss, token, delimiter)) {
            Scalar val = static_cast<Scalar>(std::stod(token));
            if (target_cols.count(col)) {
                y_row.push_back(val);
            } else {
                x_row.push_back(val);
            }
            ++col;
        }

        if (col == 0) {
            continue;
        }
        if (expected_cols == 0) {
            expected_cols = col;
            for (size_t target_col : target_cols) {
                if (target_col >= expected_cols) {
                    throw std::invalid_argument("load_csv: target column index is out of range");
                }
            }
            if (target_cols.size() == expected_cols) {
                throw std::invalid_argument("load_csv: at least one feature column is required");
            }
        } else if (col != expected_cols) {
            throw std::runtime_error("load_csv: inconsistent number of columns");
        }

        x_rows.push_back(std::move(x_row));
        y_rows.push_back(std::move(y_row));
    }

    if (x_rows.empty()) {
        throw std::runtime_error("load_csv: no data in " + path);
    }

    const auto n = static_cast<Index>(x_rows.size());
    const auto x_dim = static_cast<Index>(x_rows[0].size());
    const auto y_dim = static_cast<Index>(y_rows[0].size());

    Matrix X(x_dim, n);
    Matrix Y(y_dim, n);
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < x_dim; ++j)
            X(j, i) = x_rows[i][j];
        for (Index j = 0; j < y_dim; ++j)
            Y(j, i) = y_rows[i][j];
    }

    return {std::move(X), std::move(Y)};
}
}  // namespace nns
