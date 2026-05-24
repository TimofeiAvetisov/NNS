# Техническое руководство по NNS

## Назначение

`NNS` - header-only библиотека C++23 для построения и обучения полносвязных нейронных сетей.

Основные зависимости:

- Eigen - матрицы и векторные операции.
- proxy - type erasure для слоев, loss-функций, оптимизаторов и learning rate scheduler-ов.

## Подключение

Рекомендуемый include:

```cpp
#include <nns/NNS.hpp>
```

Подключение через CMake:

```cmake
add_executable(app main.cpp)
target_link_libraries(app PRIVATE NNS::nns)
```

## Базовые типы

Файл: `include/nns/core/Types.hpp`.

```cpp
using Scalar = double;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Index = Eigen::Index;
```

`Scalar` задает числовой тип библиотеки. `Matrix`, `Vector`, learning rate, loss value и
параметры оптимизаторов используют `Scalar`.

Формат данных:

- `X`: `input_dim x batch_size`.
- `Y`: `output_dim x batch_size`.
- Один столбец - один объект.
- `DataLoader` использует только полные батчи.

## RandomGenerator

Файл: `include/nns/utils/Random.hpp`.

```cpp
nns::RandomGenerator rng(nns::Seed{42});
```

Инициализация матрицы:

```cpp
nns::Matrix W(128, 784);
rng.init_matrix(W, nns::Distribution::HeNormal, nns::Gain{1.0});
```

Поддерживаемые схемы:

- `Distribution::Normal`
- `Distribution::XavierNormal`
- `Distribution::HeNormal`

Проверки:

- матрица должна быть непустой;
- `gain > 0`.

## Слои

### LinearLayer

Файл: `include/nns/layers/LinearLayers.hpp`.

```cpp
nns::LinearLayer layer(
    nns::In{784},
    nns::Out{128},
    rng,
    nns::Distribution::HeNormal,
    nns::Gain{1.0});
```

Аргументы:
 - `In, Out` : размер слоя;
 - `rng` : объект класса RandomGenerator, передавать для того чтобы зафиксировать seed, по умолчанию создает свой объект класса;
 - `Distribution`: одна из поддерживаемых схем инициализации, значение по умолчанию - `Distribution::Normal`;
 - `Gain` : параметр gain для `RandomGenerator::init_matrix`, по умолчанию равен 1;

Проверки:

- `in > 0`;
- `out > 0`;
- `X.rows() == in`;
- `X.cols() > 0`;
- форма `dY` соответствует выходу слоя и сохраненному batch size.

### ActivationLayer

Файл: `include/nns/layers/ActivationLayers.hpp`.

Создается автоматически при передаче activation-объекта в `NeuralNetwork`.

Встроенные activation-функции:

- `ReLU`
- `Sigmoid`
- `Tanh`

Интерфейс пользовательской activation-функции:

```cpp
struct CustomActivation {
    nns::Scalar forward(nns::Scalar x) const;
    nns::Scalar derivative(nns::Scalar x) const;
};
```

Поддерживаются только скалярные функции активации.

## NeuralNetwork

Файл: `include/nns/network/Network.hpp`.

```cpp
nns::NeuralNetwork net(
    nns::LinearLayer(nns::In{784}, nns::Out{128}, rng, nns::Distribution::HeNormal),
    nns::ReLU(),
    nns::LinearLayer(nns::In{128}, nns::Out{10}, rng, nns::Distribution::XavierNormal));
```
В конструктор можно передать произвольную конфигурацию состоящую из объектов конструируемых в ActivationLayer или LinearLayer.

Методы:

- `Matrix predict(const Matrix& X) const`
- `std::pair<Matrix, std::any> forward(Matrix&& X)`
- `std::pair<Matrix, std::any> backward(Matrix&& dL_dy, const std::any& layers_cache)`
- `std::any update(std::any&& layers_gradients, AnyOptimizer& opt, std::any&& opt_cache)`

Проверки:

- сеть должна содержать минимум один слой;
- cache в `backward` должен быть непустым;
- размер cache должен совпадать с числом слоев;
- размер gradients/cache в `update` должен совпадать с числом слоев.

## Loss-функции

Файлы:

- `include/nns/loss/BuiltinLoss.hpp`
- `include/nns/loss/AnyLossFunction.hpp`

Встроенные loss-функции:

- `MSELoss`
- `MAELoss`
- `HuberLoss`
- `BCELoss`
- `BCEWithLogitsLoss`
- `CrossEntropyLoss`

Общий интерфейс:

```cpp
nns::Scalar loss(const nns::Matrix& y_hat, const nns::Matrix& y);
nns::Matrix gradient(const nns::Matrix& y_hat, const nns::Matrix& y);
```

Type-erased wrapper:

```cpp
nns::AnyLossFunction loss = nns::make_AnyLossFunction<nns::CrossEntropyLoss>();
```

Проверки:

- `y_hat` и `y` должны иметь одинаковую форму;
- входы должны быть непустыми;
- `HuberLoss::delta > 0`;
- `BCELoss::eps` должен быть в диапазоне `(0, 0.5)`;
- `CrossEntropyLoss` принимает logits, softmax применяется внутри.

## Learning rate scheduler

Файлы:

- `include/nns/learningrates/BuiltinLearningRates.hpp`
- `include/nns/learningrates/AnyLearningRate.hpp`

Встроенные scheduler-ы:

- `ConstantLR`
- `TimeDecayLR`

```cpp
nns::ConstantLR lr(nns::LR{0.001});
```

Интерфейс:

```cpp
nns::Scalar get_lr();
void iter_step();
size_t get_iter() const;
```

Проверки:

- learning rate должен быть больше нуля;
- для `TimeDecayLR`: `s0 > 0`, `p >= 0`.

## Оптимизаторы

Файлы:

- `include/nns/optimizer/BuiltinOptimizers.hpp`
- `include/nns/optimizer/AnyOptimizer.hpp`

Встроенные оптимизаторы:

- `SGDOptimizer`
- `AdamOptimizer`

```cpp
nns::AnyOptimizer opt =
    nns::make_AnyOptimizer(nns::AdamOptimizer(nns::ConstantLR(nns::LR{0.001})));
```

Проверки:

- форма параметров должна совпадать с формой градиентов;
- для Adam: `beta1` и `beta2` в диапазоне `[0, 1)`;
- для Adam: `eps > 0`.

## DataLoader

Файл: `include/nns/utils/DataLoader.hpp`.

```cpp
nns::DataLoader loader(
    std::move(X),
    std::move(Y),
    nns::BatchSize{64},
    nns::Shuffle{true});
```

Методы:

- `size_t size() const`
- `size_t batch_size() const`
- `size_t num_batches() const`
- `Batch get_batch(size_t batch_idx) const`
- `void reset_epoch()`
- `void reset_epoch(RandomGenerator& rng)`
- `begin()` / `end()` для range-for

Проверки:

- `X.cols() == Y.cols()`;
- `batch_size > 0`;
- dataset не должен быть пустым;
- `get_batch` принимает только индекс полного батча;
- `reset_epoch()` без RNG допустим только при `Shuffle{false}`.

## CSV loader

Файл: `include/nns/utils/LoadCSV.hpp`.

```cpp
auto [X, Y] = nns::load_csv("data.csv", {0}, ',', true);
```

Аргументы:

- `path`
- `target_cols`
- `delimiter`
- `has_header`

Проверки:

- `target_cols` не должен быть пустым;
- файл должен открываться;
- файл должен содержать данные;
- все строки должны иметь одинаковое число колонок;
- индексы target-колонок должны попадать в диапазон;
- должна существовать минимум одна feature-колонка.

## Trainer

Файл: `include/nns/trainer/Trainer.hpp`.

```cpp
nns::Trainer trainer(net, opt, loss, loader);
```

`Trainer` хранит ссылки:

- `NeuralNetwork&`
- `AnyOptimizer&`
- `AnyLossFunction&`
- `DataLoader&`

Объекты должны жить дольше `Trainer`.

Методы:

- `Scalar fit_epoch(RandomGenerator& rng)`
- `Scalar fit_epoch()`
- `std::vector<Scalar> fit(size_t epochs, RandomGenerator& rng)`
- `std::vector<Scalar> fit(size_t epochs)`
- `void reset_optimizer_state()`

`fit()` вызывает `fit_epoch()` и возвращает средний loss по каждой эпохе.

Проверки:

- `DataLoader` должен содержать минимум один полный батч;
- `fit_epoch()` без RNG допустим только при `Shuffle{false}`.

## Полный пример обучения

```cpp
#include <iostream>

#include <nns/NNS.hpp>

int main() {
    nns::RandomGenerator rng(nns::Seed{42});

    nns::Matrix X(2, 4);
    X << 0.0, 0.0, 1.0, 1.0,
         0.0, 1.0, 0.0, 1.0;

    nns::Matrix Y = nns::Matrix::Zero(2, 4);
    Y(0, 0) = 1.0;
    Y(1, 1) = 1.0;
    Y(1, 2) = 1.0;
    Y(0, 3) = 1.0;

    nns::DataLoader loader(std::move(X), std::move(Y), nns::BatchSize{4}, nns::Shuffle{true});

    nns::NeuralNetwork net(
        nns::LinearLayer(nns::In{2}, nns::Out{8}, rng, nns::Distribution::HeNormal),
        nns::ReLU(),
        nns::LinearLayer(nns::In{8}, nns::Out{2}, rng, nns::Distribution::XavierNormal));

    nns::AnyOptimizer opt =
        nns::make_AnyOptimizer(nns::AdamOptimizer(nns::ConstantLR(nns::LR{0.01})));
    nns::AnyLossFunction loss = nns::make_AnyLossFunction<nns::CrossEntropyLoss>();

    nns::Trainer trainer(net, opt, loss, loader);
    std::vector<nns::Scalar> history = trainer.fit(200, rng);

    nns::Matrix logits = net.predict(loader.get_batch(0).X);
    nns::Matrix probs = nns::CrossEntropyLoss::softmax(logits);

    std::cout << "last loss = " << history.back() << '\n';
    std::cout << probs << '\n';
}
```

## Расширение

### Custom loss

```cpp
struct CustomLoss {
    nns::Scalar loss(const nns::Matrix& y_hat, const nns::Matrix& y) {
        return (y_hat - y).squaredNorm();
    }

    nns::Matrix gradient(const nns::Matrix& y_hat, const nns::Matrix& y) {
        return nns::Scalar{2.0} * (y_hat - y);
    }
};

nns::AnyLossFunction loss = nns::make_AnyLossFunction<CustomLoss>();
```

### Custom optimizer

```cpp
struct CustomOptimizer {
    std::any update_weights(nns::Matrix& param, const nns::Matrix& grad, std::any&& cache) {
        param -= nns::Scalar{0.01} * grad;
        return std::move(cache);
    }

    std::any update_weights(nns::Vector& param, const nns::Vector& grad, std::any&& cache) {
        param -= nns::Scalar{0.01} * grad;
        return std::move(cache);
    }

    void iter_step() {}
};
```

### Custom activation

```cpp
struct LeakyReLU {
    nns::Scalar forward(nns::Scalar x) const {
        return x > nns::Scalar{0.0} ? x : nns::Scalar{0.01} * x;
    }

    nns::Scalar derivative(nns::Scalar x) const {
        return x > nns::Scalar{0.0} ? nns::Scalar{1.0} : nns::Scalar{0.01};
    }
};
```
