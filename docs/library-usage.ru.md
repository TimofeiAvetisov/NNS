# Использование библиотеки NNS

`NNS` - header-only библиотека для простых нейронных сетей на C++23. Она использует Eigen для
матриц и `proxy` для type erasure слоев, функций потерь, оптимизаторов и learning rate
scheduler-ов.

## Подключение

В CMake подключайте цель `NNS::nns`:

```cmake
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE NNS::nns)
```

Минимальный набор include зависит от того, какие части библиотеки нужны:

```cpp
#include <nns/activation/BuiltinActivations.hpp>
#include <nns/layers/LinearLayers.hpp>
#include <nns/learningrates/BuiltinLearningRates.hpp>
#include <nns/loss/AnyLossFunction.hpp>
#include <nns/network/Network.hpp>
#include <nns/optimizer/AnyOptimizer.hpp>
#include <nns/trainer/Trainer.hpp>
#include <nns/utils/DataLoader.hpp>
#include <nns/utils/Random.hpp>
```

Все публичные сущности находятся в namespace `nns`.

## Базовые типы

Основные типы объявлены в `nns/core/Types.hpp`:

- `Scalar` - alias для числового типа библиотеки. Сейчас это `double`; если нужно перейти на
  `float`, поменяйте этот alias в одном месте.
- `Matrix` - alias для `Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>`.
- `Vector` - alias для `Eigen::Matrix<Scalar, Eigen::Dynamic, 1>`.
- `Index` - alias для `Eigen::Index`.
- `StrongType<Tag, T>` - простой strong type для параметров вроде `LR`, `Gain`, `In`, `Out`.

`Matrix` и `Vector` используют `Scalar` как тип элемента. Матрицы в библиотеке обычно хранят
данные в формате:

- `X`: `input_dim x batch_size`.
- `Y`: `output_dim x batch_size`.
- Каждый столбец - один объект батча.

Пример:

```cpp
nns::Matrix X(2, 3);
X << 1.0, 2.0, 3.0,
     4.0, 5.0, 6.0;
```

Здесь `X` содержит 3 объекта, каждый объект имеет 2 признака.

## Генератор случайных чисел

`RandomGenerator` используется для инициализации весов и перемешивания данных:

```cpp
nns::RandomGenerator rng(nns::Seed{42});
```

Доступные распределения для весов:

- `Distribution::Normal` - нормальное распределение со стандартным отклонением `Gain`.
- `Distribution::XavierNormal` - Xavier normal.
- `Distribution::HeNormal` - He normal.

Пример ручной инициализации матрицы:

```cpp
nns::Matrix weights(4, 2);
rng.init_matrix(weights, nns::Distribution::HeNormal, nns::Gain{1.0});
```

## Слои

### LinearLayer

Линейный слой задается входной и выходной размерностью:

```cpp
nns::LinearLayer layer(
    nns::In{784},
    nns::Out{128},
    rng,
    nns::Distribution::XavierNormal,
    nns::Gain{1.0});
```

Слой хранит матрицу весов `A` размера `out x in` и bias `b` размера `out x 1`.

### ActivationLayer

Обычно его не нужно создавать вручную. В `NeuralNetwork` можно передавать встроенные activation
объекты напрямую:

```cpp
nns::ReLU()
nns::Sigmoid()
nns::Tanh()
```

Каждая activation-функция должна иметь методы:

```cpp
nns::Scalar forward(nns::Scalar x) const;
nns::Scalar derivative(nns::Scalar x) const;
```

Пример своей activation-функции:

```cpp
struct LeakyReLU {
    nns::Scalar forward(nns::Scalar x) const {
        return x > 0.0 ? x : 0.01 * x;
    }

    nns::Scalar derivative(nns::Scalar x) const {
        return x > 0.0 ? 1.0 : 0.01;
    }
};
```

После этого ее можно использовать как обычный слой:

```cpp
nns::NeuralNetwork net(
    nns::LinearLayer(nns::In{2}, nns::Out{4}, rng),
    LeakyReLU(),
    nns::LinearLayer(nns::In{4}, nns::Out{1}, rng));
```

## Нейронная сеть

`NeuralNetwork` принимает список слоев в конструкторе:

```cpp
nns::NeuralNetwork net(
    nns::LinearLayer(nns::In{784}, nns::Out{128}, rng, nns::Distribution::HeNormal),
    nns::ReLU(),
    nns::LinearLayer(nns::In{128}, nns::Out{10}, rng, nns::Distribution::XavierNormal));
```

Основные методы:

- `predict(const Matrix& X) const` - прямой проход без сохранения cache для backward.
- `forward(Matrix&& X)` - прямой проход с cache для обучения.
- `backward(Matrix&& dL_dy, const std::any& cache)` - обратный проход.
- `update(std::any&& gradients, AnyOptimizer& opt, std::any&& opt_cache)` - обновление параметров.

Для обычного обучения вручную эти методы лучше не вызывать напрямую: используйте `Trainer`.
Для inference достаточно `predict`.

```cpp
nns::Matrix logits = net.predict(X);
```

## Функции потерь

Встроенные loss-функции находятся в `nns/loss/BuiltinLoss.hpp`:

- `MSELoss` - mean squared error.
- `MAELoss` - mean absolute error.
- `HuberLoss` - Huber loss, поле `delta` по умолчанию равно `1.0`.
- `BCELoss` - binary cross entropy для вероятностей.
- `BCEWithLogitsLoss` - binary cross entropy для логитов, со встроенной sigmoid-логикой.
- `CrossEntropyLoss` - softmax + cross entropy для многоклассовой классификации.

Каждая loss-функция имеет интерфейс:

```cpp
nns::Scalar loss(const nns::Matrix& y_hat, const nns::Matrix& y) const;
nns::Matrix gradient(const nns::Matrix& y_hat, const nns::Matrix& y) const;
```

Для `Trainer` loss нужно завернуть в `AnyLossFunction`:

```cpp
nns::AnyLossFunction loss = nns::make_AnyLossFunction<nns::CrossEntropyLoss>();
```

Для ручного использования можно создать loss напрямую:

```cpp
nns::MSELoss mse;
nns::Scalar value = mse.loss(y_hat, y);
nns::Matrix grad = mse.gradient(y_hat, y);
```

## Learning rate scheduler

Встроенные scheduler-ы:

- `ConstantLR` - постоянный learning rate.
- `TimeDecayLR` - убывающий learning rate.

Learning rate задается strong type `LR`:

```cpp
nns::ConstantLR lr(nns::LR{0.001});
```

Scheduler имеет методы:

- `get_lr()` - текущий learning rate.
- `iter_step()` - перейти к следующей итерации.
- `get_iter()` - номер текущей итерации.

## Оптимизаторы

Встроенные оптимизаторы:

- `SGDOptimizer`.
- `AdamOptimizer`.

Пример SGD:

```cpp
nns::AnyOptimizer opt = nns::make_AnyOptimizer(
    nns::SGDOptimizer(nns::ConstantLR(nns::LR{0.01})));
```

Пример Adam:

```cpp
nns::AnyOptimizer opt = nns::make_AnyOptimizer(
    nns::AdamOptimizer(
        nns::ConstantLR(nns::LR{0.001}),
        nns::Beta1{0.9},
        nns::Beta2{0.999},
        nns::Eps{1e-8}));
```

Оптимизатор получает параметры и градиенты от слоев через `NeuralNetwork::update`.
При использовании `Trainer` это происходит автоматически.

## DataLoader

`DataLoader` хранит `X`, `Y`, размер батча и порядок индексов:

```cpp
nns::DataLoader loader(
    std::move(X),
    std::move(Y),
    nns::BatchSize{32},
    nns::Shuffle{true});
```

Требования:

- `X.cols() == Y.cols()`.
- `batch_size > 0`.

Полезные методы:

- `size()` - число объектов.
- `batch_size()` - размер батча.
- `num_batches()` - число полных батчей.
- `get_batch(batch_idx)` - получить батч.
- `reset_epoch(rng)` - начать новую эпоху и перемешать индексы, если `Shuffle{true}`.
- `reset_epoch()` - начать новую эпоху без RNG; подходит только если `Shuffle{false}`.

`DataLoader` поддерживает range-for:

```cpp
for (auto [bX, bY] : loader) {
    // bX: input_dim x batch_size
    // bY: output_dim x batch_size
}
```

Важно: сейчас `num_batches()` использует целочисленное деление, поэтому неполный последний батч
не попадает в итерацию.

## CSV loader

Функция `load_csv` читает CSV и возвращает пару `{X, Y}`:

```cpp
auto [X, Y] = nns::load_csv("data.csv", {0}, ',', true);
```

Аргументы:

- `path` - путь к файлу.
- `target_cols` - номера колонок, которые нужно положить в `Y`.
- `delimiter` - разделитель, по умолчанию `','`.
- `has_header` - пропустить первую строку, если есть заголовок.

Формат результата:

- Колонки из `target_cols` становятся строками `Y`.
- Остальные колонки становятся строками `X`.
- Каждая строка CSV становится одним столбцом в `X` и `Y`.

Пример CSV:

```text
label,x1,x2
1,0.5,0.2
0,0.1,0.9
```

Вызов:

```cpp
auto [X, Y] = nns::load_csv("data.csv", {0}, ',', true);
```

Результат:

- `X` имеет размер `2 x 2`.
- `Y` имеет размер `1 x 2`.

## Trainer

`Trainer` связывает сеть, оптимизатор, loss-функцию и `DataLoader`. Он хранит
`AnyLossFunction&`, поэтому loss должен жить дольше `Trainer`. Это позволяет использовать
stateful loss-функции и менять состояние объекта loss снаружи:

```cpp
nns::Trainer trainer(net, opt, loss, loader);
```

Методы:

- `fit_epoch(rng)` - обучить одну эпоху с перемешиванием через `RandomGenerator`, вернуть средний
  loss за эпоху.
- `fit_epoch()` - обучить одну эпоху без RNG, подходит для `DataLoader` с `Shuffle{false}`.
- `fit(epoches, rng)` - обучение с перемешиванием через `RandomGenerator`.
- `fit(epoches)` - обучение без RNG, подходит для `DataLoader` с `Shuffle{false}`.

`fit` внутри вызывает `fit_epoch` и возвращает `std::vector<Scalar>` с loss по эпохам:

```cpp
std::vector<nns::Scalar> history = trainer.fit(10, rng);
```

## Полный пример регрессии

```cpp
#include <iostream>

#include <nns/activation/BuiltinActivations.hpp>
#include <nns/layers/LinearLayers.hpp>
#include <nns/learningrates/BuiltinLearningRates.hpp>
#include <nns/loss/AnyLossFunction.hpp>
#include <nns/network/Network.hpp>
#include <nns/optimizer/AnyOptimizer.hpp>
#include <nns/trainer/Trainer.hpp>
#include <nns/utils/DataLoader.hpp>
#include <nns/utils/Random.hpp>

int main() {
    nns::RandomGenerator rng(nns::Seed{42});

    nns::Matrix X(1, 4);
    X << 0.0, 1.0, 2.0, 3.0;

    nns::Matrix Y(1, 4);
    Y << 0.0, 2.0, 4.0, 6.0;

    nns::DataLoader loader(std::move(X), std::move(Y), nns::BatchSize{2}, nns::Shuffle{false});

    nns::NeuralNetwork net(
        nns::LinearLayer(nns::In{1}, nns::Out{4}, rng, nns::Distribution::XavierNormal),
        nns::Tanh(),
        nns::LinearLayer(nns::In{4}, nns::Out{1}, rng, nns::Distribution::XavierNormal));

    nns::AnyOptimizer opt = nns::make_AnyOptimizer(
        nns::SGDOptimizer(nns::ConstantLR(nns::LR{0.01})));
    nns::AnyLossFunction loss = nns::make_AnyLossFunction<nns::MSELoss>();

    nns::Trainer trainer(net, opt, loss, loader);
    std::vector<nns::Scalar> history = trainer.fit(100);

    std::cout << "last loss = " << history.back() << '\n';
}
```

## Полный пример классификации

```cpp
#include <iostream>

#include <nns/activation/BuiltinActivations.hpp>
#include <nns/layers/LinearLayers.hpp>
#include <nns/learningrates/BuiltinLearningRates.hpp>
#include <nns/loss/AnyLossFunction.hpp>
#include <nns/network/Network.hpp>
#include <nns/optimizer/AnyOptimizer.hpp>
#include <nns/trainer/Trainer.hpp>
#include <nns/utils/DataLoader.hpp>
#include <nns/utils/Random.hpp>

int main() {
    nns::RandomGenerator rng(nns::Seed{7});

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

    nns::AnyOptimizer opt = nns::make_AnyOptimizer(
        nns::AdamOptimizer(nns::ConstantLR(nns::LR{0.01})));
    nns::AnyLossFunction loss = nns::make_AnyLossFunction<nns::CrossEntropyLoss>();

    nns::Trainer trainer(net, opt, loss, loader);
    trainer.fit(200, rng);

    nns::Matrix logits = net.predict(loader.get_batch(0).X);
    nns::Matrix probs = nns::CrossEntropyLoss::softmax(logits);
    std::cout << probs << '\n';
}
```

## Расширение библиотеки

### Своя loss-функция

Loss-функция должна иметь методы `loss` и `gradient`:

```cpp
struct CustomLoss {
    nns::Scalar loss(const nns::Matrix& y_hat, const nns::Matrix& y) const {
        return (y_hat - y).squaredNorm();
    }

    nns::Matrix gradient(const nns::Matrix& y_hat, const nns::Matrix& y) const {
        return 2.0 * (y_hat - y);
    }
};

nns::AnyLossFunction loss = nns::make_AnyLossFunction<CustomLoss>();
```

### Свой optimizer

Optimizer должен поддерживать обновление `Matrix`, обновление `Vector` и шаг итерации:

```cpp
struct CustomOptimizer {
    std::any update_weights(nns::Matrix& param, const nns::Matrix& grad, std::any&& cache) {
        param -= 0.01 * grad;
        return std::move(cache);
    }

    std::any update_weights(nns::Vector& param, const nns::Vector& grad, std::any&& cache) {
        param -= 0.01 * grad;
        return std::move(cache);
    }

    void iter_step() {}
};

nns::AnyOptimizer opt = nns::make_AnyOptimizer(CustomOptimizer{});
```

### Свой learning rate scheduler

Scheduler должен иметь методы `get_lr`, `iter_step`, `get_iter`:

```cpp
struct StepLR {
    nns::Scalar get_lr() {
        return iter < 100 ? 0.1 : 0.01;
    }

    void iter_step() {
        ++iter;
    }

    size_t get_iter() const {
        return iter;
    }

    size_t iter = 1;
};

nns::AnyOptimizer opt = nns::make_AnyOptimizer(nns::SGDOptimizer(StepLR{}));
```

## Практические ограничения

- Библиотека ожидает Eigen-матрицы с элементами типа `nns::Scalar`.
- Данные должны быть представлены столбцами, а не строками.
- `DataLoader` сейчас итерируется только по полным батчам.
- Для `CrossEntropyLoss` передавайте logits, а не softmax-вероятности: softmax уже встроен в loss.
- Для `BCELoss` передавайте вероятности после sigmoid.
- Для `BCEWithLogitsLoss` передавайте logits.
- `Trainer::fit(epoches, rng)` требует `DataLoader`, который можно перемешивать.
- `Trainer::fit(epoches)` используйте с `Shuffle{false}`.
