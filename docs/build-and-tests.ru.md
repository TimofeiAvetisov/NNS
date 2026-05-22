## Требования

- CMake 3.21 или новее для работы с `CMakePresets.json`.
- Ninja.
- Компилятор C++23, доступный для CMake. Пресеты не фиксируют конкретный компилятор.
- `clang-format` для проверки форматирования.
- `just` для коротких команд.
- Доступ к сети при первой конфигурации: Eigen, proxy и Catch2 загружаются через
  `FetchContent`.

## CMakePresets

Доступны два основных пресета:

- `debug` - отладочная сборка в `build-debug`.
- `release` - релизная сборка в `build-release`.

Оба пресета используют Ninja, включают `NNS_BUILD_TESTING=ON`, экспортируют
`compile_commands.json` и по умолчанию не собирают пример из `main.cpp`. Пример можно включить
вручную:

```powershell
cmake --preset debug -DNNS_BUILD_EXAMPLE=ON
cmake --build --preset debug
```

## Команды just

Основной сценарий разработки:

```powershell
just configure
just build
just test
```

Полная проверка:

```powershell
just check
```

Эта команда собирает проект, запускает тесты и проверяет форматирование.

Для релизной конфигурации можно передать пресет:

```powershell
just configure release
just build release
just test release
```

## Тесты

Тесты находятся в папке `tests`.

`tests/CMakeLists.txt` создает исполняемый файл `nns_tests`, подключает библиотеку `NNS::nns`
и Catch2. Тесты регистрируются через `catch_discover_tests`, поэтому CTest видит отдельные
Catch2 test cases.

Запуск напрямую через CMake/CTest:

```powershell
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

## Форматирование

Форматирование использует `.clang-format`.

Проверить формат:

```powershell
just format-check
```

Исправить формат:

```powershell
just format
```

Форматируются заголовки из `include` и тесты из `tests`.
