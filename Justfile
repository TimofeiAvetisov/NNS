default_preset := "debug"

default:
    just --list

configure preset=default_preset:
    cmake --preset "{{preset}}"

build preset=default_preset:
    cmake --build --preset "{{preset}}"

test preset=default_preset:
    ctest --preset "{{preset}}"

format:
    cmake --build --preset format

format-check:
    cmake --build --preset format-check

check preset=default_preset:
    cmake --build --preset "{{preset}}"
    ctest --preset "{{preset}}"
    cmake --build --preset format-check
