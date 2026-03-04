#include <proxy.h>
#include <iostream>

PRO_DEF_MEM_DISPATCH(MemDraw, draw);

struct Drawable : pro::facade_builder
    ::add_convention<MemDraw, void()>
    ::build {};

struct Circle {
    void draw() {
        std::cout << "Drawing a circle" << std::endl;
    }
};

struct Square {
    void draw() {
        std::cout << "Drawing a square" << std::endl;
    }
};

using AnyDrawable = pro::proxy<Drawable>;

template<typename T, typename... Args>
AnyDrawable make_drawable(Args&&... args) {
    return pro::make_proxy<Drawable>(T{std::forward<Args>(args)...});
}

int main() {
    AnyDrawable p = make_drawable<Circle>();
    p->draw();
    AnyDrawable q = make_drawable<Square>();
    q->draw();
}