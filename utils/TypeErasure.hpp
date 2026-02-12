#pragma once
#include <memory>
#include <utility>
#include <type_traits>

// References: DimaTrushin: https://github.com/DimaTrushin/Library/tree/master/AnyObject
//---------------------------------------------------------------------------
// How To Use
//---------------------------------------------------------------------------
//
// Description
// CAnyMovable provides *type erasure* for movable-only objects of any type,
// allowing you to store and use values with a common interface at runtime.
//
// It lets you define your own interface (a set of virtual methods),
// then automatically generates implementations for any type you assign.
// This approach generalizes the idea of *Run-time Polymorphism*
// discussed by Sean Parent (CppCon talk), in a cleaner and simpler form.
//
// Unlike std::any, you can directly call interface methods on the stored object:
//   x->print();
//   x->size();
//
// The container does not use Small Object Optimization (SBO),
// but move operations are always cheap because storage is via `std::unique_ptr`.
//
// `operator->()` performs *no null check* — you must ensure the object is defined
// by calling `isDefined()` before use.
//
//---------------------------------------------------------------------------
// 1) Define your interface
//---------------------------------------------------------------------------
//
// struct IAny {
//   virtual void print() const = 0;
//   virtual std::size_t size() const = 0;
//   virtual void foo(int x) const = 0;
//   virtual ~IAny() = default;
// };
//
// This describes the common API for all stored objects.
// Any method declared here will be available via `operator->()`.
// The interface no longer needs to be a template (unlike in the legacy version).
// You can declare any number of methods with arbitrary signatures.
//
// We don’t need any template signatures for the methods in IAny,
// because the main point of type erasure is to erase the type (pretty smart)).
//---------------------------------------------------------------------------
// 2) Provide a generic implementation
//---------------------------------------------------------------------------
//
// template<class Base, class TObject>
// class CAnyImpl : public Base {
//   TObject object_;
// public:
//   template<class U>
//   explicit CAnyImpl(U&& obj) : object_(std::forward<U>(obj)) {}
//
//   void print() const override {
//     if constexpr (requires { std::cout << object_; })
//       std::cout << "data = " << object_ << '\n';
//     else
//       std::cout << "[unprintable type]\n";
//   }
//
//   std::size_t size() const override {
//     if constexpr (requires { object_.size(); })
//       return object_.size();
//     else
//       return sizeof(object_);
//   }
//
//   void foo(int x) const override {
//     std::cout << "foo(" << x << ") called for stored object\n";
//   }
// };
//
// Each method implements the interface for a given type `TObject`.
// The `requires` syntax (C++20) allows compile-time adaptation:
// if a method or operator is not available for `TObject`,
// a fallback behavior is used.
//
//---------------------------------------------------------------------------
// 3) Create your type-erased class
//---------------------------------------------------------------------------
//
// using CAny = CAnyMovable<IAny, CAnyImpl>;
//
// This binds the interface and implementation together.
// After this, `CAny` can store *any movable type* and expose
// the interface methods defined in `IAny`.
//
//---------------------------------------------------------------------------
// 4) Example usage
//---------------------------------------------------------------------------
//
// int main() {
//   CAny a = 42;
//   a->print();                    // data = 42
//   std::cout << a->size() << '\n'; // sizeof(int)
//   a->foo(10);
//
//   a.emplace<std::string>("hello");
//   a->print();                    // data = hello
//   std::cout << a->size() << '\n'; // 5
//   a->foo(20);
//
//   a.emplace<std::vector<int>>(3, 7);
//   a->print();                    // [unprintable type]
//   std::cout << a->size() << '\n'; // 3
//   a->foo(30);
// }
//---------------------------------------------------------------------------
// Key Differences from Legacy Version
//---------------------------------------------------------------------------
//
// 1) Interface definition is simpler
//    - OLD: interface had to be a template (e.g., IAny<TBase>).
//    - NEW: interface is a plain struct (e.g., struct IAny {...}).
//    → Cleaner syntax and easier integration with existing APIs.
// To elaborate: the whole idea of type erasure is precisely to remove
// compile-time type dependency, so it’s better that the interface methods
// are not templated.
//
// 2) Internal structure is flatter
//    - OLD: involved multiple layers (IEmpty → IObjectStored → CObjectKeeper → CObjectStored).
//    - NEW: only two layers (IHolder and Holder<T>).
//    → Just clearer
//
//
// 3) More readable semantics
//    - `using CAny = CAnyMovable<IAny, CAnyImpl>;` is explicit and clear.
//    - No template gymnastics or nested template interfaces.
//
//
// 4) Fully compatible with the same usage style
//    - All old usage patterns (`CAny x = ...;`, `x->print();`, `x.emplace<T>()`) work identically.
//    - Move-only semantics are preserved.
//
// Personally, I find it much clearer to see explicitly what the code does,
// without going through all that nested template mess.
//
//---------------------------------------------------------------------------
namespace nns {
    template <class Interface, template <class, class> class Implementation>
    class CAnyMovable {
        struct IHolder : Interface {
            virtual std::unique_ptr<IHolder> move_clone() noexcept = 0;
            ~IHolder() override = default;
        };

        template <class T>
        struct Holder final : Implementation<IHolder, T> {
            using Base = Implementation<IHolder, T>;
            using Base::Base;

            std::unique_ptr<IHolder> move_clone() noexcept override {
                return std::make_unique<Holder<T>>(std::move(*this));
            }
        };

        std::unique_ptr<IHolder> ptr_;

    public:
        CAnyMovable() = default;
        CAnyMovable(const CAnyMovable&) = delete;
        CAnyMovable& operator=(const CAnyMovable&) = delete;
        CAnyMovable(CAnyMovable&&) noexcept = default;
        CAnyMovable& operator=(CAnyMovable&&) noexcept = default;

        template <class T>
        CAnyMovable(T&& v)
            : ptr_(std::make_unique<Holder<std::remove_cvref_t<T>>>(std::forward<T>(v))) {
        }

        template <class T, class... Args>
        void emplace(Args&&... args) {
            ptr_ = std::make_unique<Holder<T>>(T(std::forward<Args>(args)...));
        }

        bool isDefined() const noexcept {
            return static_cast<bool>(ptr_);
        }
        void clear() noexcept {
            ptr_.reset();
        }

        Interface* operator->() noexcept {
            return ptr_.get();
        }
        const Interface* operator->() const noexcept {
            return ptr_.get();
        }
    };
}