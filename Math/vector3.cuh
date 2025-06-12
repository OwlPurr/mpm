#pragma once

namespace sim{

template<typename T>
struct Vector3 {
    T x, y, z;
    
    __host__ __device__
    Vector3() : x(T(0)), y(T(0)), z(T(0)) {}
    
    __host__ __device__
    Vector3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
    
    __host__ __device__
    T& operator()(int i) {
        return (&x)[i];  // you can access like this because the fields (x,y,z) coordinate continuously
    }
    
    __host__ __device__
    const T& operator()(int i) const {
        return (&x)[i];
    }
    
    // inner product
    __host__ __device__
    T operator*(const Vector3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    // vector add
    __host__ __device__
    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }
    
    // vector substract
    __host__ __device__
    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }
    
    // scalar product
    __host__ __device__
    Vector3 operator*(T scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }

    __host__ __device__
    friend Vector3 operator*(T scalar, const Vector3& v) {
        return Vector3(v.x * scalar, v.y * scalar, v.z * scalar);
    }

    __host__ __device__
    Vector3& operator+=(const Vector3& other) {
        x += other.x; y += other.y; z += other.z;
        return *this;
    }

    __host__ __device__
    Vector3& operator-=(const Vector3& other) {
        x -= other.x; y -= other.y; z -= other.z;
        return *this;
    }

    __host__ __device__
    Vector3& operator*=(T scalar) {
        x *= scalar; y *= scalar; z *= scalar;
        return *this;
    }

    __host__ __device__
    Vector3 operator/(T scalar) const {
        assert(scalar != T(0));  // ホスト側のみ、またはデバッグ用
        return Vector3(x / scalar, y / scalar, z / scalar);
    }

    __host__ __device__
    Vector3 operator-() const {
        return Vector3(-x, -y, -z);
    }

    __host__ __device__
    bool operator==(const Vector3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    __host__ __device__
    bool operator!=(const Vector3& other) const {
        return !(*this == other);
    }

    __host__ __device__
    bool is_zero(T eps = T(1e-6)) const {
        return fabs(x) < eps && fabs(y) < eps && fabs(z) < eps;
    }

    __host__ __device__
    Vector3& operator=(const Vector3& other) {
        if (this == &other) return *this; // 自己代入チェック
        x = other.x; y = other.y; z = other.z;
        return *this;
    }
};

}