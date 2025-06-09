#pragma once

namespace sim{

template<typename T>
struct Matrix3x3 {
    T data[9];
    
    __host__ __device__
    Matrix3x3() {
        for(int i = 0; i < 9; i++) data[i] = T(0);
    }
    
    // 単位行列コンストラクタ
    __host__ __device__
    static Matrix3x3 identity() {
        Matrix3x3 result;
        result(0,0) = result(1,1) = result(2,2) = T(1);
        return result;
    }
    
    __host__ __device__
    T& operator()(int i, int j) {
        return data[i * 3 + j];
    }
    
    __host__ __device__
    const T& operator()(int i, int j) const {
        return data[i * 3 + j];
    }
    
    // Matrix product
    __host__ __device__
    Matrix3x3 operator*(const Matrix3x3& other) const {
        Matrix3x3<T> result;
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                for(int k = 0; k < 3; k++) {
                    result(i,j) += (*this)(i,k) * other(k,j);
                }
            }
        }
        return result;
    }
    
    // Matrix Vector multiply
    __host__ __device__
    Vector3<T> operator*(const Vector3<T>& vec) const {
        Vector3<T> result;
        for(int i = 0; i < 3; i++) {
            for(int k = 0; k < 3; k++) {
                result(i) += (*this)(i,k) * vec(k);  // vec(k) でアクセス
            }
        }
        return result;
    }
    
    // Matrix add
    __host__ __device__
    Matrix3x3<T> operator+(const Matrix3x3& other) const {
        Matrix3x3<T> result;
        for(int i = 0; i < 9; i++) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    // Matrix substract
    __host__ __device__
    Matrix3x3<T> operator-(const Matrix3x3& other) const {
        Matrix3x3<T> result;
        for(int i = 0; i < 9; i++) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    // scalar * matrix
    __host__ __device__
    friend Matrix3x3<T> operator*(T scalar, const Matrix3x3& m) {
        Matrix3x3 result;
        for (int i = 0; i < 9; i++) result.data[i] = scalar * m.data[i];
        return result;
    }

    // matrix * scalar
    __host__ __device__
    Matrix3x3<T> operator*(T scalar) const {
        Matrix3x3<T> result;
        for (int i = 0; i < 9; i++) result.data[i] = data[i] * scalar;
        return result;
    }
    
    // transpose
    __host__ __device__
    Matrix3x3<T> transpose() const {
        Matrix3x3<T> result;
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                result(i,j) = (*this)(j,i);
            }
        }
        return result;
    }
    
    // deteminant
    __host__ __device__
    T determinant() const {
        return (*this)(0,0) * ((*this)(1,1) * (*this)(2,2) - (*this)(1,2) * (*this)(2,1))
             - (*this)(0,1) * ((*this)(1,0) * (*this)(2,2) - (*this)(1,2) * (*this)(2,0))
             + (*this)(0,2) * ((*this)(1,0) * (*this)(2,1) - (*this)(1,1) * (*this)(2,0));
    }

    __host__ __device__ 
    Matrix3x3<T> inverse() const {
        T det = determinant();
        if (fabs(det) < 1e-8) {
            return Matrix3x3::identity();
        }

        Matrix3x3<T> inv;

        inv(0,0) =  (*this)(1,1)*(*this)(2,2) - (*this)(1,2)*(*this)(2,1);
        inv(0,1) = -(*this)(0,1)*(*this)(2,2) + (*this)(0,2)*(*this)(2,1);
        inv(0,2) =  (*this)(0,1)*(*this)(1,2) - (*this)(0,2)*(*this)(1,1);

        inv(1,0) = -(*this)(1,0)*(*this)(2,2) + (*this)(1,2)*(*this)(2,0);
        inv(1,1) =  (*this)(0,0)*(*this)(2,2) - (*this)(0,2)*(*this)(2,0);
        inv(1,2) = -(*this)(0,0)*(*this)(1,2) + (*this)(0,2)*(*this)(1,0);

        inv(2,0) =  (*this)(1,0)*(*this)(2,1) - (*this)(1,1)*(*this)(2,0);
        inv(2,1) = -(*this)(0,0)*(*this)(2,1) + (*this)(0,1)*(*this)(2,0);
        inv(2,2) =  (*this)(0,0)*(*this)(1,1) - (*this)(0,1)*(*this)(1,0);

        T invDet = 1.0f / det;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                inv(r, c) *= invDet;

        return inv;
    }
};

}