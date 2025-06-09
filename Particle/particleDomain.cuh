#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <math.h>

#include "../Math/vector3.cuh"
#include "../Math/matrix3x3.cuh"

namespace sim{

template <typename T>
std::ostream& operator<<(std::ostream& os, const Vector3<T>& p) {
    return os << "(" << p.x << ", " << p.y << ", " << p.z << ")";
}

template<typename T>
struct NodesData {
    using T3 = Vector3<T>;
    T3* pos;
    T3* vel;
    T* mass;
    T* volume;
    size_t num;
};

// Base class of Particles and Cells, SoA structure

template <typename T>
class Nodes {
protected:
    using Vec = thrust::host_vector<T>;
    using DVec = thrust::device_vector<T>;
    using T3 = Vector3<T>;
    using Vec3 = thrust::host_vector<T3>;
    using DVec3 = thrust::device_vector<T3>;
    using T3x3 = Matrix3x3<T>;
    using Mat3x3 = thrust::host_vector<T3x3>;
    using DMat3x3 = thrust::device_vector<T3x3>;
    

public:
    size_t num_;                     // the number of Nodes
    Vec h_mass_, h_volume_;
    Vec3 h_pos_, h_vel_;
    DVec d_mass_, d_volume_;
    DVec3 d_pos_, d_vel_;

    enum class CopyDir { HostToDevice, DeviceToHost };

    Nodes(size_t num, const Vec3& position, const Vec3& velocity, const Vec& mass, const Vec& volume)
        : num_(num), h_pos_(position), h_vel_(velocity), h_mass_(mass), h_volume_(volume) {
        #if defined(__cpp_lib_source_location) && !defined(__CUDA_ARCH__)
            #include <source_location>
            std::source_location loc = std::source_location::current();
            std::cout << "[Constructed] Nodes at " << loc.file_name() << ":" << loc.line() << "\n";
        #else
            std::cout << "[Constructed] Nodes\n";
        #endif
    }

    virtual ~Nodes() = default;

    virtual void printInfo(std::ostream& os = std::cout) const noexcept {
        os << "[Nodes] Count = " << num_ << " | Structure: SoA | Dimension: 3D\n";
        for(int i=1000; i<1010; i++){
            os << d_pos_[i] <<"\n";
        }
    }

    void sync(CopyDir dir) {
        switch (dir) {
            case CopyDir::HostToDevice: upload(); break;
            case CopyDir::DeviceToHost: download(); break;
            default: throw std::invalid_argument("Unknown copy direction");
        }
    }

    NodesData<T> getGPUData(){
        return NodesData<T>{
            thrust::raw_pointer_cast(this->d_pos_.data()),
            thrust::raw_pointer_cast(this->d_vel_.data()),
            thrust::raw_pointer_cast(this->d_mass_.data()),
            thrust::raw_pointer_cast(this->d_volume_.data()),
            this->num_
        };
    }

protected:
    // Default implementation
    virtual void upload() {
        d_pos_ = h_pos_;
        d_vel_ = h_vel_;
        d_mass_ = h_mass_;
        d_volume_ = h_volume_;
    }

    virtual void download() {
        h_pos_ = d_pos_;
        h_vel_ = d_vel_;
        h_mass_ = d_mass_;
        h_volume_ = d_volume_;
    }
};

template<typename T>
struct ParticlesData{
    using T3 = Vector3<T>;
    T3* pos;
    T3* vel;
    T* mass;
    T* volume_0;
    size_t num;
    using T3x3 = Matrix3x3<T>;
    T3x3* C;
    T3x3* F;
    T3x3* eq_16_term_0;
};

// Particle class represents simulation Lagrangian points

template <typename T>
class Particles : public Nodes<T> {
    using Base = Nodes<T>;
    using typename Base::Vec;
    using typename Base::Vec3;
    using typename Base::Mat3x3;
    using typename Base::DVec;
    using typename Base::DVec3;
    using typename Base::DMat3x3;
    Mat3x3 h_C_;
    Mat3x3 h_F_;
    Mat3x3 h_eq_16_term_0_;
    DMat3x3 d_C_;
    DMat3x3 d_F_;
    DMat3x3 d_eq_16_term_0_;

public:
    Particles(size_t num, const Vec3& pos,
              const Vec3& vel, const Vec& mass, const Vec& initVolume, const Mat3x3&  C, const Mat3x3& F, const Mat3x3& eq_16_term_0)
        : Base(num, pos, vel, mass, initVolume), h_C_(C), h_F_(F), h_eq_16_term_0_(eq_16_term_0) {}

    void resort(thrust::device_vector<int>& d_cellids){
        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(
            Base::d_pos_.begin(), Base::d_vel_.begin(), Base::d_mass_.begin(), Base::d_volume_.begin(), d_C_.begin(), d_F_.begin(), d_eq_16_term_0_.begin()));
        thrust::sort_by_key(d_cellids.begin(), d_cellids.end(), zip_begin);
    }

    ParticlesData<T> getGPUData() {
        return ParticlesData<T>{
            thrust::raw_pointer_cast(this->d_pos_.data()),
            thrust::raw_pointer_cast(this->d_vel_.data()),
            thrust::raw_pointer_cast(this->d_mass_.data()),
            thrust::raw_pointer_cast(this->d_volume_.data()),
            this->num_,
            thrust::raw_pointer_cast(this->d_C_.data()),
            thrust::raw_pointer_cast(this->d_F_.data()),
            thrust::raw_pointer_cast(this->d_eq_16_term_0_.data())
        };
    }

    void printInfo(std::ostream& os = std::cout) const noexcept override{
        os << "[Particles] Count = " << Base::num_ << " | Structure: SoA | Dimension: 3D\n";

        os << Base::d_vel_[0] << "\n";
        os << Base::d_pos_[0] << "\n";
    }

    void particleIncrease(size_t num, const Vec3& pos, const Vec3& vel, const Vec& mass, const Vec& initVolume, const Mat3x3&  C, const Mat3x3& F, const Mat3x3& eq_16_term_0){
        download();
        Base::num_ += num;
        Base::h_pos_.insert(Base::h_pos_.begin(), pos.begin(), pos.end());
        Base::h_vel_.insert(Base::h_vel_.begin(), vel.begin(), vel.end());
        Base::h_mass_.insert(Base::h_mass_.begin(), mass.begin(), mass.end());
        Base::h_volume_.insert(Base::h_volume_.begin(), initVolume.begin(), initVolume.end());
        h_C_.insert(h_C_.begin(), C.begin(), C.end());
        h_F_.insert(h_F_.begin(), F.begin(), F.end());
        h_eq_16_term_0_.insert(h_eq_16_term_0_.begin(), eq_16_term_0.begin(), eq_16_term_0.end());
        upload();
    }

protected:
    void upload() override {
        Base::upload();
        d_C_ = h_C_;
        d_F_ = h_F_;
        d_eq_16_term_0_ = h_eq_16_term_0_;
    }

    void download() override {
        Base::download();
        h_C_ = d_C_;
        h_F_ = d_F_;
        h_eq_16_term_0_ = d_eq_16_term_0_;
    }

};

// Cell class represents simulation Eulerian points

template <typename T>
class Cells : public Nodes<T> {
    using Base = Nodes<T>;
    using typename Base::Vec;
    using typename Base::Vec3;
    using typename Base::DVec;
    using typename Base::DVec3;
    using typename Base::T3;

public:
    Cells(size_t num, const Vec3& pos,
          const Vec3& vel, const Vec& mass, const Vec& volume)
        : Base(num, pos, vel, mass, volume) {}

    void printInfo(std::ostream& os = std::cout) const noexcept override{
        os << "[Cells] Count = " << Base::num_ << " | Structure: SoA | Dimension: 3D\n";


        float sum = thrust::reduce(Base::d_mass_.begin(), Base::d_mass_.begin()+(size_t)(Base::num_/2), 0.0f);
        os << sum << "\n";
        /*
        os << Base::d_vel_[32*32*16+32*16+16] << "\n";

        for(int i=0; i<Base::num_; i++){
            if(Base::d_mass_[i]!=0.f) os << Base::d_mass_[i]/Base::d_volume_[i] << std::endl;
        }
        */
    }

    void reset(){
        thrust::fill(Base::d_mass_.begin(),Base::d_mass_.end(), T(0));
        thrust::fill(Base::d_vel_.begin(),Base::d_vel_.end(), T3());
        thrust::fill(Base::d_volume_.begin(),Base::d_volume_.end(), T(0));
    }

protected:

};

} // namespace sim
