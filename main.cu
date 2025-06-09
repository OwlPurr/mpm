// Ubuntu22.04 / nvidia-driver-560 / cuda-12-6 tested.

#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>

#include <cublas_v2.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <math.h>

#include "Particle/particleKernels.cuh"
#include "Particle/particleDomain.cuh"
#include "Math/vector3.cuh"
#include "Math/matrix3x3.cuh"
#include "Timeintegrator/G2P.cuh"
#include "Timeintegrator/P2G.cuh"

using T = float;
__constant__ T d_mu_, d_lambda_;

namespace sim {

void displayProgress(float progress) {
    const int barWidth = 50; 
    int pos = static_cast<int>(progress * barWidth); 

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
            std::cout << "|";
        else
            std::cout << ".";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << progress * 100.0 << "%\r";
    std::cout.flush();
}

__global__
void massDevide(NodesData<float> cdata, const size_t c_row, const size_t c_n, const float dt, const float gravity){
    //Velocity Computation
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= c_n) return;
    float mass = cdata.mass[i];
    Vector3<float> v = cdata.vel[i];
    if(mass != 0.0f){
        v /= mass;
    }
    
    int minCell = 2; 
    int maxCell = c_row - 3;
    int x = i / (c_row*c_row);
    int y = (i % (c_row*c_row)) / c_row;
    int z = i % c_row;

    v(2) += dt*gravity;
    if(x < minCell || x > maxCell){ v(0) = 0.0;}
    if(y < minCell || y > maxCell){ v(1) = 0.0;}
    if(z < minCell || z > maxCell){ v(2) = 0.0;}

    cdata.vel[i] = v;
}

__global__
void forceCalcParticle(ParticlesData<float> pdata, const int p_n, const float dt){
    //P2G and Velocity Update
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=p_n) return;

    Matrix3x3<float> F_T, F_inv_T, F_minus_F_inv_T, P, stress;

    float J = pdata.F[i].determinant();
    float volume = J * pdata.volume_0[i];

    F_T = pdata.F[i].transpose();
    F_inv_T = F_T.inverse();
    F_minus_F_inv_T = pdata.F[i] - F_inv_T;

    P = 2*d_mu_ * F_minus_F_inv_T + d_lambda_ * J*(J-1.f) * F_inv_T;
    stress = 1.f / J * P * F_T;
    pdata.eq_16_term_0[i] = -volume * 4.f * dt * stress;
} 



template<typename T>
void save_binary_data(int step, const std::vector<std::pair<std::string, const thrust::host_vector<Vector3<T>>&>>& data_pairs) {
    for (const auto& [prefix, data] : data_pairs) {
        std::ostringstream filename;
        filename << "../output/" << prefix << "_" << std::setw(6) << std::setfill('0') << step << ".bin";
        
        std::ofstream ofs(filename.str(), std::ios::binary);
        if (!ofs) {
            std::cerr << "ファイル書き込み失敗: " << filename.str() << std::endl;
            continue;
        }
        
        ofs.write(reinterpret_cast<const char*>(data.data()), sizeof(Vector3<T>) * data.size());
    }
}

} // namespace sim

int main(){

    using T3 = sim::Vector3<T>;
    using T3x3 = sim::Matrix3x3<T>;
    size_t p_n, c_n;
    size_t c_row = 32;
    c_n = pow(c_row, 3);
    T dt = 0.01f;
    
    // import positions of particles 
    std::ifstream ifs("../Assets/cube.txt");
    if(!ifs){
        std::cerr << "Error file not opened." << std::endl;
        return 1;
    }

    std::vector<T> raw_data((std::istream_iterator<T>(ifs)),
                                    std::istream_iterator<T>());

    size_t size_data = raw_data.size();
    if (size_data % 3 != 0) {
        std::cerr << "Error: data size is not divisible by 3." << std::endl;
        return 1;
    }
    p_n = size_data / 3;
    std::cout << "Read " << p_n << " particles (" << size_data << " values)" << std::endl;
    int p_n0 = p_n;

    // other field of particles class (but not initialized yet)
    thrust::host_vector<T3> pos(p_n, T3{0.0f, 0.0f, 0.0f});
    thrust::host_vector<T3> vel(p_n, T3{-1.0f, 1.0f, 1.0f});
    thrust::host_vector<T> mass(p_n, 100.0f/p_n); // for now, any value is fine
    thrust::host_vector<T> vol(p_n, 1900.0f/p_n);  
    thrust::host_vector<T3x3> C(p_n, T3x3());
    thrust::host_vector<T3x3> F(p_n, T3x3::identity());  
    thrust::host_vector<T3x3> eq_16_term_0(p_n, T3x3());  
    T mu = 1.f;
    T lambda = 1.f;
    cudaMemcpyToSymbol(d_mu_, &mu, sizeof(T)); 
    cudaMemcpyToSymbol(d_lambda_, &lambda, sizeof(T)); 

    // construct particles class
    for (size_t i = 0; i < p_n; i++) {
        pos[i] = {raw_data[3*i], raw_data[3*i+1], raw_data[3*i+2]};
    }
    sim::Particles<T> particles(p_n, pos, vel, mass, vol, C, F, eq_16_term_0);
    particles.sync(sim::Nodes<T>::CopyDir::HostToDevice); 

    // construct cells class
    thrust::host_vector<T3> pos2(c_n, T3{0.0f, 0.0f, 0.0f});
    thrust::host_vector<T3> vel2(c_n, T3{0.0f, 0.0f, 0.0f});
    thrust::host_vector<T> mass2(c_n, 0.0);
    thrust::host_vector<T> vol2(c_n, 0.0f);  
    sim::Cells<T> cells(c_n, pos2, vel2, mass2, vol2);
    cells.sync(sim::Nodes<T>::CopyDir::HostToDevice); 
    
    float total = 0.0f;
    int count = 0;
    float TIME = 100.0f;
    while(total < TIME){

        cells.reset();

        // 0. estimate the property of CUDA kernel, you can adopt your own evironment
        int blocksize = 512;
        dim3 block (blocksize, 1, 1);
        dim3 gridParticle ((p_n+block.x-1) / block.x, 1, 1);
        dim3 gridCell ((c_n+block.x-1) / block.x, 1, 1);

        // 1. calculate d_cellids
        thrust::device_vector<int> d_cellids(p_n);
        int* raw_ptr_cellids = thrust::raw_pointer_cast(d_cellids.data());
        T3* raw_ptr_pos = thrust::raw_pointer_cast(particles.d_pos_.data());
        sim::presort<<<gridParticle, block>>>(raw_ptr_pos, raw_ptr_cellids, p_n, c_row);
        cudaDeviceSynchronize();

        // 2. sort field of Particle class using d_cellids
        particles.resort(d_cellids);

        
        auto pdata = particles.getGPUData();
        auto cdata = cells.getGPUData();
        
        if(count==0){
            sim::P2G_simple<<<gridParticle, block>>>(cdata, pdata, c_row, p_n);
            cudaDeviceSynchronize();
            sim::G2P_volume_simple<<<gridParticle, block>>>(cdata, pdata, c_row, p_n, dt);
            cudaDeviceSynchronize();
            cells.reset();
        }
        
        sim::forceCalcParticle<<<gridParticle, block>>>(pdata, p_n, dt);
        cudaDeviceSynchronize();

        sim::P2G_simple<<<gridParticle, block>>>(cdata, pdata, c_row, p_n);
        cudaDeviceSynchronize();
        //sim::P2G_volume_simple<<<gridParticle, block>>>(cdata, pdata, c_row, p_n);
        //cudaDeviceSynchronize();

        sim::massDevide<<<gridCell, block>>>(cdata, c_row, c_n, dt, -9.8f);
        cudaDeviceSynchronize();

        //sim::linearSolver<T>(cdata, pdata, raw_ptr_cellids, raw_ptr_gsp_idx, raw_ptr_pages, raw_ptr_vir, CUDABlock_num, c_row, c_n,  p_n, dt);

        sim::G2P_simple<<<gridParticle, block>>>(cdata, pdata, c_row, p_n, dt);
        cudaDeviceSynchronize();
        
        total += dt;
        count++;

        // progress
        if(count % 100 == 0){
            float progress = total / TIME;
            sim::displayProgress(progress);
            particles.sync(sim::Nodes<T>::CopyDir::DeviceToHost); 
            sim::save_binary_data<float>(count, {
                {"pos", particles.h_pos_},
                {"vel", particles.h_vel_}
            });
        }
        if(count % 1000 == 0){
            particles.particleIncrease(p_n0, pos, vel, mass, vol, C, F, eq_16_term_0);
            p_n = particles.num_;
        }
    }

    cells.printInfo();

    return 0;

}