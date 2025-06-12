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
#include <limits>
#include <vector>
#include <algorithm> 

#include "Particle/particleDomain.cuh"
#include "Math/vector3.cuh"
#include "Math/matrix3x3.cuh"
#include "Timeintegrator/G2P.cuh"
#include "Timeintegrator/P2G.cuh"
#include "experimental/fluidKernel.cuh"

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
        v = v / mass;
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
/*
__global__
void velocityExtrapolate(NodesData<float> cdata, const size_t c_row, const size_t c_n){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= c_n) return;

    if(cdata.mass[i] != 0.f || !cdata.vel[i].is_zero()) return;

    int c_row2 = c_row * c_row;
    int x = i / c_row2;
    int y = (i % c_row2) / c_row;
    int z = i % c_row;

    Vector3<float> sum = Vector3<float>();
    int count = 0;

    for(int a=-1; a<=1; ++a){
        for(int b=-1; b<=1; ++b){
            for(int c=-1; c<=1; ++c){
                int nx = x + a;
                int ny = y + b;
                int nz = z + c;
                if(nx<0 || ny<0 || nz<0 || nx>=c_row || ny>=c_row || nz>=c_row) continue;
                int neighbor_idx = nx * c_row2 + ny * c_row + nz;

                if(!cdata.vel[neighbor_idx].is_zero()){
                    sum += cdata.vel[neighbor_idx];
                    count += 1;
                }
            }
        }
    }

    if(count > 0){
        cdata.vel[i] = sum / float(count);
    }
}
*/
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

template<typename U>
U sqr(U x) { return x * x; }

template<typename T>
void fastSweepKernel(NodesData<float> cdata, const size_t c_n, const size_t c_r, std::vector<T>& dst, int cycle[6]){
    T a,b,c,hat;
    int c_r2 = c_r*c_r;
    T h = 1.f;
    int near;
    int dx = (cycle[0]<cycle[1])? 1:-1;
    for(int x=cycle[0]; x!=cycle[1]; x+=dx){
        int dy = (cycle[2]<cycle[3])? 1:-1;
        for(int y=cycle[2]; y!=cycle[3]; y+=dy){
            int dz = (cycle[4]<cycle[5])? 1:-1;
            for(int z=cycle[4]; z!=cycle[5]; z+=dz){
                int idx = c_r2*x+c_r*y+z;
                assert(idx >= 0 && idx < static_cast<int>(c_n));
                if(cdata.mass[idx]>=1.e-9){
                    dst[idx] = 0.f; continue;
                }
                a = std::min(dst[c_r2*((x<c_r-1)? x+1:x)+c_r*y+z],dst[c_r2*((x>0)? x-1:x)+c_r*y+z]);
                b = std::min(dst[c_r2*x+c_r*((y<c_r-1)? y+1:y)+z],dst[c_r2*x+c_r*((y>0)? y-1:y)+z]);
                c = std::min(dst[c_r2*x+c_r*y+(z<c_r-1? z+1:z)],dst[c_r2*x+c_r*y+(z>0? z-1:z)]);
                std::array<T, 3> u_arr = {a,b,c};
                std::sort(u_arr.begin(), u_arr.end());
                if(u_arr[0]+h<u_arr[1]) hat = u_arr[0]+h;
                else if(u_arr[0]+h<u_arr[2]) hat = 0.5*(u_arr[0]+u_arr[1]+std::sqrt(2*h*h-sqr(u_arr[0]-u_arr[1])));
                else hat = 1.f/3.f * (a+b+c+std::sqrt(sqr(a+b+c)-3.f*(a*a+b*b+c*c-h*h)));
                if (hat < dst[idx]) {
                    // 最も近い最小の方向を探す
                    int best_offset[3] = {0, 0, 0};
                    if (a <= b && a <= c)
                        best_offset[0] = (dst[c_r2*((x>0)? x-1 : x)+c_r*y+z] < dst[c_r2*((x<c_r-1)? x+1 : x)+c_r*y+z]) ? -1 : 1;
                    else if (b <= a && b <= c)
                        best_offset[1] = (dst[c_r2*x+c_r*((y>0)? y-1 : y)+z] < dst[c_r2*x+c_r*((y<c_r-1)? y+1 : y)+z]) ? -1 : 1;
                    else
                        best_offset[2] = (dst[c_r2*x+c_r*y+((z>0)? z-1 : z)] < dst[c_r2*x+c_r*y+((z<c_r-1)? z+1 : z)]) ? -1 : 1;

                    int nx = std::clamp(x + best_offset[0], 0, static_cast<int>(c_r-1));
                    int ny = std::clamp(y + best_offset[1], 0, static_cast<int>(c_r-1));
                    int nz = std::clamp(z + best_offset[2], 0, static_cast<int>(c_r-1));
                    int near_idx = c_r2 * nx + c_r * ny + nz;

                    assert(near_idx >= 0 && near_idx < static_cast<int>(c_n));
                    cdata.vel[idx] = cdata.vel[near_idx];
                    dst[idx] = hat;
                }
            }
        }
    }
}

template<typename T>
void fastSweep(NodesData<float> cdata, const size_t c_r, const size_t c_n){
    std::vector<T> dst(c_n, std::numeric_limits<float>::max());
    /*call fastSweepKernel*/
    int x=0,y=0,z=0;
    int cycle[6] = {};
    for(int i=0; i<8; i++){
        x = i & 1;
        cycle[0] = (x==0)? 0:c_r-1;
        cycle[1] = (x==0)? c_r:-1;
        y = (i >> 1) & 1;
        cycle[2] = (y==0)? 0:c_r-1;
        cycle[3] = (y==0)? c_r:-1;
        z = (i >> 2) & 1;
        cycle[4] = (z==0)? 0:c_r-1;
        cycle[5] = (z==0)? c_r:-1;
        fastSweepKernel(cdata, c_n, c_r, dst, cycle);
    }
}

template<typename T>
void save_binary_data(int step, const std::vector<std::pair<std::string, const thrust::host_vector<T>&>>& data_pairs) {
    for (const auto& [prefix, data] : data_pairs) {
        std::ostringstream filename;
        filename << "../output/" << prefix << "_" << std::setw(6) << std::setfill('0') << step << ".bin";
        
        std::ofstream ofs(filename.str(), std::ios::binary);
        if (!ofs) {
            std::cerr << "ファイル書き込み失敗: " << filename.str() << std::endl;
            continue;
        }
        
        ofs.write(reinterpret_cast<const char*>(data.data()), sizeof(T) * data.size());
    }
}


} // namespace sim

int main(int argc, char** argv){

    std::string mode = "ELASTIC";
    if(argc > 1){
        mode = argv[1];
    }

    bool isFluid = (mode == "FLUID");

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
    T mu = 5.f;
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

    thrust::device_vector<T> div(c_n, T());
    thrust::host_vector<T> h_div(c_n, T());
    T* div_ptr = thrust::raw_pointer_cast(div.data());
    h_div = div;
    
    float total = 0.0f;
    int count = 0;
    float TIME = 15.0f;
    while(total < TIME){

        cells.reset();

        // 0. estimate the property of CUDA kernel, you can adopt your own evironment
        int blocksize = 512;
        dim3 block (blocksize, 1, 1);
        dim3 gridParticle ((p_n+block.x-1) / block.x, 1, 1);
        dim3 gridCell ((c_n+block.x-1) / block.x, 1, 1);

        auto pdata = particles.getGPUData();
        auto cdata = cells.getGPUData();
        
        if(count==0){
            sim::P2G_simple<<<gridParticle, block>>>(cdata, pdata, c_row, p_n);
            cudaDeviceSynchronize();
            sim::G2P_volume_simple<<<gridParticle, block>>>(cdata, pdata, c_row, p_n, dt);
            cudaDeviceSynchronize();
            cells.reset();
        }

        if(mode == "ELASTIC"){
            sim::forceCalcParticle<<<gridParticle, block>>>(pdata, p_n, dt);
            cudaDeviceSynchronize();
        }

        sim::P2G_simple<<<gridParticle, block>>>(cdata, pdata, c_row, p_n);
        cudaDeviceSynchronize();
        sim::P2G_volume_simple<<<gridParticle, block>>>(cdata, pdata, c_row, p_n);
        cudaDeviceSynchronize();

        sim::massDevide<<<gridCell, block>>>(cdata, c_row, c_n, dt, -9.8f);
        cudaDeviceSynchronize();

        cells.sync(sim::Nodes<T>::CopyDir::DeviceToHost); 
        auto cdataCPU = cells.getCPUData();
        sim::fastSweep<T>(cdataCPU, c_row, c_n);
        cells.sync(sim::Nodes<T>::CopyDir::HostToDevice); 

        if(mode == "FLUID"){
            sim::linearSolver<T>(cdata, pdata, c_row, c_n,  p_n, dt);
        }

        sim::G2P_simple<<<gridParticle, block>>>(cdata, pdata, c_row, p_n, dt);
        cudaDeviceSynchronize();
        
        total += dt;
        count++;

        // progress
        if(count % 20 == 0){
            float progress = total / TIME;
            sim::displayProgress(progress);
            particles.sync(sim::Nodes<T>::CopyDir::DeviceToHost); 
            cells.sync(sim::Nodes<T>::CopyDir::DeviceToHost); 
            sim::save_binary_data<sim::Vector3<T>>(count, {
                {"pos", particles.h_pos_},
                {"grid", cells.h_vel_}
            });
            h_div = div;
            sim::save_binary_data<T>(count, {
                {"div", h_div}
            });
        }
        if(count % 200 == 0 && count < 1000){
            particles.particleIncrease(p_n0, pos, vel, mass, vol, C, F, eq_16_term_0);
            p_n = particles.num_;
            //cells.printInfo();
        }
    }

    cells.printInfo();

    return 0;

}