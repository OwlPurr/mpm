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

#include "../Particle/particleDomain.cuh"
#include "../Math/vector3.cuh"
#include "../Math/matrix3x3.cuh"
#include "../Timeintegrator/G2P.cuh"
#include "../Timeintegrator/P2G.cuh"

namespace sim{
    
__global__
void calcDivVel(float* d_y, NodesData<float> cdata, const size_t c_row, const size_t c_n, const float dt){
    //Velocity Computation
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= c_n) return;
    float mass = cdata.mass[i];
    Vector3<float> v = cdata.vel[i];
    
    int a = i / (c_row*c_row);
    int b = (i % (c_row*c_row)) / c_row;
    int c = i % c_row;

    d_y[i] = 0.f;
    if(a>0 && a<c_row-1 && b>0 && b<c_row-1 && c>0 && c<c_row-1 && mass >= 1.e-9){
        
        for(int j=-1; j<2; j++){
            for(int k=-1; k<2; k++){
                int count = abs(j) + abs(k);
                float coe = 0.125 * std::pow(0.5,count);
                d_y[i] += coe*cdata.vel[(a+1)*c_row*c_row+(b+j)*c_row+c+k](0);
                d_y[i] -= coe*cdata.vel[(a-1)*c_row*c_row+(b+j)*c_row+c+k](0);
                d_y[i] += coe*cdata.vel[(a+j)*c_row*c_row+(b+1)*c_row+c+k](1);
                d_y[i] -= coe*cdata.vel[(a+j)*c_row*c_row+(b-1)*c_row+c+k](1);
                d_y[i] += coe*cdata.vel[(a+j)*c_row*c_row+(b+k)*c_row+c+1](2);
                d_y[i] -= coe*cdata.vel[(a+j)*c_row*c_row+(b+k)*c_row+c-1](2);
            }
        }
    }
}

__global__
void addPressureGrad(float* d_x, NodesData<float> cdata, const size_t c_row, const size_t c_n, const float dt){
    //Velocity Computation
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= c_n) return;
    float mass = cdata.mass[i];
    Vector3<float> v = cdata.vel[i];
    
    int a = i / (c_row*c_row);
    int b = (i % (c_row*c_row)) / c_row;
    int c = i % c_row;

    if(a>0 && a<c_row-1 && b>0 && b<c_row-1 && c>0 && c<c_row-1 && mass >= 1.e-9){
        
        for(int j=-1; j<2; j++){
            for(int k=-1; k<2; k++){
                int count = abs(j) + abs(k);
                float coe = -dt*0.125 * std::pow(0.5,count);
                cdata.vel[i](0) += coe*d_x[(a+1)*c_row*c_row+(b+j)*c_row+c+k];
                
                cdata.vel[i](0) -= coe*d_x[(a-1)*c_row*c_row+(b+j)*c_row+c+k];
                
                cdata.vel[i](1) += coe*d_x[(a+j)*c_row*c_row+(b+1)*c_row+c+k];
                
                cdata.vel[i](1) -= coe*d_x[(a+j)*c_row*c_row+(b-1)*c_row+c+k];
                
                cdata.vel[i](2) += coe*d_x[(a+j)*c_row*c_row+(b+k)*c_row+c+1];
                
                cdata.vel[i](2) -= coe*d_x[(a+j)*c_row*c_row+(b+k)*c_row+c-1];
                
            }
        }
    }
}

__global__
void funcA(float* d_Avec, float* d_p, NodesData<float> cdata, const size_t c_row, const size_t c_n, const float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c_n) return;

    int a = i / (c_row * c_row);
    int b = (i % (c_row * c_row)) / c_row;
    int c = i % c_row;

    float laplacian = 0.0f;
    if (a > 0 && a < c_row - 1 && b > 0 && b < c_row - 1 && c > 0 && c < c_row - 1) {
        float p_center = d_p[i];


        int ip1 = i + c_row * c_row;
        int im1 = i - c_row * c_row;
        int jp1 = i + c_row;
        int jm1 = i - c_row;
        int kp1 = i + 1;
        int km1 = i - 1;


        laplacian = (6.0f * p_center - d_p[ip1] - d_p[im1] - d_p[jp1] - d_p[jm1] - d_p[kp1] - d_p[km1]);

    }
    d_Avec[i] = dt*laplacian;
}


__global__
void projectPressure(float* d_x, const size_t c_row, const size_t c_n){
    //Velocity Computation
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= c_n) return;
    
    int minCell = 2;
    int maxCell = c_row-3;
    int i = idx / (c_row*c_row);
    int j = (idx % (c_row*c_row)) / c_row;
    int k = idx % c_row;

    if(i < minCell){ 
        d_x[idx] = d_x[(minCell) * c_row * c_row + j * c_row + k];
    }
    if(i > maxCell){ 
        d_x[idx] = d_x[(maxCell) * c_row * c_row + j * c_row + k];
    }
    if(j < minCell){ 
        d_x[idx] = d_x[i * c_row * c_row + (minCell) * c_row + k];
    }
    if(j > maxCell){ 
        d_x[idx] = d_x[i * c_row * c_row + (maxCell) * c_row + k];
    }
    if(k < minCell){ 
        d_x[idx] = d_x[i * c_row * c_row + j * c_row + (minCell)];
    }
    if(k > maxCell){ 
        d_x[idx] = d_x[i * c_row * c_row + j * c_row + (maxCell)];
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

template <typename T>
void linearSolver(   
    NodesData<T> cdata, 
    ParticlesData<T> pdata, 
    const int c_row,
    const int c_n, 
    const int p_n,
    const T dt)
    {

    int blocksize = 512;
    dim3 block (blocksize, 1, 1);
    dim3 gridCell  ((c_n+block.x-1) / block.x, 1, 1);
    dim3 gridParticle  ((p_n+block.x-1) / block.x, 1, 1);

    // CUBLASハンドル作成
    cublasHandle_t handle;
    cublasCreate(&handle);

    T *d_y, *d_r; // they equals Div of velocity
    cudaMalloc(&d_y, c_n*sizeof(T));
    cudaMalloc(&d_r, c_n*sizeof(T));

    calcDivVel<<<gridCell, block>>>(d_y, cdata, c_row, c_n, dt); // this calc d_y(=Div of vel) using c_data
    CUDA_CHECK(cudaDeviceSynchronize());
    //float yy;
    //cublasSdot(handle, c_n, d_y, 1, d_y, 1, &yy);
    //std::cout << "yy: " << yy << std::endl;

    cudaMemcpy(d_r, d_y, c_n*sizeof(T), cudaMemcpyDeviceToDevice);

    T *d_p; // this p vector is initial conjugate directions
    cudaMalloc(&d_p, c_n*sizeof(T));
    const float minus_one = -1.0f;
    cublasSscal(handle, c_n, &minus_one, d_y, 1);
    cudaMemcpy(d_p, d_y, c_n * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaFree(d_y);

    T *d_x; // this x vector is A*x(<-here!)-b=r.
    cudaMalloc(&d_x, c_n*sizeof(T));
    cudaMemset(d_x, (T)0, c_n*sizeof(T));

    T *d_Avec; // the vector that underwent hessian operation
    cudaMalloc(&d_Avec, c_n*sizeof(T));
    cudaMemset(d_Avec, (T)0, c_n*sizeof(T));

    T rr, rr_new;
    cublasSdot(handle, c_n, d_r, 1, d_r, 1, &rr_new);
    int ite = 0;
    float tolerance = 1.0e-2;
    
    while(ite < 500){
        funcA<<<gridCell, block>>>(d_Avec, d_p, cdata, c_row, c_n, dt); // hessian operation, Avec = A*p
        CUDA_CHECK(cudaDeviceSynchronize());

        cublasSdot(handle, c_n, d_r, 1, d_r, 1, &rr);
        float pAp;
        cublasSdot(handle, c_n, d_p, 1, d_Avec, 1, &pAp);
        //std::cout << "pAp: " << pAp << std::endl;
        //std::cout << "rr: " << rr << std::endl;
        if(pAp < tolerance) break;
        float alpha = rr / pAp;
        //cout << "alpha: " << alpha << endl;
        cublasSaxpy(handle, c_n, &alpha, d_p, 1, d_x, 1);
        cublasSaxpy(handle, c_n, &alpha, d_Avec, 1, d_r, 1);

        cublasSdot(handle, c_n, d_r, 1, d_r, 1, &rr_new);
        if(rr < tolerance || rr_new < tolerance) break;
        float beta = rr_new / rr;
        //std::cout << "linear solver error: " << sqrt(rr_new)  << " iteration: " << ite << std::endl;
        
        cublasSscal(handle, c_n, &beta, d_p, 1);
        cublasSaxpy(handle, c_n, &minus_one, d_r, 1, d_p, 1);
    
        
        ite++;
    }
    
    projectPressure<<<gridCell, block>>>(d_x, c_row, c_n);
    CUDA_CHECK(cudaDeviceSynchronize());
    addPressureGrad<<<gridCell, block>>>(d_x, cdata, c_row, c_n, dt); // update cdata.vel
    CUDA_CHECK(cudaDeviceSynchronize());
    //forceCalcParticleFluidPressure_Implicit<<<gridParticle, block>>>(cdata, pdata, c_row, p_n, dt, d_x);
    //CUDA_CHECK(cudaDeviceSynchronize());

    cublasDestroy(handle);
    cudaFree(d_p);
    cudaFree(d_x);
    cudaFree(d_Avec);
    cudaFree(d_r);

}

}