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

#include "../Particle/particleDomain.cuh"
#include "../Math/vector3.cuh"
#include "../Math/matrix3x3.cuh"
#include "P2G.cuh"

namespace sim{

__global__ void P2G_simple(
    NodesData<float> cdata, 
    ParticlesData<float> pdata, 
    const int c_row, 
    const int p_n)
    {

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=p_n) return;
    float val[4];
    Vector3<float> par_x = pdata.pos[i];
    Matrix3x3<float> C = pdata.C[i];
    Matrix3x3<float> eq_16_term_0 = pdata.eq_16_term_0[i];
    Vector3<float> v = pdata.vel[i];
    float mass = pdata.mass[i];
    float weights[9];
    float cell_idx, cell_diff;
    for(int j=0; j<3; j++){
        cell_idx = floor(par_x(j)); 
        cell_diff = par_x(j) - cell_idx - 0.5;
        weights[3*j] = 0.5 * (0.5-cell_diff) * (0.5-cell_diff);
        weights[3*j+1] = 0.75 - cell_diff * cell_diff;
        weights[3*j+2] = 0.5 * (0.5+cell_diff) * (0.5+cell_diff);
    }
    int smallest_node[3] = {
        static_cast<int>(par_x.x)-1,
        static_cast<int>(par_x.y)-1,
        static_cast<int>(par_x.z)-1
    };
    Vector3<float> c_dst, vel;
    for(int j=0; j<3; j++){
        for(int k=0; k<3; k++){
            for(int l=0; l<3; l++){
                c_dst(0) = smallest_node[0] + j + 0.5 - par_x(0);
                c_dst(1) = smallest_node[1] + k + 0.5 - par_x(1);
                c_dst(2) = smallest_node[2] + l + 0.5 - par_x(2);
                float weight = weights[j]*weights[3+k]*weights[6+l];
                float weighted_mass = weight * mass;
                val[0] = weighted_mass;
                vel = weighted_mass * (v + C * c_dst) + eq_16_term_0 * c_dst * weight; 
                val[1] = vel(0);
                val[2] = vel(1);
                val[3] = vel(2);
                
                int grid_idx = (smallest_node[0] + j)*c_row*c_row+(smallest_node[1] + k)*c_row+smallest_node[2] + l;
                atomicAdd(&cdata.mass[grid_idx], val[0]);
                atomicAdd(&cdata.vel[grid_idx](0), val[1]);
                atomicAdd(&cdata.vel[grid_idx](1), val[2]);
                atomicAdd(&cdata.vel[grid_idx](2), val[3]);
            }
        }      
    }
    
}

__global__ void P2G_volume_simple(
    NodesData<float> cdata, 
    ParticlesData<float> pdata, 
    const int c_row, 
    const int p_n)
    {

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=p_n) return;
    float val[4];
    Vector3<float> par_x = pdata.pos[i];
    Matrix3x3<float> C = pdata.C[i];
    Matrix3x3<float> eq_16_term_0 = pdata.eq_16_term_0[i];
    Vector3<float> v = pdata.vel[i];
    float mass = pdata.volume_0[i];
    float weights[9];
    float cell_idx, cell_diff;
    for(int j=0; j<3; j++){
        cell_idx = floor(par_x(j)); 
        cell_diff = par_x(j) - cell_idx - 0.5;
        weights[3*j] = 0.5 * (0.5-cell_diff) * (0.5-cell_diff);
        weights[3*j+1] = 0.75 - cell_diff * cell_diff;
        weights[3*j+2] = 0.5 * (0.5+cell_diff) * (0.5+cell_diff);
    }
    int smallest_node[3] = {
        static_cast<int>(par_x.x)-1,
        static_cast<int>(par_x.y)-1,
        static_cast<int>(par_x.z)-1
    };
    int x,y,z;
    x = smallest_node[0] & 0x3;
    y = smallest_node[1] & 0x3;
    z = smallest_node[2] & 0x3;
    Vector3<float> c_dst, vel;
    for(int j=0; j<3; j++){
        for(int k=0; k<3; k++){
            for(int l=0; l<3; l++){
                c_dst(0) = smallest_node[0] + j + 0.5 - par_x(0);
                c_dst(1) = smallest_node[1] + k + 0.5 - par_x(1);
                c_dst(2) = smallest_node[2] + l + 0.5 - par_x(2);
                float weight = weights[j]*weights[3+k]*weights[6+l];
                float weighted_mass = weight * mass;
                val[0] = weighted_mass;
                vel = weighted_mass * (v + C * c_dst) + eq_16_term_0 * c_dst * weight; 
                val[1] = vel(0);
                val[2] = vel(1);
                val[3] = vel(2);
                
                int grid_idx = (smallest_node[0] + j)*c_row*c_row+(smallest_node[1] + k)*c_row+smallest_node[2] + l;
                atomicAdd(&cdata.volume[grid_idx], val[0]);
            }
        }      
    }
    
}



}
