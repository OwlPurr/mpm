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
#include "G2P.cuh"

namespace sim{

__global__ void G2P_simple(    
    NodesData<float> cdata, 
    ParticlesData<float> pdata, 
    const int c_row, 
    const int p_n,
    const float dt){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=p_n) return;
    Vector3<float> par_x;
    par_x = pdata.pos[i];
    Matrix3x3<float> C;
    Vector3<float> v;
    float weights[9];
    float cell_idx, cell_diff;
    for(int j=0; j<3; j++){
        cell_idx = floor(par_x(j)); 
        cell_diff = par_x(j) - cell_idx - 0.5;
        weights[3*j] = 0.5 * (0.5-cell_diff) * (0.5-cell_diff);
        weights[3*j+1] = 0.75 - cell_diff * cell_diff;
        weights[3*j+2] = 0.5 * (0.5+cell_diff) * (0.5+cell_diff);
    }
    int smallest_node[3];
    smallest_node[0] = (int)par_x(0)-1;
    smallest_node[1] = (int)par_x(1)-1;
    smallest_node[2] = (int)par_x(2)-1;
    Vector3<float> cell_x;
    Vector3<float> cell_v;
    Vector3<float> weighted_velocity;
    for(int j=0; j<3; j++){
        for(int k=0; k<3; k++){
            for(int l=0; l<3; l++){
                float weight = weights[j]*weights[3+k]*weights[6+l];

                cell_v = cdata.vel[(j+smallest_node[0])*c_row*c_row+(k+smallest_node[1])*c_row+l+smallest_node[2]];

                weighted_velocity(0) = weight * cell_v(0);
                weighted_velocity(1) = weight * cell_v(1);
                weighted_velocity(2) = weight * cell_v(2);
                
                v(0) += weighted_velocity(0);
                v(1) += weighted_velocity(1);
                v(2) += weighted_velocity(2);
                
                cell_x(0) = 4.0f*(j + smallest_node[0] + 0.5f - par_x(0));
                cell_x(1) = 4.0f*(k + smallest_node[1] + 0.5f - par_x(1));
                cell_x(2) = 4.0f*(l + smallest_node[2] + 0.5f - par_x(2));
                C(0,0) += weighted_velocity(0)*cell_x(0);
                C(1,0) += weighted_velocity(1)*cell_x(0);
                C(2,0) += weighted_velocity(2)*cell_x(0);
                C(0,1) += weighted_velocity(0)*cell_x(1);
                C(1,1) += weighted_velocity(1)*cell_x(1);
                C(2,1) += weighted_velocity(2)*cell_x(1);
                C(0,2) += weighted_velocity(0)*cell_x(2);
                C(1,2) += weighted_velocity(1)*cell_x(2);
                C(2,2) += weighted_velocity(2)*cell_x(2);
                
            }
        }      
    }
    pdata.vel[i] = v;
    pdata.C[i] = C;
    
    par_x += dt*v;
    // to prevent to access out of domain particles can exist.
    for(int j=0; j<3; j++) par_x(j) = max(1.0f, min(float(c_row)-1.0f, par_x(j)));
    pdata.pos[i] = par_x;

    Matrix3x3<float> I = Matrix3x3<float>::identity();
    pdata.F[i] = ( I + dt*C )*pdata.F[i];
}

__global__ void G2P_volume_simple(    
    NodesData<float> cdata, 
    ParticlesData<float> pdata, 
    const int c_row, 
    const int p_n,
    const float dt){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=p_n) return;
    Vector3<float> par_x;
    par_x = pdata.pos[i];
    Matrix3x3<float> C;
    Vector3<float> v;
    float weights[9];
    float cell_idx, cell_diff;
    for(int j=0; j<3; j++){
        cell_idx = floor(par_x(j)); 
        cell_diff = par_x(j) - cell_idx - 0.5;
        weights[3*j] = 0.5 * (0.5-cell_diff) * (0.5-cell_diff);
        weights[3*j+1] = 0.75 - cell_diff * cell_diff;
        weights[3*j+2] = 0.5 * (0.5+cell_diff) * (0.5+cell_diff);
    }
    int smallest_node[3];
    smallest_node[0] = (int)par_x(0)-1;
    smallest_node[1] = (int)par_x(1)-1;
    smallest_node[2] = (int)par_x(2)-1;
    Vector3<float> cell_x;
    Vector3<float> cell_v;
    Vector3<float> weighted_velocity;
    float density = 0.f;
    for(int j=0; j<3; j++){
        for(int k=0; k<3; k++){
            for(int l=0; l<3; l++){
                float weight = weights[j]*weights[3+k]*weights[6+l];

                density += weight * cdata.mass[(j+smallest_node[0])*c_row*c_row+(k+smallest_node[1])*c_row+l+smallest_node[2]];
                
            }
        }      
    }

    if(density!=0.f)  pdata.volume_0[i] = pdata.mass[i] / density;

}

}