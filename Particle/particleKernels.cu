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
#include "particleKernels.cuh"
#include "../Math/vector3.cuh"
#include "../Math/matrix3x3.cuh"

namespace sim {

    //CUDA
    __global__ void presort(Vector3<float>* d_pos, int* d_cellids, const size_t p_n, const int c_row){
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        if(i>=p_n) return;
        
        int x = int(d_pos[i].x)-1;
        int y = int(d_pos[i].y)-1;
        int z = int(d_pos[i].z)-1;
        int result = 0;

        int bits = 31 - __clz(c_row);
        
        result |= (x & 3) | (y & 3) << 2 | (z & 3) << 4;
        x >>= 2;
        y >>= 2;
        z >>= 2;
        
        for (int j=2; j < bits; j++){
            result |= (x & 1) << (3*j) | (y & 1) << (3*j + 1) | (z & 1) << (3*j + 2);
            x >>= 1;
            y >>= 1;
            z >>= 1;
        } 
        
        d_cellids[i] = result;  
    } 

    //CUDA
    __global__ void CountGSPBlock(int* d_cellids, int* BlockBdr, const int p_n){

        int i = blockIdx.x*blockDim.x + threadIdx.x;
        if(i>=p_n) return;
        bool bBoundary;
        int blockid = d_cellids[i] >> 6;
        bBoundary = i == 0 || blockid != d_cellids[i-1] >> 6;
        if(bBoundary){
            BlockBdr[i] = 1;
        }else{
            BlockBdr[i] = 0;
        }
    }

    //CUDA
    __global__ void CountCudaBlock_pre(const int p_n, int* d_blockbdr, int* d_block_offsets){

        int i = blockIdx.x*blockDim.x + threadIdx.x;
        if(i>=p_n) return;
        bool bBoundary;
        int blockid = d_blockbdr[i];
        bBoundary = i == 0 || blockid != d_blockbdr[i-1];
        if(bBoundary){
            d_block_offsets[blockid-1] = i;
        }
    }

    //CUDA
    __global__ void CountCudaBlock(const int p_n, int* d_blockbdr, int* d_block_offsets, int* CUDABlock, int GSPBlock_num){

        int i = blockIdx.x*blockDim.x + threadIdx.x;
        if(i>=p_n) return;
        bool bBoundary;
        int blockid = d_blockbdr[i];
        bBoundary = i == 0 || blockid != d_blockbdr[i-1];
        if(bBoundary){
            CUDABlock[blockid-1] = (d_block_offsets[blockid] - d_block_offsets[blockid-1] + blockDim.x - 1) / blockDim.x;
        }
    }

    //CUDA
    __global__ void CountCudaBlock_post(const int p_n, int* d_blockbdr, int* CUDABlock, int* d_targetPages, int* d_virtualPageOffsets){

        int i = blockIdx.x*blockDim.x + threadIdx.x;
        if(i>=p_n) return;
        bool bBoundary;
        int blockid = d_blockbdr[i];
        bBoundary = i == 0 || blockid != d_blockbdr[i-1];
        if(bBoundary){
            if(i == 0){
                d_targetPages[0] = 1;
                d_virtualPageOffsets[0] = 0;
            }else{
                d_targetPages[CUDABlock[blockid-2]] = 1;
                d_virtualPageOffsets[blockid-1] = CUDABlock[blockid-2];
            }
        }
    }

}