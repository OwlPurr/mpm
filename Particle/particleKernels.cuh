#ifndef PARTICLE_KERNELS_CUH
#define PARTICLE_KERNELS_CUH
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

namespace sim {

    __global__ void presort(Vector3<float>* d_pos, int* d_cellids, const size_t p_n, const int c_row);
    __global__ void CountGSPBlock(int* d_cellids, int* BlockBdr, const int p_n);
    __global__ void CountCudaBlock_pre(const int p_n, int* d_blockbdr, int* d_block_offsets);
    __global__ void CountCudaBlock(const int p_n, int* d_blockbdr, int* d_block_offsets, int* CUDABlock, int GSPBlock_num);
    __global__ void CountCudaBlock_post(const int p_n, int* d_blockbdr, int* CUDABlock, int* d_targetPages, int* d_virtualPageOffsets);

}

#endif