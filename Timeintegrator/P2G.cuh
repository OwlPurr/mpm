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

#include "../Particle/particleDomain.cuh"
#include "../Math/vector3.cuh"
#include "../Math/matrix3x3.cuh"

namespace sim{

    __global__ void P2G_simple(NodesData<float> cdata, ParticlesData<float> pdata, const int c_row, const int p_n);
    __global__ void P2G_volume_simple(NodesData<float> cdata, ParticlesData<float> pdata, const int c_row, const int p_n);
    
}