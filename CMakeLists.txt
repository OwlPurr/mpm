cmake_minimum_required(VERSION 3.18)
project(MPM_CUDA_PROJECT LANGUAGES CXX CUDA)

# CUDA設定
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -O3")

# ソースファイル
set(SRC 
    main.cu
    Particle/particleKernels.cu
    Particle/particleDomain.cuh
    Timeintegrator/G2P.cu
    Timeintegrator/P2G.cu
)

# ターゲット作成
set(TARGET_NAME "mpm_cuda_executable")
add_executable(${TARGET_NAME} ${SRC})

# ライブラリリンク
target_link_libraries(${TARGET_NAME}
    cublas
)
