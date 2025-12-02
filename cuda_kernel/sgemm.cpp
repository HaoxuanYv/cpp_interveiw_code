#include <iostream>
#include <cuda_runtime.h>

template <int block_size>
__global__ sgemm_naive(float* A, float* B, float* C, float alpha, float beta, int M, int N, int K){
    int row = blockIdx.y * block_size + threadIdx.x / block_size;
    int col = blockIdx.x * block_size + threadIdx.x % block_size;

    if(row < M && col < N){
        float accumulator = 0.0f;
        for(int j = 0; j < K; j++){
            accumulator += A[row * K + j] * B[k * N + col];
        }
        C[row * N + col] = accumulator * alpha + beta * C[row*N + col];
    }
    return;
}

template <int block_size>
__global__ sgemm_shared_memory(float* A, float* B, float* C, float alpha, float beta, int M, int N, int K){
    int rowBlock = blockIdx.y;
    int colBlock = blockIdx.x;

    int rowThread = threadIdx.x / block_size;
    int colThread = threadIdx.x % block_size;

    __shared__ tileA[block_size][block_size];
    __shared__ tileB[block_size][block_size];

    int globalRow = rowBlock * block_size + rowThread;
    int globalCol = colBlock * block_size + colThread;

    A += rowBlock * block_size * K; //A[rowBlock][0]
    B += colBlock * block_size; // B[0][colBlock]
    C += rowBlock * block_size * N + colBlock * block_size;
    float accumulator = 0.0f;
    for(int tile = 0; tile < K; tile += block_size){
        tileA[rowThread][colThread] = A[rowThread * K + colThread];
        tileB[rowThread][colThread] = B[rowThread * N + colThread];

        __syncthreads();
        A += block_size;
        B += block_size * N;

        for(int j = 0; j < block_size; j++){
            accumulator += tileA[rowThread][j] * tileB[j][colThread];
        }
        __syncthreads();
    }
    C[globalRow * N + globalCol] = accumulator * alpha + C[globalRow * N + globalCol] * beta;
}

