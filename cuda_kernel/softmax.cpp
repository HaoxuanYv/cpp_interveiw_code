#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#define warpSize 32
/***
 * T __shfl_down_sync(unsigned mask, T var, unsigned int offset, int width = warpSize);
 * mask: 参与操作的线程掩码
 * var: 要交换的数据值
 * offset: 向下移动的偏移量
 * witdh: 参与操作的线程组大小（默认warpSize = 32)
 *
 * 原理：线程i从i+offset获取数据， 对于超出范围的线程（i + delta >= width)，获取自身val
 *       只在寄存器层面交换数据，不需要共享内存，效率高
 ***/
__device__ float warp_reduce_max(float val){
    for(int offset = warpSize / 2; offset > 0; offset /= 2){
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warp_reduce_sum(float val){
    for(int offset = warpSize / 2; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

int main(){
}
__device__ float block_reduce_sum(float val){
    val = warp_reduce_sum(float val);
    // 一个block 包括多个lane(warp)，我们把多个warp的结果再次reduce
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    // 把warp_reduce后的多个warp的warp[0]放到shared memory
    if(lane == 0){
        shared[wid] = val;
    }

    __syncthreads();
    // warp 0 再次reduce,针对blockSize != 1024做了处理，
    val = (threadIdx.x < (blockDim.x / warpSize)) ? shared[lane] : 0.0f;
    // warp 0 再次规约
    if(wid == 0){
        val = warp_reduce_sum(val);
    }
    // 保存结果，shared[0] block内均可见
    if(threadIdx.x == 0) shared[0] = val;
    __syncthreads();

    return shared[0];

}

__device__ float block_reduce_max(float val){
    val = warp_reduce_max(val);
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if(lane == 0){
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / warpSize)) ? shared[lane]: -1e20f;

    if(wid == 0){
        warp_reduce_sum(val);
    }
    if(threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}
template <typename T>
__global__ void softmax_kernel(T* input, T* output, int D){
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if(row_idx >= gridDim.x) return;

    T* row_input = input + row_idx * D;
    T* output = output + row_idx * D;

    // find row max
    float local_max = -1e20;

    for(int i = tid; i < D; i += blockDim.x){
        local_max = max(local_max, (float)row_input[tid]);
    }

    float row_max = block_reduce_max(local_max);

    float local_sum = 0.0f;
    for(int i = tid; i < D; i += blockDim.x){
        local_sum += expf((float)row_input[tid] - row_max);
    }
    float row_sum = block_reduce_sum(local_sum);

    float(int i = tid; i < D; i += blockDim.x){
        row_output[i] = expf((float)row_input[i] - row_max) / row_sum;
    }
}



