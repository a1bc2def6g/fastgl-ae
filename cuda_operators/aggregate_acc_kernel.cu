#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <time.h>
constexpr size_t BLOCK_SIZE = 128; 
constexpr size_t BLOCK_X = 8; 
constexpr size_t BLOCK_Y = 32; 
constexpr size_t BLOCK_X_GAT = 8; 
constexpr size_t BLOCK_Y_GAT = 8; 
constexpr size_t BLOCK_Z_GAT = 8; 
constexpr size_t neighbor_size = 5; 
constexpr size_t dim_size = 256; 


__device__ inline 
void atomicAdd_F(float* address, float value)
{
  float old = value;  
  while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
}

template <typename scalar_t>
__global__ void aggregate_forward_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_norm_degs,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_norm_degs,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> output,
                    int neighbor_num,
                    int dim,
                    int dst_nodes
);

// gcn
template <typename scalar_t>
__global__ void computation_aware_forward_gcn_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_norm_degs,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_norm_degs,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> output,
                    int neighbor_num,
                    int dim,
                    int dst_nodes
);

template <typename scalar_t>
__global__ void computation_aware_backward_gcn_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_norm_degs,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_norm_degs,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_input,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_output,
                    int dim,
                    int src_nodes
);

template <typename scalar_t>
__global__ void aggregate_forward_outer_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_edges,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_norm_degs,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_norm_degs,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> output,
                    int max_degree,
                    int dim,
                    int src_nodes
);

template <typename scalar_t>
__global__ void computation_aware_forward_gin_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> output,
                    int neighbor_num,
                    int dim,
                    int dst_nodes
);

template <typename scalar_t>
__global__ void computation_aware_backward_gin_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_input,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_output,
                    int dim,
                    int dst_nodes
);

template <typename scalar_t>
__global__ void computation_aware_forward_gat_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> edge_data,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> output,
                    int neighbor_num,
                    int dim,
                    int unit_dim,
                    int dst_nodes,
                    int num_heads
);

template <typename scalar_t>
__global__ void computation_aware_backward_gat_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> edge_data,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_input,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_output,
                    int dim,
                    int unit_dim,
                    int dst_nodes,
                    int num_heads
);


std::vector<torch::Tensor> aggregate_forward_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor src_edges,
    torch::Tensor src_norm_degs,
    torch::Tensor dst_norm_degs,
    torch::Tensor input_feat,
    torch::Tensor weight,
    int neighbor_num,
    int dst_nodes
)
{   
    auto tmp = torch::mm(input_feat, weight);
    int dim = tmp.size(1);
    auto output = torch::zeros({dst_nodes, tmp.size(1)}, torch::kCUDA);
    int shared_memory = BLOCK_X * (neighbor_num + 1)*sizeof(int) + BLOCK_X * neighbor_num * sizeof(int);
    const dim3 threads(BLOCK_X, BLOCK_Y);
    const dim3 blocks((dst_nodes+threads.x-1)/threads.x, (dim+threads.y-1)/threads.y);
    
    // const int threadsPerBlock = BLOCK_SIZE;
    // const int numBlocks = dst_nodes;
    AT_DISPATCH_FLOATING_TYPES(tmp.type(), "aggregate_forward_cuda", ([&] {
                                    aggregate_forward_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        edge_ptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_norm_degs.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        dst_norm_degs.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        tmp.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        output.packed_accessor32<float,2,torch::RestrictPtrTraits>(), 
                                        neighbor_num,
                                        dim,
                                        dst_nodes
                                    );
                                }));
    return {output};
}


template <typename scalar_t>
__global__ void aggregate_forward_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_norm_degs,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_norm_degs,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> output,
                    int neighbor_num,
                    int dim,
                    int dst_nodes
)
{
    
    
    // the target node aggregated by this current block
    int current_dst_id = blockIdx.x * blockDim.x + threadIdx.x;

    // the feature dimension
    int feat_id = blockIdx.y * blockDim.y + threadIdx.y;

    if((current_dst_id>=dst_nodes) || (feat_id>=dim)){
        return;
    }
    
    
    // store w and f into the shared memory
    extern __shared__ int shared_space[];
    int *src_id = shared_space;
    int *src_norm_deg = &shared_space[blockDim.x*neighbor_num];
    float dst_norm_deg;

    dst_norm_deg = 1/sqrtf(dst_norm_degs[current_dst_id]);
    int src_num = edge_ptr[current_dst_id + 1] - edge_ptr[current_dst_id];
    
    if(current_dst_id<dst_nodes){
        if(src_num==0){
            output[current_dst_id][feat_id] = input_feat[current_dst_id][feat_id];
        }
        else{
            auto edge_index = edge_ptr[current_dst_id];
            #pragma unroll
            for (int index = 0; index < src_num; index++){
                int src_nid = src_edges[edge_index+index];
                src_id[threadIdx.x*neighbor_num + index] = src_nid;
                src_norm_deg[threadIdx.x*neighbor_num + index] = src_norm_degs[src_nid]; 
            }
            __syncthreads();
            #pragma unroll
            for (int src = 0; src < src_num; src += 1){
                int nid = src_id[threadIdx.x*neighbor_num + src];
                float degree_norm_inv = __fmaf_rn(dst_norm_deg, 1/sqrtf(src_norm_deg[threadIdx.x*neighbor_num + src]), 0);

                output[current_dst_id][feat_id] += __fmaf_rn(degree_norm_inv, input_feat[nid][feat_id], 0);
            }
            // __syncthreads();
        }   
    }
}

std::vector<torch::Tensor> computation_aware_forward_gcn_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor src_edges,
    torch::Tensor src_norm_degs,
    torch::Tensor dst_norm_degs,
    torch::Tensor input_feat,
    torch::Tensor weight,
    int neighbor_num,
    int dst_nodes
)
{   
    // float mm_start_time = clock();

    // perform the update phase by the optimized function of torch
    auto tmp = torch::mm(input_feat, weight);

    // float mm_end_time = clock();
    int dim = tmp.size(1);
    auto output = torch::zeros({dst_nodes, tmp.size(1)}, torch::kCUDA);
    int shared_memory = BLOCK_X * (neighbor_num + 1)*sizeof(int) + BLOCK_X * neighbor_num * sizeof(int) + BLOCK_X * BLOCK_Y * sizeof(float);
    const dim3 threads(BLOCK_X, BLOCK_Y);
    const dim3 blocks((dst_nodes+threads.x-1)/threads.x, (dim+threads.y-1)/threads.y);

    // float agg_start_time = clock();
    
    // const int threadsPerBlock = BLOCK_SIZE;
    // const int numBlocks = dst_nodes;
    AT_DISPATCH_FLOATING_TYPES(tmp.type(), "aggregate_forward_cuda", ([&] {
                                    computation_aware_forward_gcn_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        edge_ptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_norm_degs.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        dst_norm_degs.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        tmp.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        output.packed_accessor32<float,2,torch::RestrictPtrTraits>(), 
                                        neighbor_num,
                                        dim,
                                        dst_nodes
                                    );
                                }));
    // float agg_end_time = clock();

    // printf("mm time:%f;agg time:%f\n",(mm_end_time-mm_start_time),(agg_end_time-agg_start_time));
    return {output};
}


template <typename scalar_t>
__global__ void computation_aware_forward_gcn_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_norm_degs,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_norm_degs,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> output,
                    int neighbor_num,
                    int dim,
                    int dst_nodes
)
{
    
    
    // the target node aggregated by this current block
    int current_dst_id = blockIdx.x * blockDim.x + threadIdx.x;

    // the feature dimension is accumulating in this thread
    int feat_id = blockIdx.y * blockDim.y + threadIdx.y;

    if((current_dst_id>=dst_nodes) || (feat_id>=dim)){
        return;
    }
    
    
    // store w and f into the shared memory
    extern __shared__ int shared_space[];
    int *src_id = shared_space;
    int *src_norm_deg = &shared_space[blockDim.x*neighbor_num];
    float *partial_output = (float*)&src_norm_deg[blockDim.x*neighbor_num];
    float dst_norm_deg;

    dst_norm_deg = 1/sqrtf(dst_norm_degs[current_dst_id]);
    int src_num = edge_ptr[current_dst_id + 1] - edge_ptr[current_dst_id];
    
    if(current_dst_id<dst_nodes){
        if(src_num==0){
            output[current_dst_id][feat_id] = input_feat[current_dst_id][feat_id];
        }
        else{
            auto edge_index = edge_ptr[current_dst_id];
            #pragma unroll
            for (int index = 0; index < src_num; index++){
                int src_nid = src_edges[edge_index+index];
                src_id[threadIdx.x*neighbor_num + index] = src_nid;
                src_norm_deg[threadIdx.x*neighbor_num + index] = src_norm_degs[src_nid]; 
            }
            partial_output[threadIdx.x*blockDim.y + threadIdx.y] = 0.0f;
            __syncwarp();

            #pragma unroll
            for (int src = 0; src < src_num; src += 1){
                int nid = src_id[threadIdx.x*neighbor_num + src];
                float degree_norm_inv = __fmaf_rn(dst_norm_deg, 1/sqrtf(src_norm_deg[threadIdx.x*neighbor_num + src]), 0);
                // use the partial sum stored in the shared memory to increase the overall bandwidth and the achievable gpu performance
                partial_output[threadIdx.x*blockDim.y + threadIdx.y] += __fmaf_rn(degree_norm_inv, input_feat[nid][feat_id], 0);
            }
            output[current_dst_id][feat_id] = partial_output[threadIdx.x*blockDim.y + threadIdx.y];
        }   
    }
}

std::vector<torch::Tensor> computation_aware_forward_gin_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor src_edges,
    torch::Tensor input_feat,
    torch::Tensor weight,
    int neighbor_num,
    int dst_nodes
)
{   


    // float mm_end_time = clock();
    int dim = input_feat.size(1);
    auto tmp = torch::zeros({dst_nodes, dim}, torch::kCUDA);
    int shared_memory = BLOCK_X * (neighbor_num + 1)*sizeof(int) + BLOCK_X * 128 * sizeof(float);
    const dim3 threads(BLOCK_X, 128);
    const dim3 blocks((dst_nodes+threads.x-1)/threads.x, (dim+threads.y-1)/threads.y);

    // float agg_start_time = clock();
    
    // const int threadsPerBlock = BLOCK_SIZE;
    // const int numBlocks = dst_nodes;
    AT_DISPATCH_FLOATING_TYPES(tmp.type(), "computation_aware_forward_gin_cuda_kernel", ([&] {
                                    computation_aware_forward_gin_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        edge_ptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        input_feat.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        tmp.packed_accessor32<float,2,torch::RestrictPtrTraits>(), 
                                        neighbor_num,
                                        dim,
                                        dst_nodes
                                    );
                                }));
    // float agg_end_time = clock();

    auto output = torch::mm(tmp, weight);


    // printf("mm time:%f;agg time:%f\n",(mm_end_time-mm_start_time),(agg_end_time-agg_start_time));
    return {output, tmp};
}

template <typename scalar_t>
__global__ void computation_aware_forward_gin_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> output,
                    int neighbor_num,
                    int dim,
                    int dst_nodes
)
{
    
    
    int current_dst_id = blockIdx.x * blockDim.x + threadIdx.x;

    int feat_id = blockIdx.y * blockDim.y + threadIdx.y;

    if((current_dst_id>=dst_nodes) || (feat_id>=dim)){
        return;
    }
    
    
    // 使用shmem
    extern __shared__ int shared_space[];
    int *src_id = shared_space;
    float *partial_output = (float*)&src_id[blockDim.x*neighbor_num];

    int src_num = edge_ptr[current_dst_id + 1] - edge_ptr[current_dst_id];
    
    if(current_dst_id<dst_nodes){
        if(src_num==0){
            output[current_dst_id][feat_id] = input_feat[current_dst_id][feat_id];
        }
        else{
            auto edge_index = edge_ptr[current_dst_id];
            #pragma unroll
            for (int index = 0; index < src_num; index++){
                int src_nid = src_edges[edge_index+index];
                src_id[threadIdx.x*neighbor_num + index] = src_nid;
            }
            partial_output[threadIdx.x*blockDim.y + threadIdx.y] = 0.0f;
            __syncwarp();

            #pragma unroll
            for (int src = 0; src < src_num; src += 1){
                int nid = src_id[threadIdx.x*neighbor_num + src];
                partial_output[threadIdx.x*blockDim.y + threadIdx.y] += input_feat[nid][feat_id];
                
            }
            output[current_dst_id][feat_id] = partial_output[threadIdx.x*blockDim.y + threadIdx.y];
        }   
    }
}


std::vector<torch::Tensor> computation_aware_backward_gcn_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor src_edges,
    torch::Tensor src_norm_degs,
    torch::Tensor dst_norm_degs,
    torch::Tensor d_input,
    torch::Tensor weight,
    torch::Tensor input_feat
)
{   
    int src_nodes = input_feat.size(0);
    int dst_nodes = d_input.size(0);
    int dim = d_input.size(1);
    auto d_output = torch::zeros({src_nodes, dim}, torch::kCUDA);
    int shared_memory = 128 * sizeof(float);
    const dim3 threads(128);
    const dim3 blocks(dst_nodes,(dim+threads.x-1)/threads.x);
    
    // const int threadsPerBlock = BLOCK_SIZE;
    // const int numBlocks = dst_nodes;
    AT_DISPATCH_FLOATING_TYPES(d_input.type(), "computation_aware_backward_cuda", ([&] {
                                    computation_aware_backward_gcn_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        edge_ptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_norm_degs.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        dst_norm_degs.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        d_input.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        d_output.packed_accessor32<float,2,torch::RestrictPtrTraits>(), 
                                        dim,
                                        dst_nodes
                                    );
                                }));
    auto d_x = torch::mm(d_output, weight.transpose(0,1));
    auto d_weight = torch::mm(input_feat.transpose(0,1), d_output);
    return {d_x,d_weight};
}

template <typename scalar_t>
__global__ void computation_aware_backward_gcn_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_norm_degs,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_norm_degs,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_input,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_output,
                    int dim,
                    int dst_nodes
)
{
    
    
    int current_src_id = blockIdx.x;

    int feat_id = threadIdx.x + blockIdx.y*blockDim.x;
    int local_feat_id = threadIdx.x;

    if((current_src_id>=dst_nodes) || (feat_id>=dim)){
        return;
    }


    float src_norm_deg;
    extern __shared__ float src_feat[];

    src_norm_deg = 1/sqrtf(src_norm_degs[current_src_id]);
    int dst_num = edge_ptr[current_src_id + 1] - edge_ptr[current_src_id];
    
    if(current_src_id<dst_nodes){
        if(dst_num==0){
            ;
        }
        else{
            auto edge_index = edge_ptr[current_src_id]; 
            
            src_feat[local_feat_id] = d_input[current_src_id][feat_id];
            __syncwarp();
            #pragma unroll
            for (int dst = 0; dst < dst_num; dst += 1){
                int nid = src_edges[edge_index+dst];
                float degree_norm_inv = __fmaf_rn(src_norm_deg, 1/sqrtf(dst_norm_degs[src_edges[edge_index+dst]]), 0);
                
                float partial_sum = __fmaf_rn(degree_norm_inv, src_feat[local_feat_id], 0);
                atomicAdd_F((float*)&d_output[nid][feat_id], partial_sum);
            }
            // __syncthreads();
        }   
    }
}

std::vector<torch::Tensor> computation_aware_backward_gin_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor src_edges,
    torch::Tensor d_input,
    torch::Tensor weight,
    torch::Tensor input_feat,
    int src_nodes
)
{   
    auto d_tmp = torch::mm(d_input, weight.transpose(0,1));
    auto d_weight = torch::mm(input_feat.transpose(0,1), d_input);

    int dst_nodes = d_input.size(0);
    int dim = d_tmp.size(1);
    auto d_x = torch::zeros({src_nodes, dim}, torch::kCUDA);
    // int shared_memory = ((dim>128)? dim:128) * sizeof(float);
    int shared_memory = 128 * sizeof(float);
    const dim3 threads(128);
    const dim3 blocks(dst_nodes,(dim+threads.x-1)/threads.x);
    
    // const int threadsPerBlock = BLOCK_SIZE;
    // const int numBlocks = dst_nodes;
    AT_DISPATCH_FLOATING_TYPES(d_input.type(), "computation_aware_backward_gin_cuda_kernel", ([&] {
                                    computation_aware_backward_gin_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        edge_ptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        d_tmp.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        d_x.packed_accessor32<float,2,torch::RestrictPtrTraits>(), 
                                        dim,
                                        dst_nodes
                                    );
                                }));
    return {d_x,d_weight};
}

template <typename scalar_t>
__global__ void computation_aware_backward_gin_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_input,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_output,
                    int dim,
                    int dst_nodes
)
{
    
    
    int current_src_id = blockIdx.x;

    int local_feat_id = threadIdx.x;
    int feat_id = threadIdx.x + blockIdx.y * blockDim.x;

    if((current_src_id>=dst_nodes) || (feat_id>=dim)){
        return;
    }

    // float src_norm_deg;
    extern __shared__ float src_feat[];

    // src_norm_deg = 1/sqrtf(src_norm_degs[current_src_id]);
    int dst_num = edge_ptr[current_src_id + 1] - edge_ptr[current_src_id];
    
    if(current_src_id<dst_nodes){
        if(dst_num==0){
            ;
        }
        else{
            auto edge_index = edge_ptr[current_src_id]; 
            src_feat[local_feat_id] = d_input[current_src_id][feat_id];
            __syncwarp();
            #pragma unroll
            for (int dst = 0; dst < dst_num; dst += 1){
                int nid = src_edges[edge_index+dst];
                atomicAdd((float*)&d_output[nid][feat_id], src_feat[local_feat_id]);
            }
        }   
    }
}

template <typename scalar_t>
__global__ void computation_aware_backward_gin_debug_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_input,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_output,
                    int dim,
                    int dst_nodes
)
{
    
    
    int current_src_id = blockIdx.x;

    int feat_id = threadIdx.x;

    if((current_src_id>=dst_nodes) || (feat_id>=dim)){
        return;
    }

    // float src_norm_deg;
    extern __shared__ float src_feat[];

    // src_norm_deg = 1/sqrtf(src_norm_degs[current_src_id]);
    int dst_num = edge_ptr[current_src_id + 1] - edge_ptr[current_src_id];
    
    if(current_src_id<dst_nodes){
        if(dst_num==0){
            ;
        }
        else{
            auto edge_index = edge_ptr[current_src_id]; //这句没问题
            #pragma unroll
            for (int d = feat_id; d<dim; d+=blockDim.x){
                src_feat[d] = d_input[current_src_id][d];
            }
            __syncwarp();
            #pragma unroll
            for (int dst = 0; dst < dst_num; dst += 1){
                int nid = src_edges[edge_index+dst];
                #pragma unroll
                for (int d = feat_id; d < dim; d += blockDim.x){
                    atomicAdd_F((float*)&d_output[nid][d], src_feat[d]);
                }
            }
        }   
    }
}

std::vector<torch::Tensor> computation_aware_backward_gin_debug_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor src_edges,
    torch::Tensor d_input,
    torch::Tensor weight,
    torch::Tensor input_feat,
    int src_nodes
)
{   
    auto d_tmp = torch::mm(d_input, weight.transpose(0,1));
    auto d_weight = torch::mm(input_feat.transpose(0,1), d_input);

    int dst_nodes = d_input.size(0);
    int dim = d_tmp.size(1);
    auto d_x = torch::zeros({src_nodes, dim}, torch::kCUDA);
    int shared_memory = ((dim>128)? dim:128) * sizeof(float);
    const dim3 threads(128);
    const dim3 blocks(dst_nodes);
    
    // const int threadsPerBlock = BLOCK_SIZE;
    // const int numBlocks = dst_nodes;
    AT_DISPATCH_FLOATING_TYPES(d_input.type(), "computation_aware_backward_gin_debug_cuda_kernel", ([&] {
                                    computation_aware_backward_gin_debug_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        edge_ptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        d_tmp.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        d_x.packed_accessor32<float,2,torch::RestrictPtrTraits>(), 
                                        dim,
                                        dst_nodes
                                    );
                                }));
    return {d_x,d_weight};
}


std::vector<torch::Tensor> aggregate_forward_outer_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor dst_edges,
    torch::Tensor src_norm_degs,
    torch::Tensor dst_norm_degs,
    torch::Tensor input_feat,
    torch::Tensor weight,
    int max_degree,
    int dst_nodes,
    int src_nodes
)
{   
    auto tmp = torch::mm(input_feat, weight);
    int dim = tmp.size(1);
    int edge_num = dst_edges.size(0);
    auto output = torch::zeros({dst_nodes, tmp.size(1)}, torch::kCUDA);
    int shared_memory = max_degree*sizeof(float) + (max_degree) *sizeof(int) + (dim + 1) * sizeof(float) ;
    const dim3 threads(128);
    const dim3 blocks(src_nodes);
    
    // const int threadsPerBlock = BLOCK_SIZE;
    // const int numBlocks = dst_nodes;
    AT_DISPATCH_FLOATING_TYPES(tmp.type(), "aggregate_forward_cuda", ([&] {
                                    aggregate_forward_outer_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        edge_ptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        dst_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_norm_degs.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        dst_norm_degs.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        tmp.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        output.packed_accessor32<float,2,torch::RestrictPtrTraits>(), 
                                        max_degree,
                                        dim,
                                        src_nodes
                                    );
                                }));
    return {output};
}

template <typename scalar_t>
__global__ void aggregate_forward_outer_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_edges,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_norm_degs,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_norm_degs,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> output,
                    int max_degree,
                    int dim,
                    int src_nodes
)
{
    
    
    int current_src_id = blockIdx.x;

    int feat_id = threadIdx.x;

    if((current_src_id>=src_nodes) || (feat_id>=dim)){
        return;
    }

    
    
    extern __shared__ int shared_space[];
    int *dst_id = shared_space;
    float *dst_norm_deg = (float*)&shared_space[max_degree];
    __shared__ float src_norm_deg;
    extern __shared__ float src_feat[];

    src_norm_deg = 1/sqrtf(src_norm_degs[current_src_id]);
    int dst_num = edge_ptr[current_src_id + 1] - edge_ptr[current_src_id];
    
    if(current_src_id<src_nodes){
        if(dst_num==0){
            ;
        }
        else{
            auto edge_index = edge_ptr[current_src_id];
            #pragma unroll
            for (int index = 0; index < dst_num; index++){
                if(edge_index+index>dst_edges.size(0)){
                    printf("dst_edge max:%d, dst_edge index:%d\n",dst_edges.size(0), edge_index+index);
                }
                if(index>=max_degree){
                    printf("dst id index:%d max degree:%d\n",index,max_degree);}
                if(dst_edges[edge_index+index]>dst_norm_degs.size(0)){
                    printf("dst_nid max:%d, dst_nid index:%d\n",dst_norm_degs.size(0), dst_edges[edge_index+index]);
                }
                dst_norm_deg[index] = 1/sqrtf(dst_norm_degs[dst_edges[edge_index+index]]); 
                for (int d = feat_id; d<dim; d+=blockDim.x){
                    src_feat[d] = input_feat[current_src_id][d];
                }
            }
            __syncwarp();
            #pragma unroll
            for (int dst = 0; dst < dst_num; dst += 1){
                int nid = dst_edges[edge_index+dst];
                float degree_norm_inv = __fmaf_rn(src_norm_deg, dst_norm_deg[dst], 0);
                for (int d = feat_id; d < dim; d += blockDim.x){
                    float partial_sum = __fmaf_rn(degree_norm_inv, src_feat[d], 0);
                    atomicAdd_F((float*)&output[nid][d], partial_sum);
                }
            }
        }   
    }
}

std::vector<torch::Tensor> computation_aware_forward_gat_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor edge_data,
    torch::Tensor src_edges,
    torch::Tensor input_feat,
    int neighbor_num,
    int dst_nodes,
    int num_heads
)
{   

    int block_dim_x = 8;
    int dim = input_feat.size(1);
    int unit_dim = dim/num_heads;
    auto output = torch::zeros({dst_nodes, dim}, torch::kCUDA);
    int shared_memory = block_dim_x*neighbor_num*sizeof(int) + block_dim_x * BLOCK_Y * sizeof(float);
    // num_heads*(BLOCK_Y_GAT * (neighbor_num + 1)*sizeof(int) + BLOCK_Y_GAT * neighbor_num * sizeof(int) + BLOCK_Y_GAT * BLOCK_Z_GAT * sizeof(float));
    const dim3 threads(block_dim_x, BLOCK_Y);
    const dim3 blocks((dst_nodes+threads.x-1)/threads.x, (dim+threads.y-1)/threads.y);


    // float agg_start_time = clock();
    
    // const int threadsPerBlock = BLOCK_SIZE;
    // const int numBlocks = dst_nodes;
    AT_DISPATCH_FLOATING_TYPES(input_feat.type(), "computation_aware_forward_gat_cuda_kernel", ([&] {
                                    computation_aware_forward_gat_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        edge_ptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        edge_data.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        input_feat.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        output.packed_accessor32<float,2,torch::RestrictPtrTraits>(), 
                                        neighbor_num,
                                        dim,
                                        unit_dim,
                                        dst_nodes,
                                        num_heads
                                    );
                                }));
    // kernel<<<grids, blocks>>>();
    // float agg_end_time = clock();

    // printf("mm time:%f;agg time:%f\n",(mm_end_time-mm_start_time),(agg_end_time-agg_start_time));
    return {output};
}

template <typename scalar_t>
__global__ void computation_aware_forward_gat_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> edge_data,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> output,
                    int neighbor_num,
                    int dim,
                    int unit_dim,
                    int dst_nodes,
                    int num_heads
)
{
    
    
    int current_dst_id = blockIdx.x * blockDim.x + threadIdx.x;

    int feat_id = blockIdx.y * blockDim.y + threadIdx.y;

    const int head_id = feat_id/unit_dim;

    if((current_dst_id>=dst_nodes) || (feat_id>=dim)){
        return;
    }
    
    
    extern __shared__ int shared_space[];
    int *src_id = shared_space;
    float *partial_output = (float*)&shared_space[blockDim.x*neighbor_num];

    int src_num = edge_ptr[current_dst_id + 1] - edge_ptr[current_dst_id];
    
    if(current_dst_id<dst_nodes){
        if(src_num==0){
            output[current_dst_id][feat_id] = input_feat[current_dst_id][feat_id];
        }
        else{
            auto edge_index = edge_ptr[current_dst_id];
            #pragma unroll
            partial_output[threadIdx.x*blockDim.y + threadIdx.y] = 0.0f;
            __syncwarp();
            #pragma unroll
            for (int src = 0; src < src_num; src += 1){
                int nid = src_edges[edge_index+src];
                float alpha = edge_data[edge_index+src][head_id];
                partial_output[threadIdx.x*blockDim.y + threadIdx.y] += __fmul_rn(alpha, input_feat[nid][feat_id]);
            }
            output[current_dst_id][feat_id] = partial_output[threadIdx.x*blockDim.y + threadIdx.y];
        }   
    }
}

std::vector<torch::Tensor> computation_aware_backward_gat_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor edge_data,
    torch::Tensor src_edges,
    torch::Tensor d_input,
    int src_nodes,
    int num_heads
)
{   
    int dst_nodes = d_input.size(0);
    int dim = d_input.size(1);
    int unit_dim = dim/num_heads;
    auto d_output = torch::zeros({src_nodes, dim}, torch::kCUDA);
    int shared_memory = 64 * sizeof(float);
    const dim3 threads(64);
    const dim3 blocks(dst_nodes,(dim+threads.x-1)/threads.x);
    
    // const int threadsPerBlock = BLOCK_SIZE;
    // const int numBlocks = dst_nodes;
    AT_DISPATCH_FLOATING_TYPES(d_input.type(), "computation_aware_backward_gat_cuda_kernel", ([&] {
                                    computation_aware_backward_gat_cuda_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        edge_ptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        edge_data.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        d_input.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        d_output.packed_accessor32<float,2,torch::RestrictPtrTraits>(), 
                                        dim,
                                        unit_dim,
                                        dst_nodes,
                                        num_heads
                                    );
                                }));
    
    return {d_output};
}

template <typename scalar_t>
__global__ void computation_aware_backward_gat_cuda_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> edge_data,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_input,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> d_output,
                    int dim,
                    int unit_dim,
                    int dst_nodes,
                    int num_heads
)
{
    
    

    int current_src_id = blockIdx.x;

    int feat_id = threadIdx.x + blockDim.x * blockIdx.y;
    int local_feat_id = threadIdx.x;


    const int head_id = feat_id/unit_dim;

    if((current_src_id>=dst_nodes) || (feat_id>=dim)){
        return;
    }

    float src_norm_deg;
    extern __shared__ float src_feat[];

    int dst_num = edge_ptr[current_src_id + 1] - edge_ptr[current_src_id];
    
    if(current_src_id<dst_nodes){
        if(dst_num==0){
            ;
        }
        else{
            auto edge_index = edge_ptr[current_src_id]; 
            src_feat[local_feat_id] = d_input[current_src_id][feat_id];
            __syncwarp();
            #pragma unroll
            for (int dst = 0; dst < dst_num; dst += 1){
                int nid = src_edges[edge_index+dst];
                // #pragma unroll
                // for (int d = feat_id; d < dim; d += blockDim.x){
                //     float partial_sum = __fmul_rn(edge_data[edge_index+dst][head_id], src_feat[d]);
                //     atomicAdd((float*)&d_output[nid][d], partial_sum);
                // }
                float partial_sum = __fmul_rn(edge_data[edge_index+dst][head_id], src_feat[local_feat_id]);
                atomicAdd((float*)&d_output[nid][feat_id], partial_sum);
            }
            // __syncthreads();
        }   
    }
}


template <typename scalar_t>
__global__ void computation_aware_forward_gat_cuda_3d_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> edge_data,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> output,
                    int neighbor_num,
                    int dim,
                    int dst_nodes,
                    int num_heads
);

template <typename scalar_t>
__global__ void computation_aware_backward_gat_cuda_3d_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> edge_data,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> d_input,
                    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> d_output,
                    int dim,
                    int dst_nodes,
                    int num_heads
);

std::vector<torch::Tensor> computation_aware_forward_gat_cuda_3d(
    torch::Tensor edge_ptr,
    torch::Tensor edge_data,
    torch::Tensor src_edges,
    torch::Tensor input_feat,
    int neighbor_num,
    int dst_nodes,
    int num_heads
)
{   

    
    int dim = input_feat.size(2);
    // int unit_dim = dim/num_heads;
    auto output = torch::zeros({dst_nodes, num_heads, dim}, torch::kCUDA);
    int shared_memory = BLOCK_X_GAT*neighbor_num*sizeof(int) + BLOCK_X_GAT * BLOCK_Z_GAT * num_heads * sizeof(float);
    // num_heads*(BLOCK_Y_GAT * (neighbor_num + 1)*sizeof(int) + BLOCK_Y_GAT * neighbor_num * sizeof(int) + BLOCK_Y_GAT * BLOCK_Z_GAT * sizeof(float));
    const dim3 threads(BLOCK_X_GAT, num_heads, BLOCK_Z_GAT);
    const dim3 blocks((dst_nodes+threads.x-1)/threads.x, 1, (dim+threads.z-1)/threads.z);


    // float agg_start_time = clock();
    
    // const int threadsPerBlock = BLOCK_SIZE;
    // const int numBlocks = dst_nodes;
    AT_DISPATCH_FLOATING_TYPES(input_feat.type(), "computation_aware_forward_gat_cuda_3d_kernel", ([&] {
                                    computation_aware_forward_gat_cuda_3d_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        edge_ptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        edge_data.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        input_feat.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                                        output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), 
                                        neighbor_num,
                                        dim,
                                        dst_nodes,
                                        num_heads
                                    );
                                }));
    // kernel<<<grids, blocks>>>();
    // float agg_end_time = clock();

    // printf("mm time:%f;agg time:%f\n",(mm_end_time-mm_start_time),(agg_end_time-agg_start_time));
    return {output};
}

template <typename scalar_t>
__global__ void computation_aware_forward_gat_cuda_3d_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> edge_data,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> input_feat,
                    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> output,
                    int neighbor_num,
                    int dim,
                    int dst_nodes,
                    int num_heads
)
{
    
    
    int current_dst_id = blockIdx.x * blockDim.x + threadIdx.x;

    int feat_id = blockIdx.z * blockDim.z + threadIdx.z;

    // head id
    int head_id = threadIdx.y;

    if((head_id>=num_heads) || (current_dst_id>=dst_nodes) || (feat_id>=dim)){
        return;
    }
    
    
   
    extern __shared__ int shared_space[];
    int *src_id = shared_space;
    float *partial_output = (float*)&shared_space[blockDim.x*neighbor_num];

    int src_num = edge_ptr[current_dst_id + 1] - edge_ptr[current_dst_id];
    
    if(current_dst_id<dst_nodes){
        if(src_num==0){
            output[current_dst_id][head_id][feat_id] = input_feat[current_dst_id][head_id][feat_id];
        }
        else{
            auto edge_index = edge_ptr[current_dst_id];
            #pragma unroll
            for (int index = 0; index < src_num; index++){
                int src_nid = src_edges[edge_index+index];
                src_id[threadIdx.y*neighbor_num + index] = src_nid;
            }
            partial_output[threadIdx.x * blockDim.y * blockDim.z + threadIdx.y*blockDim.z + threadIdx.z] = 0.0f;
            __syncwarp();
            #pragma unroll
            for (int src = 0; src < src_num; src += 1){
                int nid = src_id[threadIdx.y*neighbor_num + src];
                float alpha = edge_data[edge_index+src][head_id];
                partial_output[threadIdx.x * blockDim.y * blockDim.z + threadIdx.y*blockDim.z + threadIdx.z] += __fmul_rn(alpha, input_feat[nid][head_id][feat_id]);
            }
            output[current_dst_id][head_id][feat_id] = partial_output[threadIdx.x * blockDim.y * blockDim.z + threadIdx.y*blockDim.z + threadIdx.z];
        }   
    }
}

std::vector<torch::Tensor> computation_aware_backward_gat_cuda_3d(
    torch::Tensor edge_ptr,
    torch::Tensor edge_data,
    torch::Tensor src_edges,
    torch::Tensor d_input,
    int src_nodes,
    int num_heads
)
{   
    int dst_nodes = d_input.size(0);
    int dim = d_input.size(2);
    // int unit_dim = dim/num_heads;
    auto d_output = torch::zeros({src_nodes, num_heads, dim}, torch::kCUDA);
    int shared_memory = num_heads * 8 * sizeof(float);
    const dim3 threads(num_heads, 8);
    const dim3 blocks(dst_nodes);
    
    // const int threadsPerBlock = BLOCK_SIZE;
    // const int numBlocks = dst_nodes;
    AT_DISPATCH_FLOATING_TYPES(d_input.type(), "computation_aware_backward_gat_cuda_3d_kernel", ([&] {
                                    computation_aware_backward_gat_cuda_3d_kernel<scalar_t><<<blocks, threads, shared_memory>>>(
                                        edge_ptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        edge_data.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        d_input.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                                        d_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), 
                                        dim,
                                        dst_nodes,
                                        num_heads
                                    );
                                }));
    
    return {d_output};
}

template <typename scalar_t>
__global__ void computation_aware_backward_gat_cuda_3d_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
                    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> edge_data,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> d_input,
                    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> d_output,
                    int dim,
                    int dst_nodes,
                    int num_heads
)
{
    
    

    int current_src_id = blockIdx.x;

    int feat_id = threadIdx.y;

    int head_id = threadIdx.x;

    if((head_id>=num_heads) || (current_src_id>=dst_nodes) || (feat_id>=dim)){
        return;
    }

    // printf("current_dst_id:%d;feat_id:%d\n",current_dst_id,feat_id);
    
    
    
    float src_norm_deg;
    extern __shared__ float src_feat[];

    int dst_num = edge_ptr[current_src_id + 1] - edge_ptr[current_src_id];
    
    if(current_src_id<dst_nodes){
        if(dst_num==0){
            ;
        }
        else{
            auto edge_index = edge_ptr[current_src_id]; 
            #pragma unroll
            for (int d = feat_id; d<dim; d+=blockDim.x){
                src_feat[threadIdx.x*blockDim.x + d] = d_input[current_src_id][head_id][d];
            }
            __syncwarp();
            #pragma unroll
            for (int dst = 0; dst < dst_num; dst += 1){
                int nid = src_edges[edge_index+dst];
                #pragma unroll
                for (int d = feat_id; d < dim; d += blockDim.x){
                    float partial_sum = __fmul_rn(edge_data[edge_index+dst][head_id], src_feat[d]);
                    atomicAdd((float*)&d_output[nid][head_id][d], partial_sum);
                }
            }
        }   
    }
}

