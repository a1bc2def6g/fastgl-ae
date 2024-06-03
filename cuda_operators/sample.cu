#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <time.h>

constexpr size_t BLOCK_SIZE = 64;
constexpr size_t MAX_DEG = 64;

#define min(a,b)            (((a) < (b)) ? (a) : (b))


// calculate the number of the edges that each node should sample
__global__ void generate_edge_num_kernel(
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> seed_nodes,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
    int* out_deg,
    int neighbor_num,
    int sample_nodes
) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>sample_nodes){
        return;
    }

    const int in_row = seed_nodes[tid];
    const int out_row = tid;

    if (tid < sample_nodes) {
        out_deg[out_row] = min(neighbor_num, edge_ptr[in_row + 1] - edge_ptr[in_row]);
    }else{
        out_deg[sample_nodes] = 0;
    }
}

// sampling
__global__ void sample_edges_kernel(
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_edges,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edge_ptr,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> seed_nodes,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges_out,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_edges_out,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edges_num,
    int* edge_start_index,
    int sample_nodes,
    int neighbor_num

){
    edges_num[0] = edge_start_index[sample_nodes];

    // return;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int n_id = blockIdx.x;

    
    if(n_id>=sample_nodes){
        return;
    }


    // sample random seed
    curandStatePhilox4_32_10_t state;
    curand_init(tid, tid, 0, &state);

    // each block is responsible for sampling the neighbors of one node

    const int sample_node_id = seed_nodes[n_id];

    const int start_index = edge_start_index[n_id];

    const int edge_num = edge_ptr[sample_node_id+1] - edge_ptr[sample_node_id];

    const int edge_index = edge_ptr[sample_node_id];

    if(edge_num<=neighbor_num){
        // #pragma unroll
        for(int i=threadIdx.x; i<edge_num; i=i+BLOCK_SIZE){
            src_edges_out[start_index+i] = src_edges[edge_index+i];
            dst_edges_out[start_index+i] = sample_node_id;
        }
    }else{
        // #pragma unroll
        for(int i=threadIdx.x; i<neighbor_num; i=i+BLOCK_SIZE){
            const int offset = curand(&state) % (edge_num);
            src_edges_out[start_index+i] = src_edges[edge_index+offset];
            dst_edges_out[start_index+i] = sample_node_id;
        }
    }

    return;

}


std::vector<torch::Tensor> sample_node_cuda(
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    torch::Tensor edge_ptr,
    torch::Tensor seed_nodes,
    int dst_nodes,
    int neighbor_num
)
{
    int sample_nodes = seed_nodes.size(0);
    int num_nodes = edge_ptr.size(0) - 1;

    auto src_edges_out = torch::zeros({dst_nodes*neighbor_num}, torch::kCUDA).to(at::kInt);
    auto dst_edges_out = torch::zeros({dst_nodes*neighbor_num}, torch::kCUDA).to(at::kInt);

    int* out_deg;
    cudaMalloc((void**)&out_deg, (sample_nodes+1) * sizeof(int));
    // auto out_deg = torch::zeros({sample_nodes,torch::kCUDA}).to(at::kInt);

    const dim3 threads(128);
    const dim3 blocks((sample_nodes+threads.x)/threads.x);


    int shared_memory = 4*sizeof(int) + MAX_DEG * sizeof(int);

    // float gener_start_time = clock();

    AT_DISPATCH_ALL_TYPES(src_edges_out.type(), "generate_edge_num_kernel", ([&] {
                                    generate_edge_num_kernel<<<blocks, threads>>>(
                                        seed_nodes.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        edge_ptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        out_deg,
                                        neighbor_num,
                                        sample_nodes
                                    );
                                }));

    // float gener_end_time = clock();


    // float generate_edge_num_time = gener_end_time - gener_start_time;


    // float exclu_start_time = clock();
    
    // based on the number of edges that should be sampled out of each sampling node, 
    // calculate the index of each sampling node's edge at the start of edge_out
    int  *edge_start_index;

    cudaMalloc((void**)&edge_start_index, (sample_nodes+1) * sizeof(int));

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, out_deg, edge_start_index, sample_nodes+1);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, out_deg, edge_start_index, sample_nodes+1);

    cudaFree(out_deg);
    cudaFree(d_temp_storage);

    // float exclu_end_time = clock();

    // float exclusive_sum_time = exclu_end_time - exclu_start_time;

    // float read_start_time = clock();

    auto edges_num = torch::zeros({1},torch::kCUDA).to(at::kInt);

    const dim3 blocks_1(sample_nodes);

    AT_DISPATCH_ALL_TYPES(src_edges_out.type(), "sample_edges_kernel", ([&] {
                                    sample_edges_kernel<<<blocks_1, threads>>>(
                                        src_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        dst_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        edge_ptr.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        seed_nodes.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_edges_out.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        dst_edges_out.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        edges_num.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        edge_start_index,
                                        // NULL,
                                        sample_nodes,
                                        neighbor_num
                                    );
                                }));
    // cudaEventRecord(stop_2);

    // cudaEventSynchronize(stop_2);
    // float read_end_time = clock();


    // float read_edge_time = read_end_time - read_start_time;

    cudaFree(edge_start_index);
    // cudaEventElapsedTime(&read_edge_time, start_2, stop_2);

    // printf("generate time:%f,exclusive time:%f,read edge time:%f\n",generate_edge_num_time,exclusive_sum_time,read_edge_time);

    return {torch::slice(src_edges_out, 0, 0, edges_num[0].item<int>()),torch::slice(dst_edges_out, 0, 0, edges_num[0].item<int>()),edges_num};
}