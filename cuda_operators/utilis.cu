//*****
// some utilis function maybe used
//*****

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <time.h>

constexpr size_t BLOCK_SIZE = 128;


at::Tensor exclusive_sum_cuda(
    torch::Tensor in_degs
){
    // auto in_degs_pad = torch::zeros({1}, torch::kCUDA).to(at::kInt);
    auto in_degs_pad = torch::cat({in_degs,torch::zeros({1}, torch::kCUDA).to(at::kInt)});
    int num_items = in_degs_pad.size(0);
    int* input_data = in_degs_pad.data<int>();

    auto edge_ptr = torch::zeros({num_items}, torch::kCUDA).to(at::kInt);

    int* output_ptr = edge_ptr.data<int>();

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input_data, output_ptr, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input_data, output_ptr, num_items);

    // cudaFree(input_data);
    cudaFree(d_temp_storage);

    return edge_ptr;
}

__global__ void cal_deg_cuda_kernel(
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> edges,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> deg,
    int edge_num

){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=edge_num){
        return;
    }

    int id = edges[tid];

    atomicAdd((int*)&deg[id],1);
}

at::Tensor cal_deg_cuda(
    torch::Tensor edges,
    int nodes_num
){
    int edge_num = edges.size(0);
    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((edge_num+threads.x-1)/threads.x);
    int device = edges.get_device();
    auto deg = torch::zeros({nodes_num}).to(at::Device(at::kCUDA, device)).to(at::kInt);

    AT_DISPATCH_ALL_TYPES(deg.type(), "cal_deg_cuda_kernel", ([&] {
                                    cal_deg_cuda_kernel<<<blocks, threads>>>(
                                        edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        deg.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        edge_num
                                    );
                                }));

    return deg;
}

__global__ void reset_false_kernel(
    bool* flag_ptr,
    int flag_size
){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=flag_size){
        return;
    }

    flag_ptr[tid] = false;
}

__global__ void assign_new_flag_kernel(
    bool* flag_ptr,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> input_nodes,
    int size
){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=size){
        return;
    }

    flag_ptr[input_nodes[tid]] = true;
}

at::Tensor reset_flag_cuda(
    torch::Tensor flag,
    torch::Tensor input_nodes
){
    int size = input_nodes.size(0);
    int flag_size = flag.size(0);
    bool* flag_ptr = flag.data_ptr<bool>();
    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks(flag_size);

    AT_DISPATCH_ALL_TYPES(input_nodes.type(), "reset_false_kernel", ([&] {
                                    reset_false_kernel<<<blocks, threads>>>(
                                        flag_ptr,
                                        flag_size
                                    );
                                }));

    const dim3 blocks_1(size);

    AT_DISPATCH_ALL_TYPES(input_nodes.type(), "assign_new_flag_kernel", ([&] {
                                    assign_new_flag_kernel<<<blocks_1, threads>>>(
                                        flag_ptr,
                                        input_nodes.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        size
                                    );
                                }));

    return flag;

}


__global__ void reset_id_map_kernel(
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> localid2cacheid,
    int flag_size
){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=flag_size){
        return;
    }

    localid2cacheid[tid] = 0;
}

__global__ void assign_new_id_map_kernel(
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> localid2cacheid,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> input_nodes,
    int size
){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=size){
        return;
    }

    localid2cacheid[input_nodes[tid]] = tid;
}

at::Tensor reset_localid2cacheid_cuda(
    torch::Tensor localid2cacheid,
    torch::Tensor input_nodes
){
    int size = input_nodes.size(0);
    int flag_size = localid2cacheid.size(0);
    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks(flag_size);

    AT_DISPATCH_ALL_TYPES(localid2cacheid.type(), "reset_id_map_kernel", ([&] {
                                    reset_id_map_kernel<<<blocks, threads>>>(
                                        localid2cacheid.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        flag_size
                                    );
                                }));

    const dim3 blocks_1(size);

    AT_DISPATCH_ALL_TYPES(localid2cacheid.type(), "assign_new_id_map_kernel", ([&] {
                                    assign_new_id_map_kernel<<<blocks_1, threads>>>(
                                        localid2cacheid.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        input_nodes.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        size
                                    );
                                }));

    return localid2cacheid;

}

__global__ void fetch_data_kernel(
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> data,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> out_data,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> index,
    int size
){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=size){
        return;
    }

    out_data[tid] = data[index[tid]];
}

at::Tensor fetch_data_cuda(
    torch::Tensor data,
    torch::Tensor index
){
    int size = index.size(0);

    auto out_data = torch::zeros({size, data.size(1)},torch::kCUDA);

    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks(size);

    AT_DISPATCH_FLOATING_TYPES(out_data.type(), "fetch_data_kernel", ([&] {
                                    fetch_data_kernel<<<blocks, threads>>>(
                                        data.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        out_data.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        size
                                    );
                                }));
    return out_data;
}

__global__ void filter_same_index_kernel(
    bool* gpu_flag,
    bool* batch_flag,
    bool* out_flag,
    int size
){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid>=size){
        return;
    }

    if((!batch_flag[tid])&&(gpu_flag[tid])){
        out_flag[tid] = true;
    } 
}


at::Tensor filter_same_index_cuda(
    torch::Tensor gpu_flag,
    torch::Tensor batch_flag
){
    int size = gpu_flag.size(0);
    auto out_flag = torch::zeros({size},torch::kCUDA).to(at::kBool);
    bool* flag_ptr = out_flag.data_ptr<bool>();
    bool* gpu_flag_ptr = gpu_flag.data_ptr<bool>();
    bool* batch_flag_ptr = batch_flag.data_ptr<bool>();

    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks(size);

    filter_same_index_kernel<<<blocks, threads>>>(
                                        gpu_flag_ptr,
                                        batch_flag_ptr,
                                        flag_ptr,
                                        size
                                    );
    return out_flag;

    // AT_DISPATCH_FLOATING_TYPES(size.type(), "filter_same_index_kernel", ([&] {
    //                                 filter_same_index_kernel<<<blocks, threads>>>(
    //                                     gpu_flag_ptr,
    //                                     batch_flag_ptr,
    //                                     flag_ptr,
    //                                     size
    //                                 );
    //                             }));
}

__global__ void apply_edge_cuda_kernel(
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> el,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> er,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_edges,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> alpha,
    int edge_num,
    int num_heads
){
    int edge_id = threadIdx.x + blockIdx.x * blockDim.x;

    // int feat_id = threadIdx.y + blockIdx.y * blockDim.y;

    if((edge_id>=edge_num)){
        return;
    }
    #pragma unroll
    for (int i=0;i<num_heads;i++){
        alpha[edge_id][i] = el[src_edges[edge_id]][i] + er[dst_edges[edge_id]][i];
    }
    // alpha[edge_id][feat_id] = el[src_edges[edge_id]][feat_id] + er[dst_edges[edge_id]][feat_id];
    

}

at::Tensor apply_edge_cuda(
    torch::Tensor el,
    torch::Tensor er,
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    int num_heads
){
    int edge_num = src_edges.size(0);

    auto alpha = torch::zeros({edge_num, num_heads},torch::kCUDA);

    const dim3 threads(128);
    const dim3 blocks((edge_num+threads.x-1)/threads.x);

    AT_DISPATCH_FLOATING_TYPES(alpha.type(), "apply_edge_cuda_kernel", ([&] {
                                    apply_edge_cuda_kernel<<<blocks, threads>>>(
                                        el.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        er.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        dst_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        alpha.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                                        edge_num,
                                        num_heads
                                    );
                                }));
    return alpha;
}