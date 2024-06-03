#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> aggregate_forward_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor src_edges,
    torch::Tensor src_norm_degs,
    torch::Tensor dst_norm_degs,
    torch::Tensor input_feat,
    torch::Tensor weight,
    int neighbor_num,
    int dst_nodes
);

std::vector<torch::Tensor> computation_aware_forward_gcn_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor src_edges,
    torch::Tensor src_norm_degs,
    torch::Tensor dst_norm_degs,
    torch::Tensor input_feat,
    torch::Tensor weight,
    int neighbor_num,
    int dst_nodes
);

std::vector<torch::Tensor> computation_aware_backward_gcn_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor src_edges,
    torch::Tensor src_norm_degs,
    torch::Tensor dst_norm_degs,
    torch::Tensor d_input,
    torch::Tensor weight,
    torch::Tensor input_feat
);

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
);

std::vector<torch::Tensor> transfer_edge_id_cuda(
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    torch::Tensor dst_edges_unique,
    int dst_nodes, 
    int total_items,
    int unique_num,
    long int edge_nums
);

std::vector<torch::Tensor> sample_node_cuda(
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    torch::Tensor edge_ptr,
    torch::Tensor seed_nodes,
    int dst_nodes,
    int neighbor_num
);

// std::vector<torch::Tensor> aggregate_backward_reuse_out_gin_debug_cuda(
//     torch::Tensor edge_ptr,
//     torch::Tensor src_edges,
//     torch::Tensor d_input,
//     torch::Tensor weight,
//     torch::Tensor input_feat,
//     int src_nodes
// );

at::Tensor exclusive_sum_cuda(
    torch::Tensor in_degs
);

at::Tensor cal_deg_cuda(
    torch::Tensor edges,
    int nodes_num
);

at::Tensor reset_flag_cuda(
    torch::Tensor flag,
    torch::Tensor input_nodes
);

at::Tensor reset_localid2cacheid_cuda(
    torch::Tensor localid2cacheid,
    torch::Tensor input_nodes
);

at::Tensor fetch_data_cuda(
    torch::Tensor data,
    torch::Tensor index
);

std::vector<torch::Tensor> aggregate_forward(
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
  CHECK_INPUT(edge_ptr);
  CHECK_INPUT(src_edges);
  CHECK_INPUT(src_norm_degs);
  CHECK_INPUT(dst_norm_degs);
  CHECK_INPUT(input_feat);
  CHECK_INPUT(weight);

  return aggregate_forward_cuda(
    edge_ptr,
    src_edges,
    src_norm_degs,
    dst_norm_degs,
    input_feat,
    weight,
    neighbor_num,
    dst_nodes);
}

std::vector<torch::Tensor> computation_aware_forward_gcn(
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
  CHECK_INPUT(edge_ptr);
  CHECK_INPUT(src_edges);
  CHECK_INPUT(src_norm_degs);
  CHECK_INPUT(dst_norm_degs);
  CHECK_INPUT(input_feat);
  CHECK_INPUT(weight);

  return computation_aware_forward_gcn_cuda(
    edge_ptr,
    src_edges,
    src_norm_degs,
    dst_norm_degs,
    input_feat,
    weight,
    neighbor_num,
    dst_nodes);
}

std::vector<torch::Tensor> computation_aware_forward_gin_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor src_edges,
    // torch::Tensor src_norm_degs,
    // torch::Tensor dst_norm_degs,
    torch::Tensor input_feat,
    torch::Tensor weight,
    int neighbor_num,
    int dst_nodes
);

std::vector<torch::Tensor> computation_aware_forward_gin(
    torch::Tensor edge_ptr,
    torch::Tensor src_edges,
    // torch::Tensor src_norm_degs,
    // torch::Tensor dst_norm_degs,
    torch::Tensor input_feat,
    torch::Tensor weight,
    int neighbor_num,
    int dst_nodes
){
    CHECK_INPUT(edge_ptr);
    CHECK_INPUT(src_edges);
    // CHECK_INPUT(src_norm_degs);
    // CHECK_INPUT(dst_norm_degs);
    CHECK_INPUT(input_feat);
    CHECK_INPUT(weight);

    return computation_aware_forward_gin_cuda(
        edge_ptr,
        src_edges,
        // src_norm_degs,
        // dst_norm_degs,
        input_feat,
        weight,
        neighbor_num,
        dst_nodes);
}

std::vector<torch::Tensor> computation_aware_forward_gat_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor edge_data,
    torch::Tensor src_edges,
    torch::Tensor input_feat,
    int neighbor_num,
    int dst_nodes,
    int num_heads
);

std::vector<torch::Tensor> computation_aware_forward_gat(
    torch::Tensor edge_ptr,
    torch::Tensor edge_data,
    torch::Tensor src_edges,
    torch::Tensor input_feat,
    int neighbor_num,
    int dst_nodes,
    int num_heads
){
    CHECK_INPUT(edge_ptr);
    CHECK_INPUT(src_edges);
    CHECK_INPUT(edge_data);
    // CHECK_INPUT(dst_norm_degs);
    CHECK_INPUT(input_feat);
    // CHECK_INPUT(weight);

    return computation_aware_forward_gat_cuda(
        edge_ptr,
        edge_data,
        src_edges,
        // src_norm_degs,
        // dst_norm_degs,
        input_feat,
        neighbor_num,
        dst_nodes,
        num_heads);
}

std::vector<torch::Tensor> computation_aware_backward_gat_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor edge_data,
    torch::Tensor src_edges,
    torch::Tensor d_input,
    int src_nodes,
    int num_heads
);

std::vector<torch::Tensor> computation_aware_backward_gat(
    torch::Tensor edge_ptr,
    torch::Tensor edge_data,
    torch::Tensor src_edges,
    torch::Tensor d_input,
    int src_nodes,
    int num_heads
){
    CHECK_INPUT(edge_ptr);
    CHECK_INPUT(src_edges);
    CHECK_INPUT(edge_data);
    // CHECK_INPUT(src_edges);
    // CHECK_INPUT(dst_norm_degs);
    CHECK_INPUT(d_input);
    // CHECK_INPUT(weight);

    return computation_aware_backward_gat_cuda(
        edge_ptr,
        edge_data,
        src_edges,
        // src_norm_degs,
        // dst_norm_degs,
        d_input,
        src_nodes,
        num_heads);
}

std::vector<torch::Tensor> computation_aware_forward_gat_cuda_3d(
    torch::Tensor edge_ptr,
    torch::Tensor edge_data,
    torch::Tensor src_edges,
    torch::Tensor input_feat,
    int neighbor_num,
    int dst_nodes,
    int num_heads
);

std::vector<torch::Tensor> computation_aware_forward_gat_3d(
    torch::Tensor edge_ptr,
    torch::Tensor edge_data,
    torch::Tensor src_edges,
    torch::Tensor input_feat,
    int neighbor_num,
    int dst_nodes,
    int num_heads
){
    CHECK_INPUT(edge_ptr);
    CHECK_INPUT(src_edges);
    CHECK_INPUT(edge_data);
    // CHECK_INPUT(dst_norm_degs);
    CHECK_INPUT(input_feat);
    // CHECK_INPUT(weight);

    return computation_aware_forward_gat_cuda_3d(
        edge_ptr,
        edge_data,
        src_edges,
        // src_norm_degs,
        // dst_norm_degs,
        input_feat,
        neighbor_num,
        dst_nodes,
        num_heads);
}

std::vector<torch::Tensor> computation_aware_backward_gat_cuda_3d(
    torch::Tensor edge_ptr,
    torch::Tensor edge_data,
    torch::Tensor src_edges,
    torch::Tensor d_input,
    int src_nodes,
    int num_heads
);

std::vector<torch::Tensor> computation_aware_backward_gat_3d(
    torch::Tensor edge_ptr,
    torch::Tensor edge_data,
    torch::Tensor src_edges,
    torch::Tensor d_input,
    int src_nodes,
    int num_heads
){
    CHECK_INPUT(edge_ptr);
    CHECK_INPUT(src_edges);
    CHECK_INPUT(edge_data);
    // CHECK_INPUT(src_edges);
    // CHECK_INPUT(dst_norm_degs);
    CHECK_INPUT(d_input);
    // CHECK_INPUT(weight);

    return computation_aware_backward_gat_cuda_3d(
        edge_ptr,
        edge_data,
        src_edges,
        // src_norm_degs,
        // dst_norm_degs,
        d_input,
        src_nodes,
        num_heads);
}

std::vector<torch::Tensor> computation_aware_backward_gcn(
    torch::Tensor edge_ptr,
    torch::Tensor src_edges,
    torch::Tensor src_norm_degs,
    torch::Tensor dst_norm_degs,
    torch::Tensor d_input,
    torch::Tensor weight,
    torch::Tensor input_feat
) 
{
  CHECK_INPUT(edge_ptr);
  CHECK_INPUT(src_edges);
  CHECK_INPUT(src_norm_degs);
  CHECK_INPUT(dst_norm_degs);
  CHECK_INPUT(d_input);
  CHECK_INPUT(input_feat);
  CHECK_INPUT(weight);

  return computation_aware_backward_gcn_cuda(
    edge_ptr,
    src_edges,
    src_norm_degs,
    dst_norm_degs,
    d_input,
    weight,
    input_feat
    );
}

std::vector<torch::Tensor> computation_aware_backward_gin_cuda(
    torch::Tensor edge_ptr,
    torch::Tensor src_edges,
    torch::Tensor d_input,
    torch::Tensor weight,
    torch::Tensor input_feat,
    int src_nodes
);

std::vector<torch::Tensor> computation_aware_backward_gin(
    torch::Tensor edge_ptr,
    torch::Tensor src_edges,
    torch::Tensor d_input,
    torch::Tensor weight,
    torch::Tensor input_feat,
    int src_nodes
){
    CHECK_INPUT(edge_ptr);
    CHECK_INPUT(src_edges);
    CHECK_INPUT(d_input);
    CHECK_INPUT(input_feat);
    CHECK_INPUT(weight);

    return computation_aware_backward_gin_cuda(
        edge_ptr,
        src_edges,
        d_input,
        weight,
        input_feat,
        src_nodes
        );
}

// std::vector<torch::Tensor> aggregate_backward_reuse_out_gin_debug(
//     torch::Tensor edge_ptr,
//     torch::Tensor src_edges,
//     torch::Tensor d_input,
//     torch::Tensor weight,
//     torch::Tensor input_feat,
//     int src_nodes
// ){
//     CHECK_INPUT(edge_ptr);
//     CHECK_INPUT(src_edges);
//     CHECK_INPUT(d_input);
//     CHECK_INPUT(input_feat);
//     CHECK_INPUT(weight);

//     return aggregate_backward_reuse_out_gin_debug_cuda(
//         edge_ptr,
//         src_edges,
//         d_input,
//         weight,
//         input_feat,
//         src_nodes
//         );
// }

std::vector<torch::Tensor> aggregate_forward_outer(
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
  CHECK_INPUT(edge_ptr);
  CHECK_INPUT(dst_edges);
  CHECK_INPUT(src_norm_degs);
  CHECK_INPUT(dst_norm_degs);
  CHECK_INPUT(input_feat);
  CHECK_INPUT(weight);

  return aggregate_forward_outer_cuda(
    edge_ptr,
    dst_edges,
    src_norm_degs,
    dst_norm_degs,
    input_feat,
    weight,
    max_degree,
    dst_nodes,
    src_nodes);
}


std::vector<torch::Tensor> transfer_edge_id(
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    torch::Tensor dst_edges_unique,
    int dst_nodes, 
    int total_items,
    int unique_num,
    long int edge_nums
) 
{
  CHECK_INPUT(src_edges);
  CHECK_INPUT(dst_edges);
  CHECK_INPUT(dst_edges_unique);

  return transfer_edge_id_cuda(
    src_edges,
    dst_edges,
    dst_edges_unique,
    dst_nodes,
    total_items,
    unique_num,
    edge_nums);
}

std::vector<torch::Tensor> sample_node(
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    torch::Tensor edge_ptr,
    torch::Tensor seed_nodes,
    int dst_nodes,
    int neighbor_num
)
{
    CHECK_INPUT(src_edges);
    CHECK_INPUT(dst_edges);
    CHECK_INPUT(edge_ptr);
    CHECK_INPUT(seed_nodes);

    return sample_node_cuda(
        src_edges,
        dst_edges,
        edge_ptr,
        seed_nodes,
        dst_nodes,
        neighbor_num);
}

at::Tensor exclusive_sum(
    torch::Tensor in_degs
){
    CHECK_INPUT(in_degs);

    return exclusive_sum_cuda(in_degs);
}

at::Tensor cal_deg(
    torch::Tensor edges,
    int nodes_num
){
    CHECK_INPUT(edges);

    return cal_deg_cuda(edges,nodes_num);
}

at::Tensor reset_flag(
    torch::Tensor flag,
    torch::Tensor input_nodes
){
    CHECK_INPUT(flag);
    CHECK_INPUT(input_nodes);

    return reset_flag_cuda(flag,input_nodes);

}

at::Tensor reset_localid2cacheid(
    torch::Tensor localid2cacheid,
    torch::Tensor input_nodes
){
    CHECK_INPUT(localid2cacheid);
    CHECK_INPUT(input_nodes);

    return reset_localid2cacheid_cuda(localid2cacheid,input_nodes);

}

at::Tensor fetch_data(
    torch::Tensor data,
    torch::Tensor index
){
    CHECK_INPUT(data);
    CHECK_INPUT(index);

    return fetch_data_cuda(data,index);
}

at::Tensor filter_same_index_cuda(
    torch::Tensor gpu_flag,
    torch::Tensor batch_flag
);

at::Tensor filter_same_index(
    torch::Tensor gpu_flag,
    torch::Tensor batch_flag
){
    CHECK_INPUT(gpu_flag);
    CHECK_INPUT(batch_flag);
    return filter_same_index_cuda(gpu_flag, batch_flag);
}

at::Tensor apply_edge_cuda(
    torch::Tensor el,
    torch::Tensor er,
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    int num_heads
);

at::Tensor apply_edge(
    torch::Tensor el,
    torch::Tensor er,
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    int num_heads
){
    CHECK_INPUT(el);
    CHECK_INPUT(er);
    CHECK_INPUT(src_edges);
    CHECK_INPUT(dst_edges);

    return apply_edge_cuda(el,er,src_edges,dst_edges,num_heads);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("apply_edge", &apply_edge, "apply_edge (CUDA)");
  m.def("filter_same_index", &filter_same_index, "filter_same_index (CUDA)");
  m.def("fetch_data", &fetch_data, "fetch_data (CUDA)");
  m.def("reset_localid2cacheid", &reset_localid2cacheid, "reset_localid2cacheid (CUDA)");
  m.def("reset_flag", &reset_flag, "reset_flag (CUDA)");
  m.def("cal_deg", &cal_deg, "cal_deg (CUDA)");
  m.def("exclusive_sum", &exclusive_sum, "exclusive_sum (CUDA)");
  m.def("transfer_edge_id", &transfer_edge_id, "transfer_edge_id (CUDA)");
  m.def("sample_node", &sample_node, "sample_node (CUDA)");
  m.def("forward", &aggregate_forward, "aggregate forward (CUDA)");
  m.def("forward_gcn", &computation_aware_forward_gcn, "computation_aware_forward_gcn (CUDA)");
  m.def("forwaed_gin", &computation_aware_forward_gin, "computation_aware_forward_gin (CUDA)");
  m.def("forward_gat", &computation_aware_forward_gat, "computation_aware_forward_gat (CUDA)");
  m.def("forward_gat_3d", &computation_aware_forward_gat_3d, "computation_aware_forward_gat_3d (CUDA)");
  m.def("backward_gcn", &computation_aware_backward_gcn, "computation_aware_backward_gcn (CUDA)");
  m.def("backward_gin", &computation_aware_backward_gin, "computation_aware_backward_gin (CUDA)");
  m.def("backward_gat", &computation_aware_backward_gat, "computation_aware_backward_gat (CUDA)");
  m.def("backward_gat_3d", &computation_aware_backward_gat_3d, "computation_aware_backward_gat_3d (CUDA)");
  m.def("outer_forward", &aggregate_forward_outer, "aggregate_forward_outer (CUDA)");
}