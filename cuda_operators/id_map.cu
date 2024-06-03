#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include "cuda_hashtable.cuh"


constexpr static const int BLOCK_SIZE = 256;
constexpr static const size_t TILE_SIZE = 1024;



inline __device__ int32_t
AtomicCAS(int32_t* const address, const int32_t compare, const int32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(
      reinterpret_cast<Type*>(address), static_cast<Type>(compare),
      static_cast<Type>(val));
}

struct Entry {
    int key;
    int local;
    int index;
};

typedef Entry* Iterator;

/**
 * @brief This is the mutable version of the DeviceOrderedHashTable, for use in
 * inserting elements into the hashtable.
 *
 * @tparam IdType The type of ID to store in the hashtable.
 */
template <typename IdType>
class MutableDeviceOrderedHashTable {
 public:
    unsigned int index=0;

    explicit MutableDeviceOrderedHashTable(int size){
        // printf("size_:%d\n",size);
        size_ = size;
        tableSizeBytes_ = size_ * sizeof(Entry);

        // Allocate gpu memory for the hash table
        cudaMalloc((void**)&table_, tableSizeBytes_);

        // init the key and value in hash table to -1
        cudaMemsetAsync(table_, -1, tableSizeBytes_);
    }

    ~MutableDeviceOrderedHashTable() {
        cudaFree(table_);
    }


    inline __device__ Iterator Search(const IdType id) {
        const IdType pos = SearchForPosition(id);

        return GetMutable(pos);
    }

    inline __device__ IdType SearchForPosition(const IdType id) const {
        IdType pos = Hash(id);

        // linearly scan for matching entry
        IdType delta = 1;
        while (table_[pos].key != id) {
            assert(table_[pos].key != -1);
            pos = Hash(pos + delta);
            delta += 1;
        }
        assert(pos < size_);

        return pos;
  }

    inline __device__ int UpdateIndex(unsigned int max_size) {
        
        return atomicInc(&index,max_size);

    }
  
    inline __device__ void UpdateLocal(int pos, int local) {
        GetMutable(pos)->local = local;
    }

    __host__ void print(int pos) {
        printf("index:%d",pos);
        printf("key:%d,local:%d,index:%d\n",GetMutable(pos)->key,GetMutable(pos)->local,GetMutable(pos)->index);
    }
  
  inline __device__ bool AttemptInsertAt(
      const int pos, const IdType id, const int index) {
    const IdType key = AtomicCAS(&GetMutable(pos)->key, -1, id);
    if (key == -1 || key == id) {
      // we either set a match key, or found a matching key, so then place the
      // minimum index in position. Match the type of atomicMin, so ignore
      // linting
      atomicMin(
          reinterpret_cast<unsigned long long*>(  // NOLINT
              &GetMutable(pos)->index),
          static_cast<unsigned long long>(index));  // NOLINT
      return true;
    } else {
      // we need to search elsewhere
      return false;
    }
  }

    Entry GetEntry(int n) {
        Entry entry;
        int pos = n%size_;
        if (pos >= 0 && pos < size_) {
            cudaMemcpy(&entry, &table_[pos], sizeof(Entry), cudaMemcpyDeviceToHost);
        } else {
            entry.key = -1;
            entry.local = -1;
            entry.index = -1;
        }
        return entry;
    }


  inline __device__ Iterator Insert(const IdType id, const int index) {
    int pos = Hash(id);
    printf("pos:%zu\n",pos);

    // linearly scan for an empty slot or matching entry
    IdType delta = 1;
    while (!AttemptInsertAt(pos, id, index)) {
      pos = Hash(pos + delta);
      delta += 1;
    }

    return GetMutable(pos);
  }

  inline __device__ int InsertSrc(const IdType id, const int index) {
    int pos = Hash(id);

    // linearly scan for an empty slot or matching entry
    IdType delta = 1;
    while (!AttemptInsertAt(pos, id, index)) {
      pos = Hash(pos + delta);
      delta += 1;
    }

    if((GetMutable(pos)->local==-1)&&(GetMutable(pos)->key!=id)){
        return pos;
    } else{
        return -1;
    }

  }

  inline __device__ int Hash(const IdType id) const { return id % size_; }

  Entry* GetTablePointer() {
        return table_;
    } 

 private:
    int size_;
    size_t tableSizeBytes_;
    Entry* table_;
    inline __device__ Iterator GetMutable(const int pos) {
        // assert(pos < this->size_);
        return const_cast<Iterator>(table_ + pos);
    }
};


__device__ int Hash(const int id, int size_) { return id % size_; }

__device__ bool AttemptInsertAt(
      Entry* table, const int pos, const int id, const int index, int* have_key) {
    const int key = atomicCAS(&table[pos].key, -1, id);
    if (key == -1 || key == id) {
        atomicMin(&table[pos].index, index);  
        if(key==id){
            *have_key=1;
        } else{
            *have_key=0;
        }
        return true;
    } else {
      return false;
    }
}

__device__ int Insert(Entry* table, const int id, const int index, int size) {
    int pos = Hash(id, size);
    // int pos = index;

    int delta = 1;
    int have_key=0;
    while (!AttemptInsertAt(table, pos, id, -1*index, &have_key)) {
      pos = Hash(pos + delta, size);
      delta += 1;
    }

    return pos;
}

__device__ int InsertSrc(Entry* table, const int id, const int index, int size) {
    int pos = Hash(id, size);
    // int pos = index;

    int delta = 1;
    int have_key;
    while (!AttemptInsertAt(table, pos, id, -1*index, &have_key)) {
      pos = Hash(pos + delta, size);
      delta += 1;
    }
    if((table[pos].local==-1) && (have_key==0)){
        // printf("pos unique");
        return pos;
    } else{
        if(table[pos].local!=-1){
            return -1;
        }
        if((table[pos].local==-1) && (have_key==1)){
            return -2;
        }
    }
}


__device__ void UpdateLocal(Entry* table, int pos, int local) {
        table[pos].local = local;
        // atomicExch(&table[pos].local, local);
}



__device__ int Search(Entry* table, const int id, int size) {
        int pos = Hash(id,size);

        // linearly scan for matching entry
        int delta = 1;
        while (table[pos].key != id) {
            // assert(table_[pos].key != -1);
            // printf("node id:%d,pos:%d,size:%d\n",id,pos,size);
            pos = Hash(pos + delta,size);
            delta += 1;
        }

    return pos;
}


// insert the dst edges into the hash table 
__global__ void generate_unique_index_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_edges,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> unique_index,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_nodes,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_nodes_insrc,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_nodes_sorted,
                    Entry* table,
                    int dst_nodes,
                    int total_items,
                    int size
){
    int current_insert_id = blockIdx.x * blockDim.x + threadIdx.x;

    int index = 0;

    // printf("generate_unique_index_kernel\n");


    if(current_insert_id>=dst_nodes){
        return;
    }
    
    if ((current_insert_id < dst_nodes)&&(current_insert_id<dst_edges.size(0))) {
        int pos = Insert(table, dst_edges[current_insert_id], current_insert_id, size);
        atomicExch(&table[pos].local, current_insert_id);
        dst_nodes_sorted[current_insert_id] = dst_edges[current_insert_id];
    } 
}

// insert the src edges into the hash table
__global__ void generate_unique_index_kernel_src(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_edges,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> unique_index,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_nodes,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_nodes_insrc,
                    Entry* table,
                    int dst_nodes,
                    int total_items,
                    int size
){
    int current_insert_id = blockIdx.x * blockDim.x + threadIdx.x;

    int index = 0;


    // printf("generate_unique_index_kernel_src\n");

    if(current_insert_id>=(total_items-dst_nodes)){
        return;
    }
    
    
    int edge_index = current_insert_id;
    int pos = InsertSrc(table, src_edges[edge_index], current_insert_id, size);
    if((pos!=-1) && (pos!=-2)){
        // increment local id
        index = atomicAdd((int*)&src_nodes[0],1);

        // record the inserted position in the hash table and the local id
        unique_index[index] = pos;
        
    } 
}

__global__ void generate_hashmap_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_edges,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> unique_index,
                    Entry* table,
                    int dst_nodes,
                    int src_nodes,
                    int total_items
){
    int current_insert_id = blockIdx.x * blockDim.x + threadIdx.x;


    // printf("generate_hashmap_kernel\n");

    if(current_insert_id>=src_nodes){
        return;
    }
    // update the local id in the hash table without synchronization
    UpdateLocal(table,unique_index[current_insert_id],current_insert_id+dst_nodes);

    // ascending local id
    unique_index[current_insert_id] = table[unique_index[current_insert_id]].key;
}


// id map with the hash table
__global__ void transfer_edge_id_kernel(
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> dst_edges,
                    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> src_edges,
                    // torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> out_degs,
                    // torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> in_degs,
                    Entry* table,
                    int size,
                    long int edge_nums
){
    int current_edge_id = blockIdx.x * blockDim.x + threadIdx.x;

    



    if(current_edge_id>=2*edge_nums){
        return;
    }
    // src edges mapping
    if(current_edge_id >= edge_nums){
        int index = current_edge_id-edge_nums;
        int src_pos = Search(table,src_edges[index],size);
        // const int id = table[src_pos].local;
        src_edges[index] = table[src_pos].local;
        // atomicAdd((int*)&out_degs[id],1);
    }
    // dst edges mapping
    else{
        int dst_pos = Search(table,dst_edges[current_edge_id],size);
        dst_edges[current_edge_id] = table[dst_pos].local;
    }
    
}


std::vector<torch::Tensor> transfer_edge_id_cuda(
    torch::Tensor src_edges,
    torch::Tensor dst_edges,
    torch::Tensor dst_edges_unique,
    int dst_nodes, 
    int total_items,
    int unique_num, 
    long int edge_nums){

    auto unique_index = torch::zeros({unique_num}, torch::kCUDA).to(at::kInt);

    auto dst_nodes_sorted = torch::zeros({dst_edges_unique.size(0)}, torch::kCUDA).to(at::kInt);




    const dim3 threads(128);
    const dim3 blocks((dst_nodes + threads.x -1)/threads.x);

    
    

    int size = TableSize(unique_num,2);
    auto device_table = MutableDeviceOrderedHashTable<int>(TableSize(unique_num,2));

    Entry* table_ptr = device_table.GetTablePointer();

    auto src_nodes = torch::zeros({1},torch::kCUDA).to(at::kInt);
    auto dst_nodes_insrc = torch::zeros({1},torch::kCUDA).to(at::kInt);

    // for (int i =0;i<100;i++){
    //     device_table.print(i);
    // }

    
    AT_DISPATCH_ALL_TYPES(dst_edges_unique.type(), "generate_unique_index_kernel", ([&] {
                                    generate_unique_index_kernel<<<blocks, threads>>>(
                                        dst_edges_unique.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        unique_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_nodes.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        dst_nodes_insrc.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        dst_nodes_sorted.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        table_ptr,
                                        dst_nodes,
                                        total_items,
                                        size
                                    );
                                }));

    const dim3 blocks_1((edge_nums + threads.x -1)/threads.x);

    AT_DISPATCH_ALL_TYPES(dst_edges_unique.type(), "generate_unique_index_kernel_src", ([&] {
                                    generate_unique_index_kernel_src<<<blocks_1, threads>>>(
                                        dst_edges_unique.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        unique_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_nodes.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        dst_nodes_insrc.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        table_ptr,
                                        dst_nodes,
                                        total_items,
                                        size
                                    );
                                }));

    // unsigned int src_nodes = device_table.index; 

    unsigned int ui_src_nodes = static_cast<unsigned int>(src_nodes[0].item<int>());
    const dim3 blocks_2((ui_src_nodes + threads.x -1)/threads.x);
    

    AT_DISPATCH_ALL_TYPES(dst_edges_unique.type(), "generate_hashmap_kernel", ([&] {
                                    generate_hashmap_kernel<<<blocks_2, threads>>>(
                                        dst_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        unique_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        table_ptr,
                                        dst_edges_unique.size(0),
                                        ui_src_nodes,
                                        total_items
                                    );
                                }));

    const dim3 blocks_3((2*edge_nums + threads.x -1)/threads.x);

    AT_DISPATCH_ALL_TYPES(dst_edges_unique.type(), "transfer_edge_id_kernel", ([&] {
                                    transfer_edge_id_kernel<<<blocks_3, threads>>>(
                                        dst_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        src_edges.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        // out_degs.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        // in_degs.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                        table_ptr,
                                        size,
                                        edge_nums
                                    );
                                }));

    // printf("three kernels finisher\n");

    // auto src_nodes_all = src_nodes[0] + dst_nodes;


    

    return {src_edges, dst_edges, torch::cat({dst_nodes_sorted,torch::slice(unique_index, 0, 0, src_nodes[0].item<int>())})};

}






