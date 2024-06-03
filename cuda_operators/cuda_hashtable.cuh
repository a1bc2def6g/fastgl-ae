#include "cuda_runtime.h"
#include <iostream>





typedef enum {
  /** @brief CPU device */
  kDGLCPU = 1,
  /** @brief CUDA GPU device */
  kDGLCUDA = 2,
  // add more devices once supported
} AccDeviceType;

typedef struct {
  /** @brief The device type used in the device. */
  AccDeviceType device_type;
  /**
   * @brief The device index.
   * For vanilla CPU memory, pinned memory, or managed memory, this is set to 0.
   */
  int32_t device_id;
} DGLContext;


size_t TableSize(const size_t num, const int scale) {
  const size_t next_pow2 = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
  return next_pow2 << scale;
}

template <typename>
class OrderedHashTable;

template <typename IdType>
class DeviceOrderedHashTable {
 public:
  /**
   * @brief An entry in the hashtable.
   */
  struct Mapping {
    /**
     * @brief The ID of the item inserted.
     */
    IdType key;
    /**
     * @brief The index of the item in the unique list.
     */
    IdType local;
    /**
     * @brief The index of the item when inserted into the hashtable (e.g.,
     * the index within the array passed into FillWithDuplicates()).
     */
    int64_t index;
  };

  typedef const Mapping* ConstIterator;

  DeviceOrderedHashTable(const DeviceOrderedHashTable& other) = default;
  DeviceOrderedHashTable& operator=(const DeviceOrderedHashTable& other) =
      default;

  inline __device__ ConstIterator Search(const IdType id) const {
    const IdType pos = SearchForPosition(id);

    return &table_[pos];
  }

  inline __device__ bool Contains(const IdType id) const {
    IdType pos = Hash(id);

    IdType delta = 1;
    while (table_[pos].key != kEmptyKey) {
      if (table_[pos].key == id) {
        return true;
      }
      pos = Hash(pos + delta);
      delta += 1;
    }
    return false;
  }

 protected:
  // Must be uniform bytes for memset to work
  static constexpr IdType kEmptyKey = static_cast<IdType>(-1);

  const Mapping* table_;
  size_t size_;

  explicit DeviceOrderedHashTable(const Mapping* table, size_t size);


  inline __device__ IdType SearchForPosition(const IdType id) const {
    IdType pos = Hash(id);

    // linearly scan for matching entry
    IdType delta = 1;
    while (table_[pos].key != id) {
      assert(table_[pos].key != kEmptyKey);
      pos = Hash(pos + delta);
      delta += 1;
    }
    assert(pos < size_);

    return pos;
  }

  inline __device__ size_t Hash(const IdType id) const { return id % size_; }

  friend class OrderedHashTable<IdType>;
};

template <typename IdType>
class OrderedHashTable {
 public:
  static constexpr int kDefaultScale = 3;

  using Mapping = typename DeviceOrderedHashTable<IdType>::Mapping;

  OrderedHashTable(
      const size_t size, DGLContext ctx,
      const int scale = kDefaultScale);

  /**
   * @brief Cleanup after the hashtable.
   */
  ~OrderedHashTable();

  // Disable copying
  OrderedHashTable(const OrderedHashTable& other) = default;
  OrderedHashTable& operator=(const OrderedHashTable& other) = default;

  DeviceOrderedHashTable<IdType> DeviceHandle() const;

 private:
  Mapping* table_;
  size_t size_;
  DGLContext ctx_;
};


template <typename IdType>
DeviceOrderedHashTable<IdType>::DeviceOrderedHashTable(
    const Mapping* const table, const size_t size)
    : table_(table), size_(size) {}

template <typename IdType>
DeviceOrderedHashTable<IdType> OrderedHashTable<IdType>::DeviceHandle() const {
  return DeviceOrderedHashTable<IdType>(table_, size_);
}

// OrderedHashTable implementation

template <typename IdType>
OrderedHashTable<IdType>::OrderedHashTable(
    const size_t size, DGLContext ctx, const int scale)
    : table_(nullptr), size_(TableSize(size, scale)), ctx_(ctx) {
  // make sure we will at least as many buckets as items.
  CHECK_GT(scale, 0);

  cudaMalloc(&table_, sizeof(Mapping) * size_);

  if (table_ == nullptr) {
      throw std::runtime_error("Failed to allocate GPU memory.");
  }

  cudaMemsetAsync(
      table_, DeviceOrderedHashTable<IdType>::kEmptyKey,
      sizeof(Mapping) * size_);
}

template <typename IdType>
OrderedHashTable<IdType>::~OrderedHashTable() {
    cudaFree(table_);
}