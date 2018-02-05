#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"

namespace caffe {

inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
	CUDA_CHECK(cudaMallocHost(ptr, size));
	*use_cuda = true;
	return;
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
	CUDA_CHECK(cudaFreeHost(ptr));
	return;
}


class SyncedMemory 
{
 public:
  SyncedMemory() : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        cpu_malloc_use_cuda_(false), gpu_device_(-1) {}
  explicit SyncedMemory(size_t size) : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        cpu_malloc_use_cuda_(false), gpu_device_(-1) {}
  ~SyncedMemory();
  const void* cpu_data();
  const void* gpu_data();
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

	void set_cpu_data(void* data);

 private:
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool cpu_malloc_use_cuda_;

  int gpu_device_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory


inline void SyncedMemory::to_cpu() 
{
  switch (head_) 
  {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    break;
  case HEAD_AT_GPU:
    if (cpu_ptr_ == NULL) 
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);

    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() 
{
  switch (head_) 
  {
  case UNINITIALIZED:
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) 
    {
      CUDA_CHECK(cudaGetDevice(&gpu_device_));
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
}
}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
