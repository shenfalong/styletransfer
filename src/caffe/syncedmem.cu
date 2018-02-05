#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() 
{
  if (cpu_ptr_ != NULL)
  { 
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  	cpu_ptr_ = NULL;
  }  

  if (gpu_ptr_ != NULL) 
  {
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) 
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    
    CUDA_CHECK(cudaFree(gpu_ptr_));
    gpu_ptr_ = NULL;
    cudaSetDevice(initial_device);
  }
}


const void* SyncedMemory::cpu_data() 
{
  to_cpu();
  return (const void*)cpu_ptr_;
}


const void* SyncedMemory::gpu_data() 
{
	to_gpu();
  return (const void*)gpu_ptr_;
}


void* SyncedMemory::mutable_cpu_data() 
{
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() 
{
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) 
{
  if (data != NULL)
  {
		cpu_ptr_ = data;
		head_ = HEAD_AT_CPU;
	}
	else
	{
		cpu_ptr_ = data;
		head_ = UNINITIALIZED;
	}	
}




}  // namespace caffe

