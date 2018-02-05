#include "caffe/layer.hpp"
#include "caffe/solver.hpp"
#include<cfloat>
namespace caffe {
static __global__ void scale_kernel(int count, int image_dim, float sec_loss_weight, float norm_value, 
																			const float *in, const float *coef, float *out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / image_dim;
		out[i] = 2 * sec_loss_weight  *(coef[n]-norm_value)/ coef[n] * in[i];	
	} 
}

static __global__ void compute_sum(int image_dim, const float *in, float *out)
{
	__shared__ float buffer[CAFFE_CUDA_NUM_THREADS];

	buffer[threadIdx.x] = 0;
	for (int i = threadIdx.x;i < image_dim;i += blockDim.x)
		buffer[threadIdx.x] += in[blockIdx.x*image_dim+i]*in[blockIdx.x*image_dim+i];
	__syncthreads();
	
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
			buffer[threadIdx.x] += buffer[threadIdx.x+s];
		__syncthreads();
	}
	
	if (threadIdx.x == 0)
		out[blockIdx.x] = sqrt(buffer[0]);
}

void Layer::compute_sec_loss(const vector<Blob*>& top, const float sec_loss_weight, const float norm_value)
{
	vector<shared_ptr<Blob> > sum_;
	sum_.resize(top.size());
	for (int i=0;i < top.size();i++)
	{
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i%NGPUS]));
		int num = top[i]->num();
		int channels = top[i]->channels();
		int height = top[i]->height();
		int width = top[i]->width();	
		
		sum_[i].reset(new Blob(num,1,1,1));
		compute_sum<<<num,CAFFE_CUDA_NUM_THREADS>>>
		(channels*height*width,top[i]->gpu_diff(),sum_[i]->mutable_gpu_data());
		
		if (Solver::iter() % 1000 == 0)
		{
			float sum = 0;
			for (int iter = 0;iter<num;iter++)
				sum += sum_[i]->cpu_data()[iter];
			LOG(INFO)<<"sum = "<<sum/float(num);
		}
		scale_kernel<<<CAFFE_GET_BLOCKS(top[i]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(top[i]->count(), channels*height*width, sec_loss_weight, norm_value,
		top[i]->gpu_diff(), sum_[i]->gpu_data(), top[i]->mutable_gpu_sec_diff());	
		
		caffe_gpu_scal(top[i]->count(),float(1)/float(num),top[i]->mutable_gpu_sec_diff());
	}
	CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
}
//----------------------------------------- proto <->  memory--------------------

void Layer::ToProto(LayerParameter* param, bool write_diff) 
{
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (int i = 0; i < blobs_.size(); ++i) 
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  
}

}
