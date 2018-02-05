#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/solver.hpp"
#include "caffe/layers/loss/be_gd_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


static __global__ void Be_Dloss_forward_kernel(int count, const float *in_0, const float * in_1, float *out_d, float * out_g)
{

	CUDA_KERNEL_LOOP(i, count)
	{	
		out_g[i] =  abs(in_0[i] - in_1[i]);
		out_d[i] =  abs(in_0[i+count] - in_1[i+count]);
	}
}


static __global__ void Be_Gloss_forward_kernel(int count, const float *in_0, const float * in_1, float *out_g)
{
	CUDA_KERNEL_LOOP(i, count)
	{	
		out_g[i] =  abs(in_0[i] - in_1[i]);
	}
}

static __global__ void Be_Dloss_backward_kernel(int count, float k, const float *data_in_0, const float * data_in_1, float *diff_in_0, float *diff_in_1)
{

	CUDA_KERNEL_LOOP(i, count)
	{	
		if (data_in_0[i] > data_in_1[i])
		{
			diff_in_0[i] = -k;
			diff_in_1[i] = k;
		}
		else
		{
			diff_in_0[i] = k;		
			diff_in_1[i] = -k;	
		}
		
		if (data_in_0[i+count] > data_in_1[i+count])
		{
			diff_in_0[i+count] = 1;
			diff_in_1[i+count] = -1;
		}
		else
		{
			diff_in_0[i+count] = -1;
			diff_in_1[i+count] = 1;
		}	
	}
}

static __global__ void Be_Gloss_backward_kernel(int count, const float *data_in_0, const float *data_in_1, float *diff_in_0, float *diff_in_1)
{

	CUDA_KERNEL_LOOP(i, count)
	{				
		if (data_in_0[i] > data_in_1[i]) 
		{
			diff_in_0[i] = 1;
			diff_in_1[i] = -1;
		}
		else
		{
			diff_in_0[i] = -1;
			diff_in_1[i] = 1;
		}
		diff_in_0[i+count] = float(0);			
		diff_in_1[i+count] = float(0);
	}
}

void BeGdLossLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

	float gamma = 0.5;
	if (Caffe::gan_type() == "train_dnet")
	{	
		Be_Dloss_forward_kernel<<<CAFFE_GET_BLOCKS(num/2*channels*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num/2*channels*height*width,bottom[0]->gpu_data(),bottom[1]->gpu_data(),loss_d_.mutable_gpu_data(),loss_g_.mutable_gpu_data());	
		
		caffe_gpu_asum(loss_d_.count(),loss_d_.gpu_data(),&sum_d);
		caffe_gpu_asum(loss_g_.count(),loss_g_.gpu_data(),&sum_g);

		//top[0]->mutable_cpu_data()[0] = (sum_d - k_ * sum_g) / (num/2*channels*height*width);
		//top[0]->mutable_cpu_data()[0] = (sum_d + abs(gamma*sum_d - sum_g)) / (num/2*channels*height*width);	
		top[0]->mutable_cpu_data()[0] = sum_d / (num/2*channels*height*width);
	}
	else
	{
		Be_Gloss_forward_kernel<<<CAFFE_GET_BLOCKS(num/2*channels*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num/2*channels*height*width,bottom[0]->gpu_data(),bottom[1]->gpu_data(),loss_g_.mutable_gpu_data());	
	
		caffe_gpu_asum(loss_g_.count(),loss_g_.gpu_data(),&sum_g);
		top[0]->mutable_cpu_data()[0] = sum_g / (num/2*channels*height*width);
	
		
		this->blobs_[0]->mutable_cpu_data()[0] = 0.08;
		//this->blobs_[0]->mutable_cpu_data()[0] = this->blobs_[0]->cpu_data()[0] + 0.001*(gamma*sum_d - sum_g) / float(num/2*channels*height*width);

		//LOG(INFO)<<"delta = "<<(gamma*sum_d - sum_g) / float(num/2*channels*height*width);
		//if (this->blobs_[0]->cpu_data()[0] < 0)
		//	this->blobs_[0]->mutable_cpu_data()[0] = 0;
		//if (this->blobs_[0]->cpu_data()[0] > 1)
		//	this->blobs_[0]->mutable_cpu_data()[0] = 1.0;
			
		//if (Solver::iter() % 100 == 0)
		//	LOG(INFO)<<this->blobs_[0]->cpu_data()[0];
	}
}


void BeGdLossLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	if (Caffe::second_pass() == false)
	{
		float loss_weights_ =  top[0]->cpu_diff()[0] / (num/2*channels*height*width);
		if (Caffe::gan_type() == "train_dnet")
		{
			Be_Dloss_backward_kernel<<<CAFFE_GET_BLOCKS(num/2*channels*height*width), CAFFE_CUDA_NUM_THREADS>>>
			(num/2*channels*height*width,this->blobs_[0]->cpu_data()[0],bottom[0]->gpu_data(),bottom[1]->gpu_data(),bottom[0]->mutable_gpu_diff(),bottom[1]->mutable_gpu_diff());	
		}
		else
		{
			Be_Gloss_backward_kernel<<<CAFFE_GET_BLOCKS(num/2*channels*height*width), CAFFE_CUDA_NUM_THREADS>>>
			(num/2*channels*height*width,bottom[0]->gpu_data(),bottom[1]->gpu_data(),bottom[0]->mutable_gpu_diff(),bottom[1]->mutable_gpu_diff());	
		}
		caffe_gpu_scal(bottom[0]->count(),loss_weights_,bottom[0]->mutable_gpu_diff());		
		caffe_gpu_scal(bottom[1]->count(),loss_weights_,bottom[1]->mutable_gpu_diff());	
	}
	else
	{
		caffe_gpu_set(bottom[0]->count(),float(0),bottom[0]->mutable_gpu_diff());
	}
}

void BeGdLossLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	//float sum;
	//caffe_gpu_asum(bottom[0]->count(),bottom[0]->gpu_sec_diff(),&sum);
	//LOG(INFO)<<sum;
}

}  // namespace caffe
