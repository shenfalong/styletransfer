#include <vector>
#include "caffe/solver.hpp"
#include "caffe/layers/loss/w2_gd_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <cfloat>
namespace caffe {
//-----------------------------------

static __global__ void compute_sum(int num_spatial,int num, int channels, int spatial_dim, const float* bottom_sec_diff, const float * prob, float* sum) 
{
  CUDA_KERNEL_LOOP(i, num_spatial) 
  {
  	int n = i / spatial_dim;
  	int s = i % spatial_dim;
  	
  	float temp = 0;
  	for (int iter = 0; iter<channels-1; iter++)
  	{
  		int index = (n*channels+iter) * spatial_dim + s;
  		temp += bottom_sec_diff[index] * prob[index];
  	}
  	sum[i] = temp;
  }
}

static __global__ void secforward_kernel(const int count, const int num, const int channels, const int spatial_dim, 
					const float* prob, const float* label, const float* bottom_sec_diff, const float* sum_secx_p,  float* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index, count) 
  {
  	const int n = index / spatial_dim / channels;
  	const int c = index / spatial_dim % channels;
    const int s = index % spatial_dim;
    
    if (c <  channels-1)
  		bottom_diff[index] = bottom_sec_diff[index]*prob[index] - sum_secx_p[n*spatial_dim+s] *  prob[index];
  	else
  		bottom_diff[index] = 0;
  }
}
//-----------------------------------

static __global__ void Dloss_forward_kernel(int count, int num,int channels, int spatial_dim, const float *in, const float *label, 
																				float * prob, float *loss_g, float * loss_d, float *loss_c)
{
	CUDA_KERNEL_LOOP(i, count)
	{	
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		
		loss_g[n*spatial_dim+s] = -in[(n*channels+channels-1)*spatial_dim+s]; 
		loss_d[n*spatial_dim+s] = -in[((n+num)*channels+channels-1)*spatial_dim+s];
		
		#if 1
		float max_value = in[(n*channels+0)*spatial_dim+s];
		for (int iter=0;iter<channels-1;iter++)
			max_value = max(max_value,in[(n*channels+iter)*spatial_dim+s]);			
		float sum = 0;
		int label_index = label[n*spatial_dim+s];		
		for (int iter=0;iter<channels-1;iter++)
			sum += exp(in[(n*channels+iter)*spatial_dim+s]-max_value);
		for (int iter=0;iter<channels-1;iter++)
			prob[(n*channels+iter)*spatial_dim+s] = exp(in[(n*channels+iter)*spatial_dim+s]-max_value) / sum;
		loss_c[n*spatial_dim+s] = -log(max(prob[(n*channels+label_index)*spatial_dim+s],float(FLT_MIN)));
		
		max_value = in[((n+num)*channels+0)*spatial_dim+s];
		for (int iter=0;iter<channels-1;iter++)
			max_value = max(max_value,in[((n+num)*channels+iter)*spatial_dim+s]);
			
		sum = 0;
		for (int iter=0;iter<channels-1;iter++)
			sum += exp(in[((n+num)*channels+iter)*spatial_dim+s]-max_value);
		for (int iter=0;iter<channels-1;iter++)
			prob[((n+num)*channels+iter)*spatial_dim+s] = exp(in[((n+num)*channels+iter)*spatial_dim+s]-max_value) / sum;
		loss_c[(n+num)*spatial_dim+s] = -log(max(prob[((n+num)*channels+label_index)*spatial_dim+s],float(FLT_MIN)));
		#endif
	}
}

static __global__ void Gloss_forward_kernel(int count, int num,int channels, int spatial_dim, const float *in, const float *label, 
																				float * prob, float *loss_g, float * loss_c)
{
	CUDA_KERNEL_LOOP(i, count)
	{	
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		
		loss_g[n*spatial_dim+s] = -in[(n*channels+channels-1)*spatial_dim+s]; 
		
		#if 1
		float max_value = in[(n*channels+0)*spatial_dim+s];
		for (int iter=0;iter<channels-1;iter++)
			max_value = max(max_value,in[(n*channels+iter)*spatial_dim+s]);			
		float sum = 0;
		int label_index = label[n*spatial_dim+s];		
		for (int iter=0;iter<channels-1;iter++)
			sum += exp(in[(n*channels+iter)*spatial_dim+s]-max_value);
		for (int iter=0;iter<channels-1;iter++)
			prob[(n*channels+iter)*spatial_dim+s] = exp(in[(n*channels+iter)*spatial_dim+s]-max_value) / sum;
		loss_c[n*spatial_dim+s] = -log(max(prob[(n*channels+label_index)*spatial_dim+s],float(FLT_MIN)));
		#endif
	}
}


static __global__ void Dloss_backward_kernel(int count, int num,int channels, int spatial_dim, const float *data_in,const float *label, const float *prob,
																					float *diff_in)
{
	CUDA_KERNEL_LOOP(i, count)
	{	
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		
		diff_in[(n*channels+channels-1)*spatial_dim+s] = 1;
		diff_in[((n+num)*channels+channels-1)*spatial_dim+s] = -1;
#if 1
		for (int iter=0;iter<channels-1;iter++)
		{
			diff_in[(n*channels+iter)*spatial_dim+s] = 1 * prob[(n*channels+iter)*spatial_dim+s];
			diff_in[((n+num)*channels+iter)*spatial_dim+s] = 1 * prob[((n+num)*channels+iter)*spatial_dim+s];
		}	
		
		int label_index = label[n*spatial_dim+s];
		diff_in[(n*channels+label_index)*spatial_dim+s] -= 1;
		
		label_index = label[n*spatial_dim+s];
		diff_in[((n+num)*channels+label_index)*spatial_dim+s] -= 1;
#endif
	}
}

static __global__ void Gloss_backward_kernel(int count, int num,int channels, int spatial_dim, const float *data_in,const float *label, const float *prob,
																					float *diff_in)
{
	CUDA_KERNEL_LOOP(i, count)
	{		
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		
		diff_in[(n*channels+channels-1)*spatial_dim+s] = -1;
#if 1
		for (int iter=0;iter<channels-1;iter++)
		{
			diff_in[(n*channels+iter)*spatial_dim+s] = 0.5 * 0.1 * prob[(n*channels+iter)*spatial_dim+s];
		}	
		
		int label_index = label[n*spatial_dim+s];
		diff_in[(n*channels+label_index)*spatial_dim+s] -= 0.5 * 0.1;
#endif
	}
}


void W2GdLossLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	float loss_g, loss_d, loss_c;
	if (Caffe::gan_type() == "train_dnet")
	{	
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
	
	
		Dloss_forward_kernel<<<CAFFE_GET_BLOCKS(num/2*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num/2*height*width, num/2, channels, height*width, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
		prob_.mutable_gpu_data(), loss_g_.mutable_gpu_data(),loss_d_.mutable_gpu_data(), loss_c_.mutable_gpu_data());	

		
		loss_g = caffe_gpu_sum(loss_g_.count(),loss_g_.gpu_data());		
		loss_d = caffe_gpu_sum(loss_d_.count(),loss_d_.gpu_data());
		loss_c = caffe_gpu_sum(loss_c_.count(),loss_c_.gpu_data());
		top[0]->mutable_cpu_data()[0] = loss_d / float(num/2*height*width) - loss_g / float(num/2*height*width) + loss_c / float(num*height*width);
	}
	else
	{
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		
		Gloss_forward_kernel<<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num*height*width, num, channels, height*width, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
		prob_.mutable_gpu_data(), loss_g_.mutable_gpu_data(), loss_c_.mutable_gpu_data());	

		
		loss_g = 	caffe_gpu_sum(loss_g_.count(),loss_g_.gpu_data());			
		loss_c = caffe_gpu_sum(loss_c_.count(),loss_c_.gpu_data());	
		top[0]->mutable_cpu_data()[0] = loss_g / float(num*height*width) + 0.1 * loss_c / float(num*height*width);
	}
}


void W2GdLossLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	if (Caffe::second_pass() == false)
	{
		if (Caffe::gan_type() == "train_dnet")
		{
			int num = bottom[0]->num();
			int channels = bottom[0]->channels();
			int height = bottom[0]->height();
			int width = bottom[0]->width();
		
			float loss_weights_ = top[0]->cpu_diff()[0] / float(num/2*1*height*width);
			
			Dloss_backward_kernel<<<CAFFE_GET_BLOCKS(num/2*height*width), CAFFE_CUDA_NUM_THREADS>>>
			(num/2*height*width, num/2,channels,height*width, bottom[0]->gpu_data(), bottom[1]->gpu_data(), prob_.gpu_data(),
			bottom[0]->mutable_gpu_diff());	
			
			caffe_gpu_scal(bottom[0]->count(),loss_weights_,bottom[0]->mutable_gpu_diff());		
		}
		else
		{
			int num = bottom[0]->num();
			int channels = bottom[0]->channels();
			int height = bottom[0]->height();
			int width = bottom[0]->width();
		
			float loss_weights_ = top[0]->cpu_diff()[0] / float(num*1*height*width);
			
			Gloss_backward_kernel<<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
			(num*height*width, num,channels,height*width, bottom[0]->gpu_data(),bottom[1]->gpu_data(), prob_.gpu_data(),
			bottom[0]->mutable_gpu_diff());	
			
			caffe_gpu_scal(bottom[0]->count(),loss_weights_,bottom[0]->mutable_gpu_diff());	
		}
	}
	else
	{
	}
}

void W2GdLossLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
#if 0
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  compute_sum<<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
  (num*height*width, num, channels, height*width, bottom[0]->gpu_sec_diff(), prob_.gpu_data(), loss_c_.mutable_gpu_data()); 

	secforward_kernel<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(), num, channels, height*width, prob_.gpu_data(), bottom[1]->gpu_data(), bottom[0]->gpu_sec_diff(), loss_c_.gpu_data(),
	bottom[0]->mutable_gpu_diff());

	const float loss_weight = top[0]->cpu_diff()[0] / float(num/2*channels*height*width) * 1;
	caffe_gpu_scal(bottom[0]->count(), loss_weight, bottom[0]->mutable_gpu_diff()); 
#endif	
}

}  // namespace caffe
