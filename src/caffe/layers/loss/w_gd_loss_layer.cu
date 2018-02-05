
#include <vector>

#include "caffe/layers/loss/w_gd_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/solver.hpp"
namespace caffe {


static int qcompare(const void * a, const void * b)
{
   return *(float*)a - *(float*)b;
}


static int compare(float a, float b)
{
   return a < b;
}


static __global__ void Dloss_forward_kernel(int ncount, const float *in, const float *mask, float *count, float *loss_g, float *loss_d)
{

	CUDA_KERNEL_LOOP(i, ncount)
	{	
		if (mask[i] > float(0.5))
		{
			loss_g[i] =  -in[i];
			count[i] = 1;		
		}
		else
		{
			loss_g[i] =  0;
			count[i] = 0;		
		}
		
		if (mask[i+ncount] > float(0.5))
		{
			loss_d[i] = -in[i+ncount];
			count[i+ncount] = 1;		
		}
		else
		{
			loss_d[i] = 0;
			count[i+ncount] = 0;	
		}
	}
}


static __global__ void Gloss_forward_kernel(int ncount, const float *in, float *loss_g)
{
	CUDA_KERNEL_LOOP(i, ncount)
	{	
		loss_g[i] =  -in[i];
	}
}

static __global__ void Dloss_backward_kernel(int ncount, const float* mask, float *count, float *diff_in)
{

	CUDA_KERNEL_LOOP(i, ncount)
	{	
		if (mask[i] > float(0.5))
		{
			diff_in[i] = 1;
			count[i] = 1;		
		}
		else
		{
			diff_in[i] = 0;
			count[i] = 0;	
		}
		if (mask[i+ncount] > float(0.5))
		{
			diff_in[i+ncount] = -1;
			count[i+ncount] = 1;		
		}
		else
		{
			diff_in[i+ncount] = 0;
			count[i+ncount] = 0;		
		}
	}
}

static __global__ void Gloss_backward_kernel(int ncount, float *diff_in)
{

	CUDA_KERNEL_LOOP(i, ncount)
	{		
		diff_in[i] = -1;		
	}
}



void WGdLossLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	if (Caffe::gan_type() == "train_dnet")
	{	
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
	
	
		CHECK_EQ(bottom.size(),1);
		CHECK_EQ(num%2,0);
		CHECK_EQ(channels,1);
	
		
		caffe_gpu_set(mask_.count(),float(1),mask_.mutable_gpu_data());
		//caffe_gpu_rng_uniform(mask_.count(),float(0),float(1), mask_.mutable_gpu_data());
		#if 0
		caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(),mask_.mutable_gpu_data());
		float count;
		std::sort(mask_.mutable_cpu_data(),                 mask_.mutable_cpu_data() + mask_.count()/2,compare);
		std::sort(mask_.mutable_cpu_data()+ mask_.count()/2,mask_.mutable_cpu_data() + mask_.count()  ,compare);
		int index = mask_.count()/2 * 0.5; 
		float threshold_fake = mask_.cpu_data()[index];
		float threshold_true = mask_.cpu_data()[mask_.count()/2 + index];
		
		//LOG(INFO)<<"-threshold_fake = "<<-threshold_fake;
		//LOG(INFO)<<"-threshold_true = "<<-threshold_true;
		for (int i=0;i<bottom[0]->count()/2;i++)
		{
			if (-bottom[0]->cpu_data()[i] < -threshold_fake)
				mask_.mutable_cpu_data()[i] = 0.6;
			else
				mask_.mutable_cpu_data()[i] = 0.4;
			
			if (-bottom[0]->cpu_data()[i+bottom[0]->count()/2] > -threshold_true)
				mask_.mutable_cpu_data()[i+bottom[0]->count()/2] = 0.6;
			else
				mask_.mutable_cpu_data()[i+bottom[0]->count()/2] = 0.4;
		}
		#endif	
				
		Dloss_forward_kernel<<<CAFFE_GET_BLOCKS(num/2*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num/2*height*width, bottom[0]->gpu_data(),mask_.gpu_data(),count_.mutable_gpu_data(),loss_g_.mutable_gpu_data(),loss_d_.mutable_gpu_data());	
	
		;	
		float loss_g = caffe_gpu_sum(loss_g_.count(),loss_g_.gpu_data());
		float count_g;
		caffe_gpu_asum(count_.count()/2,count_.gpu_data(),&count_g);	
		
		float loss_d = caffe_gpu_sum(loss_d_.count(),loss_d_.gpu_data());
		float count_d;
		caffe_gpu_asum(count_.count()/2,count_.gpu_data()+count_.count()/2,&count_d);	
		
		top[0]->mutable_cpu_data()[0] = loss_d/max(count_d,float(1))- loss_g/max(count_g,float(1));
	}
	else
	{
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
	
	
		CHECK_EQ(bottom.size(),1);
		CHECK_EQ(channels,1);
		
		
		Gloss_forward_kernel<<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num*height*width,bottom[0]->gpu_data(),loss_g_.mutable_gpu_data());	
		
					
		float loss_g = caffe_gpu_sum(loss_g_.count(),loss_g_.gpu_data());	
		top[0]->mutable_cpu_data()[0] = loss_g / float(num*channels*height*width);
	}
	if (Solver::iter() % 100 == 0 && Caffe::gan_type() == "train_dnet")
		LOG(INFO)<<"d-loss "<<top[0]->cpu_data()[0];
}


void WGdLossLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	if (Caffe::second_pass() == false)
	{
		if (Caffe::gan_type() == "train_dnet")
		{		
			int num = bottom[0]->num();
			int channels = bottom[0]->channels();
			int height = bottom[0]->height();
			int width = bottom[0]->width();
	
	
			Dloss_backward_kernel<<<CAFFE_GET_BLOCKS(num/2*height*width), CAFFE_CUDA_NUM_THREADS>>>
			(num/2*height*width, mask_.gpu_data(),count_.mutable_gpu_data(),bottom[0]->mutable_gpu_diff());		
			
			float count_g;
			caffe_gpu_asum(count_.count()/2,count_.gpu_data(),&count_g);	
			float loss_weights_ = top[0]->cpu_diff()[0] / max(count_g,float(1));
			caffe_gpu_scal(bottom[0]->count()/2,loss_weights_,bottom[0]->mutable_gpu_diff());	
			
			float count_d;
			caffe_gpu_asum(count_.count()/2,count_.gpu_data(),&count_d);	
			loss_weights_ = top[0]->cpu_diff()[0] / max(count_d,float(1));
			caffe_gpu_scal(bottom[0]->count()/2,loss_weights_,bottom[0]->mutable_gpu_diff()+bottom[0]->count()/2);	
		}
		else
		{	
			int num = bottom[0]->num();
			int channels = bottom[0]->channels();
			int height = bottom[0]->height();
			int width = bottom[0]->width();
			
			Gloss_backward_kernel<<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
			(num*height*width,bottom[0]->mutable_gpu_diff());			
			
			float loss_weights_ = top[0]->cpu_diff()[0] / (num*1*height*width);
			caffe_gpu_scal(bottom[0]->count(),loss_weights_,bottom[0]->mutable_gpu_diff());	
		}	
	}
	else
	{
		caffe_gpu_set(bottom[0]->count(),float(0),bottom[0]->mutable_gpu_diff());
	}
}

void WGdLossLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}
}  // namespace caffe
