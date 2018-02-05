#include "caffe/layers/loss/gan_smooth_l1_loss_layer.hpp"
#include "caffe/solver.hpp"
namespace caffe {


static __global__ void SmoothL1Forward(const int n, const int ignore_value, const float* in_0, const float * in_1, float* out, float * count) 
{
  CUDA_KERNEL_LOOP(index, n) 
  {
  	if (in_0[index] == ignore_value)
  	{
  		count[index] = 0;
  		out[index] = 0;
  	}
  	else
  	{
			count[index] = 1;
			
		  float val = abs(in_0[index] - in_1[index]);
			
		  out[index] = val;
		}    
  }
}

static __global__ void SmoothL1Backward(const int n, const int ignore_value, const float* in_0, const float * in_1, float* in_0_diff, float * count) 
{
  CUDA_KERNEL_LOOP(index, n) 
  {
  	if (in_0[index] == ignore_value)
  	{
  		count[index] = 0;
  		in_0_diff[index] = 0;
  	}
  	else
  	{
		  float val = in_0[index] - in_1[index];
		  if (val > 0)
		  	in_0_diff[index] = float(1);
		  else
		  	in_0_diff[index] = float(-1);
		}    
  }
}


void GANSmoothL1LossLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
  if (Caffe::gan_type() == "train_gnet")
  {
		SmoothL1Forward<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->count(),ignore_value,bottom[0]->gpu_data(),bottom[1]->gpu_data(), loss_.mutable_gpu_data(),counts_.mutable_gpu_data());

		float counts;
		caffe_gpu_asum(counts_.count(), counts_.gpu_data(), &counts);
		
		float loss;
		caffe_gpu_asum(loss_.count(), loss_.gpu_data(), &loss);
		
		top[0]->mutable_cpu_data()[0] = loss / counts;
		
		if (Solver::iter() % 100 == 0)
			LOG(INFO)<<"reconstruction_loss = "<<top[0]->cpu_data()[0];
	}
}


void GANSmoothL1LossLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	if (Caffe::gan_type() == "train_gnet")
  {
		SmoothL1Backward<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->count(),ignore_value,bottom[0]->gpu_data(),bottom[1]->gpu_data(), bottom[0]->mutable_gpu_diff(),counts_.mutable_gpu_data());
		
		float counts;
		caffe_gpu_asum(counts_.count(), counts_.gpu_data(), &counts);
		
		caffe_gpu_scal(bottom[0]->count(),top[0]->cpu_diff()[0] / counts,bottom[0]->mutable_gpu_diff());
	}
}

void GANSmoothL1LossLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}


}  // namespace caffe
