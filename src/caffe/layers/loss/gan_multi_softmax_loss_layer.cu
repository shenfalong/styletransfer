
#include <vector>

#include "caffe/layers/loss/gan_multi_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/solver.hpp"
namespace caffe {


static __global__ void forward_kernel(const int num_spatial, const int num, const int channels, const int spatial_dim,
          const float* prob_data, const float* label,  float* loss) 
{
  CUDA_KERNEL_LOOP(index, num_spatial) 
  {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    int label_value = static_cast<int>(label[index]);
   		

    loss[index] = -log(prob_data[(n * channels + label_value) * spatial_dim + s]);
  }
}


static __global__ void backward_kernel(const int num_spatial,const int num, const int channels, const int spatial_dim, 
					const float* prob_data, const float* label,  float* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index, num_spatial) 
  {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    int label_value = static_cast<int>(label[index]);
		
  	for (int c = 0; c < channels; ++c) 
    {
    	int ind = (n*channels+c)*spatial_dim+s;
    	if (c == label_value)
    		bottom_diff[ind] = prob_data[ind] - 1;
			else
				bottom_diff[ind] = prob_data[ind];
    }
  }
}


void GANMultiSoftmaxWithLossLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	softmax_layer_->Forward_gpu(softmax_bottom_vec_, softmax_top_vec_);

 	if (Caffe::gan_type() == "train_dnet")
 	{	
		int num = bottom[0]->num()/2;
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		
		//LOG(INFO)<<"num = "<<num<<", channels = "<<channels;
		
		forward_kernel<<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num*height*width, num, channels, height*width, prob_.gpu_data()+prob_.offset(num), bottom[1]->gpu_data(), loss_.mutable_gpu_data());
		
		float loss;
		caffe_gpu_asum(num*height*width, loss_.gpu_data(), &loss);

		top[0]->mutable_cpu_data()[0] = loss / loss_.count() * float(1);
	}
	else
	{
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();

 		
		forward_kernel<<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num*height*width, num, channels, height*width, prob_.gpu_data(), bottom[1]->gpu_data(), loss_.mutable_gpu_data());
	
		float loss;
		caffe_gpu_asum(num*height*width, loss_.gpu_data(), &loss);

		top[0]->mutable_cpu_data()[0] = loss / loss_.count() * float(0.1);
	}
	
	if (Solver::iter() % 100 == 0 && Caffe::gan_type() == "train_dnet")
			LOG(INFO)<<"D softmax_loss = "<<top[0]->cpu_data()[0];
	
	if (Solver::iter() % 100 == 0 && Caffe::gan_type() == "train_gnet")
			LOG(INFO)<<"G softmax_loss = "<<top[0]->cpu_data()[0]*10;
}


void GANMultiSoftmaxWithLossLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	if (Caffe::gan_type() == "train_dnet")
 	{	
 		int num = bottom[0]->num()/2;
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		
		caffe_gpu_set(bottom[0]->count(),float(0),bottom[0]->mutable_gpu_diff());
		backward_kernel<<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num*height*width, num, channels, height*width, prob_.gpu_data()+prob_.offset(num), bottom[1]->gpu_data(), 
																					bottom[0]->mutable_gpu_diff()+bottom[0]->offset(num));
	
		const float loss_weight =  top[0]->cpu_diff()[0] / loss_.count() * float(1);
		caffe_gpu_scal(prob_.count(), loss_weight, bottom[0]->mutable_gpu_diff());
 	}
 	else
 	{
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
	
		backward_kernel<<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num*height*width, num, channels, height*width, prob_.gpu_data(), bottom[1]->gpu_data(), bottom[0]->mutable_gpu_diff());

		const float loss_weight = top[0]->cpu_diff()[0] / loss_.count() * float(0.1);
		caffe_gpu_scal(prob_.count(), loss_weight, bottom[0]->mutable_gpu_diff());
	}
}


void GANMultiSoftmaxWithLossLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}


}  // namespace caffe
