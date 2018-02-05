#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/solver.hpp"
#include "caffe/layers/loss/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

static __global__ void compute_sum(int num_spatial,int num, int channels, int spatial_dim, const float* bottom_sec_diff, const float * prob, float* sum) 
{
  CUDA_KERNEL_LOOP(i, num_spatial) 
  {
  	int n = i / spatial_dim;
  	int s = i % spatial_dim;
  	
  	float temp = 0;
  	for (int c=0;c<channels;c++)
  	{
  		int index = (n*channels+c)*spatial_dim+s;
  		temp += bottom_sec_diff[index]*prob[index];
  	}
  	sum[i] = temp;
  }
}

static __global__ void forward_kernel(const int num_spatial, const int num, const int channels, const int spatial_dim,
          const float* prob_data, const float* label, 
          const bool has_ignore_label_, const int ignore_label_,
          float* counts, float* loss) 
{
  CUDA_KERNEL_LOOP(index, num_spatial) 
  {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    int label_value = static_cast<int>(label[index]);
   		
    if (has_ignore_label_ && label_value == ignore_label_) 
    {
      loss[index] = 0;
      counts[index] = 0;
    } 
    else 
    {
      //loss[index] = -logf(max(prob_data[(n * channels + label_value) * spatial_dim + s], float(FLT_MIN)));
      loss[index] = -log(prob_data[(n * channels + label_value) * spatial_dim + s]);
      counts[index] = 1;
    }
  }
}


static __global__ void backward_kernel(const int num_spatial,const int num, const int channels, const int spatial_dim, 
					const float* prob_data, const float* label, 
					const bool has_ignore_label_, const int ignore_label_, 
					float* counts,  float* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index, num_spatial) 
  {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    int label_value = static_cast<int>(label[index]);
		
    if (has_ignore_label_ && label_value == ignore_label_) 
    {
      for (int c = 0; c < channels; ++c) 
        bottom_diff[(n*channels+c)*spatial_dim+s] = 0;       
     	counts[index] = 0;
    } 
    else 
    {
    	for (int c = 0; c < channels; ++c) 
      {
      	int ind = (n*channels+c)*spatial_dim+s;
      	if (c == label_value)
      		bottom_diff[ind] = prob_data[ind] -1;
				else
					bottom_diff[ind] = prob_data[ind];
      }
      counts[index] = 1;
    }
  }
}


static __global__ void secforward_kernel(const int count, const int num, const int channels, const int spatial_dim, 
					const float* prob, const float* label, const float* bottom_sec_diff, const float* sum_secx_p,  float* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index, count) 
  {
  	const int n = index / spatial_dim / channels;
    const int s = index % spatial_dim;
  	bottom_diff[index] = bottom_sec_diff[index]*prob[index] - sum_secx_p[n*spatial_dim+s] *  prob[index];
  }
}

void SoftmaxWithLossLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{

	//LOG(INFO)<<bottom[1]->cpu_data()[0];
		
		
  softmax_layer_->Forward_gpu(softmax_bottom_vec_, softmax_top_vec_);

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  float* loss_data = loss_.mutable_gpu_data();
  float* count_data = counts_.mutable_gpu_data();
 	
 	//for (int i=0;i<bottom[1]->count();i++)
 	//{
 	//	CHECK_GE(bottom[1]->cpu_data()[i],0);
 	//	CHECK_LE(bottom[1]->cpu_data()[i],channels-1);
 	//}
  forward_kernel<<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
  (num*height*width, num, channels, height*width, prob_.gpu_data(), bottom[1]->gpu_data(),  has_ignore_label_, ignore_label_, 
  count_data, loss_data);
  
  float loss, count;
  caffe_gpu_asum(num*height*width, loss_data, &loss);
  caffe_gpu_asum(num*height*width, count_data, &count);
  


	if (count > 0)
  	top[0]->mutable_cpu_data()[0] = loss / count;
  else
  	top[0]->mutable_cpu_data()[0] = 0; 	 	
  
  
  //if (Solver::iter() % 100 == 0 && Caffe::gan_type() == "train_dnet")
  //	LOG(INFO)<<"softmax_loss = "<<top[0]->cpu_data()[0];
 
}


void SoftmaxWithLossLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{

	if (Caffe::second_pass() == false)
	{
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		
		float* count_data = counts_.mutable_gpu_data();

		backward_kernel<<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num*height*width, num, channels, height*width, prob_.gpu_data(), bottom[1]->gpu_data(), has_ignore_label_, ignore_label_, 
		count_data, bottom[0]->mutable_gpu_diff());
		
		//LOG(ERROR)<<prob_.cpu_data()[0]<<", "<<prob_.cpu_data()[1] - 1;
		//LOG(ERROR)<<bottom[0]->cpu_diff()[0]<<", "<<bottom[0]->cpu_diff()[1];
		
		float count;
		caffe_gpu_asum(num*height*width, count_data, &count);
		
		const float loss_weight = top[0]->cpu_diff()[0] / count;
		caffe_gpu_scal(prob_.count(), loss_weight, bottom[0]->mutable_gpu_diff());
		
	}
	else
	{
	}
}

void SoftmaxWithLossLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  compute_sum<<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
  (num*height*width, num, channels, height*width, bottom[0]->gpu_sec_diff(), prob_.gpu_data(), loss_.mutable_gpu_data()); 
  
	secforward_kernel<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(), num, channels, height*width, prob_.gpu_data(), bottom[1]->gpu_data(), bottom[0]->gpu_sec_diff(), loss_.gpu_data(),
	bottom[0]->mutable_gpu_diff());
	
	
	const float loss_weight = top[0]->cpu_diff()[0] / float(num*height*width);
	caffe_gpu_scal(bottom[0]->count(), loss_weight, bottom[0]->mutable_gpu_diff()); 
}

}  // namespace caffe
