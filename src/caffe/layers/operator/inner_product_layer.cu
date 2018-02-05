#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/operator/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void InnerProductLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{ 
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();

	const float* bottom_data = bottom[0]->gpu_data();
	float* top_data = top[0]->mutable_gpu_data();
	const float* weight = this->blobs_[0]->gpu_data();
	
	caffe_gpu_gemm(CblasNoTrans, CblasTrans, num, num_output, channels*height*width, 
												(float)1., bottom_data, weight, 
												(float)0., top_data);
									
	if (this->layer_param_.inner_product_param().bias_term())
	{
	  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num, num_output, 1, 
	  										(float)1., bias_multiplier_.gpu_data(), this->blobs_[1]->gpu_data(), 
	  										(float)1., top_data);
	}  					  
}


void InnerProductLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
  int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width(); 
     
  const float* top_diff = top[0]->gpu_diff();
  const float* bottom_data = bottom[0]->gpu_data();
  float * bottom_diff = bottom[0]->mutable_gpu_diff();
  
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num, channels*height*width, num_output, 
											(float)1., top_diff, this->blobs_[0]->gpu_data(), 
											(float)0., bottom_diff);
	if (this->lr_mult()[0] > 0 && Caffe::frozen_param() == false)
	{
		caffe_gpu_gemm(CblasTrans, CblasNoTrans, num_output, channels*height*width, num, 
													(float)1., top_diff, bottom_data, 
													(float)1., this->blobs_[0]->mutable_gpu_diff());
	}
  if (this->layer_param_.inner_product_param().bias_term() && this->lr_mult()[1] > 0 && Caffe::frozen_param() == false) 
  {
    const float* top_diff = top[0]->gpu_diff();
    
    caffe_gpu_gemv(CblasTrans, num, num_output, 
    											(float)1., top_diff, bias_multiplier_.gpu_data(), 
    											(float)1., this->blobs_[1]->mutable_gpu_diff());
  }

#if 0
  #if 0
  float sum = caffe_gpu_square_sum(this->blobs_[0]->count(),this->blobs_[0]->gpu_data());
  float var = float(2) /((channels*height*width+num_output)/float(2));
  float coef = float(1e-4)*(sum - float(this->blobs_[0]->count())*var);
  //LOG(INFO)<<"weight norm = "<<sum <<" vs "<< float(this->blobs_[0]->count())*var;

  caffe_gpu_add(this->blobs_[0]->count(),float(1),this->blobs_[0]->gpu_diff(),
  																			 coef,   this->blobs_[0]->gpu_data(),
  																			 this->blobs_[0]->mutable_gpu_diff());   
  #else
  float sum = caffe_gpu_square_sum(this->blobs_[0]->count(),this->blobs_[0]->gpu_data());
  float var = float(2) /((channels*height*width+num_output)/float(2));
  float coef =  float(1e-4) * (sqrt(sum) - sqrt(float(this->blobs_[0]->count())*var)) / sqrt(sum);
  //float coef = float(1e-2);
  //if (sum > float(this->blobs_[0]->count())*var)
 	//	coef *= float(1);
 	//else
 	//	coef *= float(-1);
	//LOG(INFO)<<"weight norm = "<<(sum - float(this->blobs_[0]->count())*var);
	
  caffe_gpu_add(this->blobs_[0]->count(),float(1),this->blobs_[0]->gpu_diff(),
  																			 coef,   this->blobs_[0]->gpu_data(),
  																			 this->blobs_[0]->mutable_gpu_diff());    	
  #endif		
#endif  																	         																			 
}

void InnerProductLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{ 
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	caffe_gpu_gemm(CblasNoTrans, CblasTrans, num, num_output, channels*height*width, 
											(float)1., bottom[0]->gpu_sec_diff(), this->blobs_[0]->gpu_data(), 
											(float)0., top[0]->mutable_gpu_sec_diff());	
	if (this->lr_mult()[0] > 0 && Caffe::frozen_param() == false)
	{	
		caffe_gpu_gemm(CblasTrans, CblasNoTrans, num_output, channels*height*width, num, 
												(float)1., top[0]->gpu_diff(), bottom[0]->gpu_sec_diff(),
												(float)1., this->blobs_[0]->mutable_gpu_diff());
	}							
}


}  // namespace caffe
