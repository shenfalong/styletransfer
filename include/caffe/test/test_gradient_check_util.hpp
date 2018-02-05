#ifndef CAFFE_TEST_GRADIENT_CHECK_UTIL_H_
#define CAFFE_TEST_GRADIENT_CHECK_UTIL_H_

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"

namespace caffe {

class GradientChecker 
{
 public:
  GradientChecker(const float stepsize, const float threshold, const unsigned int seed = 1701, const float kink = 0., const float kink_range = -1)
      : stepsize_(stepsize), threshold_(threshold), seed_(seed), kink_(kink), kink_range_(kink_range) {}
  
	float GetObjAndGradient(const Layer& layer, const vector<Blob*>& top, int top_id = -1, int top_data_id = -1);
	void CheckGradientSingle(Layer* layer, const vector<Blob*>& bottom, const vector<Blob*>& top, int check_bottom,
				int top_id, int top_data_id);
//-------------------------------	
	float GetObjAndSecGradient(const Layer& layer, const vector<Blob*>& bottom, int bottom_id = -1, int bottom_diff_id = -1);
	void CheckSecGradientSingle_diff(Layer* layer, const vector<Blob*>& top, const vector<Blob*>& bottom, int check_top,
				int bottom_id, int bottom_diff_id);
	void CheckSecGradientSingle_weights(Layer* layer, const vector<Blob*>& top, const vector<Blob*>& bottom,
				int bottom_id, int bottom_diff_id);
	void CheckSecGradientSingle_data(Layer* layer, const vector<Blob*>& top, const vector<Blob*>& bottom, int check_top,
				int bottom_id, int bottom_diff_id);
//-------------------------------	

	void CheckGradientExhaustive(Layer* layer, const vector<Blob*>& bottom, const vector<Blob*>& top, int check_bottom = -1);
	void CheckSecGradientExhaustive(Layer* layer, const vector<Blob*>& top, const vector<Blob*>& bottom, int check_bottom = -1);
 protected:
  float stepsize_;
  float threshold_;
  unsigned int seed_;
  float kink_;
  float kink_range_;
};



float GradientChecker::GetObjAndGradient(const Layer& layer, const vector<Blob*>& top, int top_id, int top_data_id) 
{
  for (int i = 0; i < top.size(); ++i) 
    caffe_set(top[i]->count(), float(0), top[i]->mutable_cpu_diff());

  const float loss_weight = 2;
  float loss = top[top_id]->cpu_data()[top_data_id] * loss_weight;
  top[top_id]->mutable_cpu_diff()[top_data_id] = loss_weight;

  return loss;
}
//-------------------------------------------
float GradientChecker::GetObjAndSecGradient(const Layer& layer, const vector<Blob*>& bottom, int bottom_id, int bottom_diff_id) 
{
  for (int i = 0; i < bottom.size(); ++i) 
    caffe_gpu_set(bottom[i]->count(), float(0), bottom[i]->mutable_gpu_sec_diff());
    
  const float loss_weight = 2;
  float loss = bottom[bottom_id]->cpu_diff()[bottom_diff_id] * loss_weight;
  bottom[bottom_id]->mutable_cpu_sec_diff()[bottom_diff_id] = loss_weight;

  return loss;
}
//-------------------------------------------
void GradientChecker::CheckGradientSingle(Layer* layer, const vector<Blob*>& bottom, const vector<Blob*>& top, int check_bottom, 
			int top_id, int top_data_id) 
{
  vector<Blob*> blobs_to_check;
  for (int i = 0; i < layer->blobs().size(); ++i) 
  {
    caffe_set(layer->blobs()[i]->count(), static_cast<float>(0), layer->blobs()[i]->mutable_cpu_diff());   
    if(i == -1)  
    blobs_to_check.push_back(layer->blobs()[i].get());
  }
  if (check_bottom == -1) 
  {
    for (int i = 0; i < bottom.size(); ++i) 
   		blobs_to_check.push_back(bottom[i]);
  } 
  else if (check_bottom >= 0)
  {
    CHECK_LT(check_bottom, bottom.size());
    blobs_to_check.push_back(bottom[check_bottom]);
  }
  CHECK_GT(blobs_to_check.size(), 0) << "No blobs to check.";



	layer->Reshape(bottom, top);
  layer->Forward(bottom, top);

  GetObjAndGradient(*layer, top, top_id, top_data_id);
  layer->Backward(top, bottom);


  for (int blob_id = 0; blob_id < blobs_to_check.size(); ++blob_id) 
  {
    Blob* current_blob = blobs_to_check[blob_id];
    const float* computed_gradients = current_blob->cpu_diff();

    for (int feat_id = 0; feat_id < current_blob->count(); ++feat_id) 
    {

      float estimated_gradient = 0;
      float positive_objective = 0;
      float negative_objective = 0;
      
      current_blob->mutable_cpu_data()[feat_id] += stepsize_;
      layer->Reshape(bottom, top);
      layer->Forward(bottom, top);
      positive_objective = GetObjAndGradient(*layer, top, top_id, top_data_id);

      current_blob->mutable_cpu_data()[feat_id] -= stepsize_ * 2;
      layer->Reshape(bottom, top);
      layer->Forward(bottom, top);
      negative_objective = GetObjAndGradient(*layer, top, top_id, top_data_id);

      current_blob->mutable_cpu_data()[feat_id] += stepsize_;
      estimated_gradient = (positive_objective - negative_objective) / stepsize_ / 2.;
    
      float computed_gradient = computed_gradients[feat_id];
      float feature = current_blob->cpu_data()[feat_id];

      if (kink_ - kink_range_ > fabs(feature) || fabs(feature) > kink_ + kink_range_) 
      {
        float scale = std::max<float>( std::max(fabs(computed_gradient), fabs(estimated_gradient)), float(1.));
        EXPECT_NEAR(computed_gradient, estimated_gradient, threshold_ * scale)
          << "debug: (top_id, top_data_id, blob_id, feat_id)="
          << top_id << "," << top_data_id << "," << blob_id << "," << feat_id
          << "; feat = " << feature
          << "; objective+ = " << positive_objective
          << "; objective- = " << negative_objective;
      }
    }
  }
}
//-------------------------------------------

void GradientChecker::CheckSecGradientSingle_diff(Layer* layer, const vector<Blob*>& top, const vector<Blob*>& bottom, int check_top, 
			int bottom_id, int bottom_diff_id) 
{
  vector<Blob*> blobs_to_check;
  if (check_top == -1) 
  {
    for (int i = 0; i < top.size(); ++i) 
   		blobs_to_check.push_back(top[i]);
  } 
  else if (check_top >= 0)
  {
    CHECK_LT(check_top, top.size());
    blobs_to_check.push_back(top[check_top]);
  }
  CHECK_GT(blobs_to_check.size(), 0) << "No blobs to check.";
	
	caffe_rng_gaussian(top[0]->count(), float(0), float(1), top[0]->mutable_cpu_diff());
	
	
  layer->Backward(top, bottom);

  GetObjAndSecGradient(*layer, bottom, bottom_id, bottom_diff_id);
  layer->SecForward(bottom, top);


  for (int blob_id = 0; blob_id < blobs_to_check.size(); ++blob_id) 
  {
    Blob* current_blob = blobs_to_check[blob_id];
    const float* computed_gradients = current_blob->cpu_sec_diff();

    for (int feat_id = 0; feat_id < current_blob->count(); ++feat_id) 
    {

      float estimated_gradient = 0;
      float positive_objective = 0;
      float negative_objective = 0;
     

      current_blob->mutable_cpu_diff()[feat_id] += stepsize_;
      layer->Backward(top, bottom);
      positive_objective = GetObjAndSecGradient(*layer, bottom, bottom_id, bottom_diff_id);

      current_blob->mutable_cpu_diff()[feat_id] -= stepsize_ * 2;
      layer->Backward(top, bottom);
      negative_objective = GetObjAndSecGradient(*layer, bottom, bottom_id, bottom_diff_id);

      current_blob->mutable_cpu_diff()[feat_id] += stepsize_;//change back
      estimated_gradient = (positive_objective - negative_objective) / stepsize_ / 2.;
      
      float computed_gradient = computed_gradients[feat_id];
      float feature = current_blob->cpu_diff()[feat_id];

      if (kink_ - kink_range_ > fabs(feature) || fabs(feature) > kink_ + kink_range_) 
      {
        float scale = std::max<float>( std::max(fabs(computed_gradient), fabs(estimated_gradient)), float(1.));
        EXPECT_NEAR(computed_gradient, estimated_gradient, threshold_ * scale)
          << "debug: (bottom_id, bottom_diff_id, blob_id, feat_id)="
          << bottom_id << "," << bottom_diff_id << "," << blob_id << "," << feat_id
          << "; feat = " << feature
          << "; objective+ = " << positive_objective
          << "; objective- = " << negative_objective;
      }
    }
  }
}

void GradientChecker::CheckSecGradientSingle_weights(Layer* layer, const vector<Blob*>& top, const vector<Blob*>& bottom, 
			int bottom_id, int bottom_diff_id) 
{
  vector<Blob*> blobs_to_check;
  for (int i = 0; i < layer->blobs().size(); ++i) 
  {
    caffe_set(layer->blobs()[i]->count(), static_cast<float>(0), layer->blobs()[i]->mutable_cpu_diff());   
    if(i == 1)  
    	blobs_to_check.push_back(layer->blobs()[i].get());
  }
  if (blobs_to_check.size() == 0) 
  {
  	LOG(ERROR)<< "No blobs to check.";
		return;
	}
	
	caffe_rng_gaussian(top[0]->count(), float(0), float(1), top[0]->mutable_cpu_diff());

	
	
	Caffe::set_frozen_param(true);
  layer->Backward(top, bottom);
	Caffe::set_frozen_param(false);
	
  GetObjAndSecGradient(*layer, bottom, bottom_id, bottom_diff_id);
  layer->SecForward(bottom, top);
	Caffe::set_frozen_param(true);
	
  for (int blob_id = 0; blob_id < blobs_to_check.size(); ++blob_id) 
  {
    Blob* current_blob = blobs_to_check[blob_id];
    const float* computed_gradients = current_blob->cpu_diff();

    for (int feat_id = 0; feat_id < current_blob->count(); ++feat_id) 
    {
      float estimated_gradient = 0;
      float positive_objective = 0;
      float negative_objective = 0;
     
      current_blob->mutable_cpu_data()[feat_id] += stepsize_;
      layer->Backward(top, bottom);
      positive_objective = GetObjAndSecGradient(*layer, bottom, bottom_id, bottom_diff_id);

      current_blob->mutable_cpu_data()[feat_id] -= stepsize_ * 2;
      layer->Backward(top, bottom);
      negative_objective = GetObjAndSecGradient(*layer, bottom, bottom_id, bottom_diff_id);

      current_blob->mutable_cpu_data()[feat_id] += stepsize_;//change back
      estimated_gradient = (positive_objective - negative_objective) / stepsize_ / 2.;
      
      float computed_gradient = computed_gradients[feat_id];
      float feature = current_blob->cpu_data()[feat_id];

      if (kink_ - kink_range_ > fabs(feature) || fabs(feature) > kink_ + kink_range_) 
      {
        float scale = std::max<float>( std::max(fabs(computed_gradient), fabs(estimated_gradient)), float(1.));
        EXPECT_NEAR(computed_gradient, estimated_gradient, threshold_ * scale)
          << "debug: (bottom_id, bottom_data_id, blob_id, feat_id)="
          << bottom_id << "," << bottom_diff_id << "," << blob_id << "," << feat_id
          << "; feat = " << feature
          << "; objective+ = " << positive_objective
          << "; objective- = " << negative_objective;
      }
    }
  }
}

void GradientChecker::CheckSecGradientSingle_data(Layer* layer, const vector<Blob*>& top, const vector<Blob*>& bottom, int check_bottom, 
			int bottom_id, int bottom_diff_id) 
{
  vector<Blob*> blobs_to_check;
  if (check_bottom == -1) 
  {
    for (int i = 0; i < bottom.size(); ++i) 
   		blobs_to_check.push_back(bottom[i]);
  } 
  else if (check_bottom >= 0)
  {
    CHECK_LT(check_bottom, bottom.size());
    blobs_to_check.push_back(bottom[check_bottom]);
  }
  CHECK_GT(blobs_to_check.size(), 0) << "No blobs to check.";
	
	caffe_rng_gaussian(top[0]->count(), float(0), float(1), top[0]->mutable_cpu_diff());
	//top[0]->mutable_cpu_diff()[0]=1;
	//top[0]->mutable_cpu_diff()[1]=1;
	//caffe_gpu_set(top[0]->count(),float(0),top[0]->mutable_gpu_diff());

	
	layer->Forward(bottom, top);
  layer->Backward(top, bottom);
	
  GetObjAndSecGradient(*layer, bottom, bottom_id, bottom_diff_id);
  layer->SecForward(bottom, top);

	vector<Blob *> gradient_blob;
	gradient_blob.resize(bottom.size());
	for (int i=0;i<blobs_to_check.size();i++)	
	{
		gradient_blob[i] = new Blob();
		gradient_blob[i]->ReshapeLike(*blobs_to_check[i]);
		caffe_copy(blobs_to_check[i]->count(),blobs_to_check[i]->cpu_diff(),gradient_blob[i]->mutable_cpu_diff());
	}
	
  for (int blob_id = 0; blob_id < blobs_to_check.size(); ++blob_id) 
  {
    Blob* current_blob = blobs_to_check[blob_id];
   	const float* computed_gradients = gradient_blob[blob_id]->cpu_diff();
   	
    for (int feat_id = 0; feat_id < current_blob->count(); ++feat_id) 
    {

      float estimated_gradient = 0;
      float positive_objective = 0;
      float negative_objective = 0;
     

      current_blob->mutable_cpu_data()[feat_id] += stepsize_;
      layer->Forward(bottom, top);
      layer->Backward(top, bottom);
      positive_objective = GetObjAndSecGradient(*layer, bottom, bottom_id, bottom_diff_id);

      current_blob->mutable_cpu_data()[feat_id] -= stepsize_ * 2;
      layer->Forward(bottom, top);
      layer->Backward(top, bottom);
      negative_objective = GetObjAndSecGradient(*layer, bottom, bottom_id, bottom_diff_id);

      current_blob->mutable_cpu_data()[feat_id] += stepsize_;//change back
      estimated_gradient = (positive_objective - negative_objective) / stepsize_ / 2.;
      
      float computed_gradient = computed_gradients[feat_id];
      float feature = current_blob->cpu_data()[feat_id];

      if (kink_ - kink_range_ > fabs(feature) || fabs(feature) > kink_ + kink_range_) 
      {
        float scale = std::max<float>( std::max(fabs(computed_gradient), fabs(estimated_gradient)), float(1.));
        EXPECT_NEAR(computed_gradient, estimated_gradient, threshold_ * scale)
          << "debug: (bottom_id, bottom_diff_id, blob_id, feat_id)="
          << bottom_id << "," << bottom_diff_id << "," << blob_id << "," << feat_id
          << "; feat = " << feature
          << "; objective+ = " << positive_objective
          << "; objective- = " << negative_objective;
      }
    }
  }
  for (int i=0;i<blobs_to_check.size();i++)	
  	delete gradient_blob[i];
}
//-------------------------------------------

void GradientChecker::CheckGradientExhaustive(Layer* layer, const vector<Blob*>& bottom, const vector<Blob*>& top, int check_bottom) 
{
  CHECK_GT(top.size(), 0) << "Exhaustive mode requires at least one top blob.";
  for (int i = 0; i < top.size(); ++i) 
  {
    for (int j = 0; j < top[i]->count(); ++j) 
      CheckGradientSingle(layer, bottom, top, check_bottom, i, j);
  }
}


void GradientChecker::CheckSecGradientExhaustive(Layer* layer,const vector<Blob*>& top, const vector<Blob*>& bottom, int check_top) 
{
  CHECK_GT(bottom.size(), 0) << "Exhaustive mode requires at least one bottom blob.";
  for (int i = 0; i < bottom.size(); ++i) 
  {
    for (int j = 0; j < bottom[i]->count(); ++j) 
    {
      //CheckSecGradientSingle_diff(layer, top, bottom, check_top, i, j);
      CheckSecGradientSingle_data(layer, top, bottom, check_top, i, j);
      //CheckSecGradientSingle_weights(layer, top, bottom, i, j);
    }
  }
}

}  // namespace caffe

#endif  // CAFFE_TEST_GRADIENT_CHECK_UTIL_H_
