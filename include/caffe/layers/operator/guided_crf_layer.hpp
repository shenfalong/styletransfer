#ifndef CAFFE_GuidedCRF_LAYER_HPP_
#define CAFFE_GuidedCRF_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

// add shen

class GuidedCRFLayer : public Layer 
{
 public:
  explicit GuidedCRFLayer(const LayerParameter& param) : Layer(param) {}
  virtual ~GuidedCRFLayer();
  virtual inline const char* type() const { return "GuidedCRF"; }
  
  
  virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);

  
 protected:
	
	void guided_filter_gpu(const int num,const int channels,const int maxStates,const int height,const int width,const float *I,const float * p,float *output_p);
	
	int gpu_id_;
  
 	int radius;
  int maxIter;
  float area;
  float alpha;
  float eps;
  Blob compatPot;  
  Blob filterPot;  
  Blob tempPot; 
  std::vector< Blob * > nodeBel; 

  Blob mean_I;
  Blob II;
  Blob mean_II;
  Blob var_I;
  Blob mean_p;
  Blob b;
  Blob mean_b;
  Blob inv_var_I;
  Blob buffer_image;
  Blob buffer_score;
  Blob buffer_image_image;
  Blob output_p1;
  Blob output_p2;

//---------------------------------------  
  float * a;
  float * mean_a;
  float * cov_Ip;
  float * Ip;
  float * mean_Ip;
  float * buffer_image_score;
//---------------------------------------

	vector<Blob *> myworkspace_;
};

}  // namespace caffe

#endif  // CAFFE_GuidedCRF_LAYER_HPP_
