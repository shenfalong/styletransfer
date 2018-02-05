#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/loss/accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe 
{


void AccuracyLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ = this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) 
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
}


void AccuracyLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  //bottom[0]->Reshape(num*channels/2,2,1,1);
  //bottom[1]->Reshape(num*channels/2,1,1,1);
  
  top[0]->Reshape(1,1,1,1);
}






REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe
