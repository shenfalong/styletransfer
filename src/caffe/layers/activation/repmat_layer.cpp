
#include <vector>

#include "caffe/layers/activation/repmat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void RepmatLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
}


void RepmatLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height;
  int width;
  if (bottom.size() == 2)
  {
		height = bottom[1]->height();
		width = bottom[1]->width();
	}
	else
	{
		height = this->layer_param_.shape_param().height();
		width = this->layer_param_.shape_param().width();
	}
//----------------------------- -------------------------------------  
  one_multiplier_ = static_cast<Blob *>(Caffe::parallel_workspace_[gpu_id_+Caffe::GPUs.size()]);
//----------------------------- -------------------------------------   
  one_multiplier_->Reshape(1,1,height,width);
	top[0]->Reshape(num,channels,height,width);
}


REGISTER_LAYER_CLASS(Repmat);
}  // namespace caffe
