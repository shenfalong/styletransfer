#include <cfloat>
#include <vector>

#include "caffe/layers/func/eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void EltwiseLayer::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  CHECK(this->layer_param().eltwise_param().coeff_size() == 0
      || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
      "Eltwise Layer takes one coefficient per bottom blob.";
  CHECK(!(this->layer_param().eltwise_param().operation() == "prod" && this->layer_param().eltwise_param().coeff_size())) 
  << "Eltwise layer only takes coefficients for summation.";
  op_ = this->layer_param_.eltwise_param().operation();
  // Blob-wise coefficients for the elementwise operation.
  coeffs_ = vector<float>(bottom.size(), 1);
  if (this->layer_param().eltwise_param().coeff_size()) 
  {
    for (int i = 0; i < bottom.size(); ++i) 
    {
      coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
    }
  }
  
  backwards_ = vector<bool>(bottom.size(), true);
  if (this->layer_param().eltwise_param().backward_size())
  {
  	for (int i = 0; i < bottom.size(); ++i) 
  	{
      backwards_[i] = this->layer_param().eltwise_param().backward(i);
    }
  }
  stable_prod_grad_ = this->layer_param_.eltwise_param().stable_prod_grad();
}


void EltwiseLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	for (int i=1;i<bottom.size();i++)
	{
		CHECK_EQ(bottom[0]->num(),bottom[i]->num());
		CHECK_EQ(bottom[0]->channels(),bottom[i]->channels());
		CHECK_EQ(bottom[0]->height(),bottom[i]->height());
		CHECK_EQ(bottom[0]->width(),bottom[i]->width());
	}
  
  
  top[0]->ReshapeLike(*bottom[0]);
  // If max operation, we will initialize the vector index part.
  if (this->layer_param_.eltwise_param().operation() == "max" && top.size() == 1) 
    max_idx_.Reshape(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
}



REGISTER_LAYER_CLASS(Eltwise);

}  // namespace caffe
