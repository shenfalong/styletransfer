
#include <vector>

#include "caffe/layers/func/lambda_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



void LambdaLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	all_layer_[layer_index_]->Forward_gpu(unary_bottom_vec_[layer_index_], top);
}


void LambdaLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	all_layer_[layer_index_]->Backward_gpu(top, unary_bottom_vec_[layer_index_]);
}


void LambdaLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	all_layer_[layer_index_]->SecForward_gpu(unary_bottom_vec_[layer_index_], top);
}


}  // namespace caffe
