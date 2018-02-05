
#include <vector>

#include "caffe/layers/activation/reshape_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



void ReshapeLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}


void ReshapeLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
}

void ReshapeLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}

}  // namespace caffe
