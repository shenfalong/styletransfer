#include <vector>

#include "caffe/layers/loss/parse_output_layer.hpp"

namespace caffe {


void ParseOutputLayer::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  out_max_val_ = top.size() > 1;
}


void ParseOutputLayer::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  // Produces max_ind and max_val
  top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  if (out_max_val_) {
    top[1]->ReshapeLike(*top[0]);
  }
  max_prob_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
}




REGISTER_LAYER_CLASS(ParseOutput);

}  // namespace caffe
