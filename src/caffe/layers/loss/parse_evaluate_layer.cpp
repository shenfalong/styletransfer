#include <vector>

#include "caffe/layers/loss/parse_evaluate_layer.hpp"

namespace caffe {


void ParseEvaluateLayer::LayerSetUp(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const ParseEvaluateParameter& parse_evaluate_param =
      this->layer_param_.parse_evaluate_param();
  CHECK(parse_evaluate_param.has_num_labels()) << "Must have num_labels!!";
  num_labels_ = parse_evaluate_param.num_labels();
  ignore_labels_.clear();
  int num_ignore_label = parse_evaluate_param.ignore_label().size();
  for (int i = 0; i < num_ignore_label; ++i) {
    ignore_labels_.insert(parse_evaluate_param.ignore_label(i));
  }
}


void ParseEvaluateLayer::Reshape(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_GE(bottom[0]->width(), bottom[1]->width());
  top[0]->Reshape(1, num_labels_, 1, 3);
}




REGISTER_LAYER_CLASS(ParseEvaluate);

}  // namespace caffe
