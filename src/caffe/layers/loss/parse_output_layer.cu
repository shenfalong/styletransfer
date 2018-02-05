#include <vector>

#include "caffe/layers/loss/parse_output_layer.hpp"

namespace caffe {



void ParseOutputLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
   const float* bottom_data = bottom[0]->cpu_data();
  float* top_label_data = top[0]->mutable_cpu_data();
  float* top_prob_data = NULL;
  if (out_max_val_) {
    top_prob_data = top[1]->mutable_cpu_data();
  }
  float* max_prob_data = max_prob_.mutable_cpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  for (int i = 0; i < num; ++i) {
    caffe_set(spatial_dim, float(0), top_label_data);
    // initialize max value from first plane
    caffe_copy(spatial_dim, bottom_data, max_prob_data);
    for (int j = 1; j < channels; ++j) {
      bottom_data += bottom[0]->offset(0, 1);
      for (int k = 0; k < spatial_dim; ++k) {
        float prob = bottom_data[k];
        if (prob > max_prob_data[k]) {
          max_prob_data[k] = prob;
          top_label_data[k] = j;
        }
      }
    }
    top_label_data += top[0]->offset(1);
    if (out_max_val_) {
      caffe_copy(spatial_dim, max_prob_data, top_prob_data);
      top_prob_data += top[1]->offset(1);
    }
  }
//LOG(INFO)<<" data_height = "<<480;
//FILE *fid = fopen("debug","wb");
//fwrite(top[0]->cpu_data(),sizeof(float), 480 * 480,fid);
//fclose(fid);
//LOG(FATAL)<<" data_height = "<<480;
}


void ParseOutputLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
}

void ParseOutputLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}


}  // namespace caffe
