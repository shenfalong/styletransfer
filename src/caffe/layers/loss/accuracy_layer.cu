#include <vector>

#include "caffe/layers/loss/accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



void AccuracyLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	  float accuracy = 0;
  const float* bottom_data = bottom[0]->cpu_data();
  const float* bottom_label = bottom[1]->cpu_data();
  
  int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
  
  vector<float> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);

  int count = 0;
  for (int n = 0; n < num; n++) 
  { 
  	const int label_value = static_cast<int>(bottom_label[n]);
    if (has_ignore_label_ && label_value == ignore_label_) 
      continue;
//step 1 get top-k      
    std::vector<std::pair<float, int> > bottom_data_vector;
    bottom_data_vector.clear();
    for (int c = 0; c < channels; c++) 
      bottom_data_vector.push_back(std::make_pair(bottom_data[n * channels + c], c));
      
    std::partial_sort( bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_, bottom_data_vector.end(), 
    									std::greater<std::pair<float, int> >());   									
//step 2 check if true label is in top-k      									
    for (int k = 0; k < top_k_; k++) 
    {
      if (bottom_data_vector[k].second == label_value) 
      {
        ++accuracy;      
        break;
      }
    }
    ++count;
  }
  top[0]->mutable_cpu_data()[0] = accuracy / count;
}


void AccuracyLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
}

void AccuracyLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}

}  // namespace caffe
