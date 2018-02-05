#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {




class BatchNormLayer : public Layer {
 public:
  explicit BatchNormLayer(const LayerParameter& param) : Layer(param), handles_setup_(false)  {}
  virtual ~BatchNormLayer();
	virtual inline const char* type() const { return "BatchNorm"; }

	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Backward_gpu(const vector<Blob*>& top,   const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
 	
	
	double number_collect_sample;
	bool is_initialize;
		
  bool handles_setup_;


  Blob * mean_buffer_;
  Blob * var_buffer_;
 
  int gpu_id_;	
};


}  // namespace caffe

