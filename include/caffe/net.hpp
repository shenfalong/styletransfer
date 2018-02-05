// Copyright 2013 Yangqing Jia

#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "boost/scoped_ptr.hpp"
#include <boost/thread.hpp>

using std::map;
using std::vector;
using std::string;

namespace caffe {


class Net
{
 public:
	Net(const NetParameter& param, const vector<shared_ptr<Blob > > net_input_blobs, const vector<string> net_input_blob_names);
  ~Net();
  float Forward();
  void Backward();
  void SecForward();

	void Update(int iter, int max_iter, bool display);

  void CopyTrainedLayersFrom(const NetParameter& param);
	void ClearParamDiffs();
  void ToProto(NetParameter* param, bool write_diff = false);
	

  inline const string& name() { return name_; }
  inline const vector<shared_ptr<Layer > >& layers() { return layers_; }
  inline const vector<string>& layer_names() { return layer_names_; }
  
  inline const vector<shared_ptr<Blob > >& blobs() { return blobs_; }
  inline const vector<string>& blob_names() { return blob_names_; }
	inline const vector<float>& blob_loss_weights() const { return blob_loss_weights_; }
	inline vector<int>& output_blob_indices() { return output_blob_indices_; }  
	inline const vector<int>& input_blob_indices() const { return input_blob_indices_; }
	
	void BcastData();
	void ReduceDiff();
	void ScaleDiff(const float scalar);
	 
  inline const vector<shared_ptr<Blob > >& output_blobs() const { return output_blobs_; }
  inline const vector<shared_ptr<Blob > >& input_blobs() const { return input_blobs_; }

  inline vector<vector<Blob*> >& bottom_vecs() { return bottom_vecs_; }
  inline vector<vector<Blob*> >& top_vecs() { return top_vecs_; }
	
	

	
	float GetLearningRate(int iter, int max_iter);
	void Snapshot();
	
	void StateToProto(NetState * state);
	void RestoreState(const NetState  state);
	void set_NetOptimizer(const NetOptimizer net_opt) { optimizer_ = net_opt;}
 protected:
 	vector<bool> layer_need_backward_;
 	
  vector<shared_ptr<Layer > > layers_;
	vector<string> layer_names_;
	
  vector<shared_ptr<Blob > > blobs_;
  vector<string> blob_names_;
  vector<float> blob_loss_weights_;
  vector<int> input_blob_indices_;
  vector<int> output_blob_indices_;
  
  
  vector<vector<Blob*> > bottom_vecs_;
  vector<vector<int> > bottom_id_vecs_;
  vector<vector<Blob*> > top_vecs_;
  vector<vector<int> > top_id_vecs_;
  
 	vector<shared_ptr<Blob > > tensor_flows_;
  vector<shared_ptr<Blob > > tensor_flows_temp_;
  
	vector<shared_ptr<Blob> > output_blobs_;
	vector<shared_ptr<Blob> > input_blobs_;
	
  string name_;

	
	NetParameter param_;
	NetOptimizer optimizer_;
	
	vector<bool> flow_flag;
	int adam_iter_;
	float  momentum_power_;
	float momentum2_power_;
	
	shared_ptr<boost::thread> thread_;
	
  DISABLE_COPY_AND_ASSIGN(Net);
};


}  // namespace caffe

#endif  // CAFFE_NET_HPP_
