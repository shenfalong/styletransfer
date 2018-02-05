#ifndef CAFFE_SOLVERCNN_HPP_
#define CAFFE_SOLVERCNN_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver.hpp"

namespace caffe 
{


class SolverCNN : public Solver
{
 public:
 	explicit SolverCNN(const SolverParameter& param);
  void Solve(const char* resume_file = NULL);
  void dispaly_loss(float loss);
  
  
 	void Snapshot();
	void test();
	void testSegmentation();
  void Restore(const char* resume_file);
	virtual ~SolverCNN() 
  {}

  
  inline const SolverParameter& param() const { return param_; }
  inline shared_ptr<Net > net() { return net_; }
 protected:
  SolverParameter param_;
  shared_ptr<Net > net_;
  shared_ptr<Net > test_net_;



	int start_iter_;
	float sum_loss_;


  DISABLE_COPY_AND_ASSIGN(SolverCNN);
};


}  // namespace caffe

#endif  // CAFFE_SOLVERCNN_HPP_
