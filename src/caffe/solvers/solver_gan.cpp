#include <cstdio>
#include <stdio.h>
#include <string>
#include <vector>
#include "caffe/util/format.hpp"
#include "caffe/solvers/solver_gan.hpp"

#include "caffe/util/io.hpp"


namespace caffe {


SolverGAN::SolverGAN(const SolverParameter& param)
{
  //---------------- initial train net -----------------------
  param_ = param;
  NetParameter g_net_param;
  ReadProtoFromTextFile(param_.g_net(), &g_net_param);

	Caffe::set_bn_state(param_.bn_state());
	Caffe::set_drop_state(param_.drop_state());
	
  vector<shared_ptr<Blob > > net_intput_blobs; 
  net_intput_blobs.clear();
  vector<std::string> net_intput_blob_names; 
  net_intput_blob_names.clear();
  g_net_.reset(new Net(g_net_param, net_intput_blobs, net_intput_blob_names));
  g_net_->set_NetOptimizer(param_.g_net_opt());
  

  int num_output_blobs = g_net_->output_blobs().size();
  
  net_intput_blobs.clear(); 
  net_intput_blob_names.clear();
  net_intput_blobs.resize(num_output_blobs);
  net_intput_blob_names.resize(num_output_blobs);
  CHECK_EQ(g_net_->output_blobs().size(),g_net_->output_blob_indices().size());
	
  for (int i = 0;i < num_output_blobs; i++)
  {
		net_intput_blobs[i] = g_net_->output_blobs()[i];
		net_intput_blob_names[i] = g_net_->blob_names()[g_net_->output_blob_indices()[i]]; 
	}

  NetParameter d_net_param;
  ReadProtoFromTextFile(param_.d_net(), &d_net_param);
  d_net_.reset(new Net(d_net_param, net_intput_blobs, net_intput_blob_names));
  d_net_->set_NetOptimizer(param_.d_net_opt());
	
	
	
	
  this->iter_ = 0;
}


void SolverGAN::Solve(const char* all_state_file) 
{
  LOG(INFO) << "Solving Generative Adversial Networks";


  if (all_state_file)
  {
   	LOG(INFO) << "Restoring previous solver status from " << all_state_file;
    Restore(all_state_file);
  }

  start_iter_ = this->iter_;
	std::vector<float> g_loss;
	std::vector<float> d_loss;
	g_loss.clear();
	d_loss.clear();
	float cur_d_loss = 0;
  float cur_g_loss = 0;
  while (this->iter_ < param_.max_iter())
  {
    g_net_->ClearParamDiffs();
    d_net_->ClearParamDiffs();    
//-----------------------------------------------
		for (int d_iter = 0;d_iter < 1;d_iter ++)
		{	
			Caffe::set_gan_type("train_dnet");			
			g_net_->BcastData();
			g_net_->Forward();



			d_net_->BcastData();
			cur_d_loss = d_net_->Forward();	

			d_net_->Backward();
			d_net_->ReduceDiff();
			

			d_net_->Update(this->iter_, param_.max_iter(), this->iter_ % param_.display() == 0);
			d_net_->ClearParamDiffs();  

			d_loss.push_back(cur_d_loss);
		}
//----------------------------------------------- 
		Caffe::set_gan_type("train_gnet");		
		//-----unecessary-------
		g_net_->Forward();




		d_net_->BcastData();
		cur_g_loss = d_net_->Forward();
	

		d_net_->Backward();
		//-----unecessary-------


	

		g_net_->Backward();
		g_net_->ReduceDiff();

		g_net_->Update(this->iter_, param_.max_iter(), false);
		g_net_->ClearParamDiffs();   
		d_net_->ClearParamDiffs();   
	

		g_loss.push_back(cur_g_loss);	
    this->iter_++;
    
    if (this->iter_ % param_.display() == 0)
    {
    	dispaly_loss(g_loss, d_loss);
    	g_loss.clear(); d_loss.clear();
    }
    if (param_.snapshot() && this->iter_ % param_.snapshot() == 0)
    	Snapshot();
  }
  //at the end of the training, accumulate the batch norm statisticals 
  if (param_.accumulate_batch_norm())
  {
//-----------------------------------------------
  	LOG(INFO)<<"====accumulate statisticals of samples for batch norm=====";
  	Caffe::number_collect_sample = 0;
  	for (int i = 0; i < param_.accumulate_max_iter(); ++i)
		{	
			g_net_->Forward();
			Caffe::number_collect_sample ++;
		}	
		Caffe::number_collect_sample = -1;
  } 
	//Finally, save the model 
	//Snapshot();
	
  LOG(INFO) << "Optimization Done.";
}



void SolverGAN::dispaly_loss(std::vector<float> g_loss,std::vector<float> d_loss) 
{
	LOG(INFO) << "Iteration " << this->iter_ << " ------------";
	float g_sum = 0;
	for (int i=0;i<g_loss.size();i++)
		g_sum += g_loss[i];
	float d_sum = 0;
	for (int i=0;i<d_loss.size();i++)
		d_sum += d_loss[i];
	LOG(INFO) << " g_loss ="<< g_sum/g_loss.size()<<", d_loss ="<< d_sum/d_loss.size();
}

///------------------------------------------------ proto <->  memory------------------------------------------------------

void SolverGAN::Snapshot() 
{
//------------------------------
	NetParameter g_net_param;
  g_net_->ToProto(&g_net_param);
  string g_model_filename = param_.snapshot_prefix() + "_GNet_iter_" +format_int(this->iter_) + ".caffemodel";
  WriteProtoToBinaryFile(g_net_param, g_model_filename);
  
  NetParameter d_net_param;
  d_net_->ToProto(&d_net_param);
  string d_model_filename = param_.snapshot_prefix() + "_DNet_iter_" +format_int(this->iter_) + ".caffemodel";
	WriteProtoToBinaryFile(d_net_param, d_model_filename);
	
	LOG(INFO) << "Snapshotting to binary proto file " << g_model_filename<< " and "<<d_model_filename;
//-----------------------------

//-----------------------------	

	SolverState g_state;
	g_state.set_learned_net(g_model_filename);
	g_net_->StateToProto(g_state.mutable_net_state());
	string g_snapshot_filename = param_.snapshot_prefix() + "_GNet_iter_"+ format_int(this->iter_)+ ".solverstate";
	WriteProtoToBinaryFile(g_state, g_snapshot_filename.c_str());
	
	SolverState d_state;
	d_state.set_learned_net(d_model_filename);
	d_net_->StateToProto(d_state.mutable_net_state());
	string d_snapshot_filename = param_.snapshot_prefix() + "_DNet_iter_"+ format_int(this->iter_)+ ".solverstate";
	WriteProtoToBinaryFile(d_state, d_snapshot_filename.c_str());
	
	LOG(INFO) << "Snapshotting to binary proto file " << g_snapshot_filename<<" and "<<d_snapshot_filename;
//-----------------------------

//-----------------------------
	SolverState all_state;
	all_state.set_iter(this->iter_);
	all_state.set_g_state_file(g_snapshot_filename);
	all_state.set_d_state_file(d_snapshot_filename);
	string all_snapshot_filename = param_.snapshot_prefix() + "_all_iter_"+ format_int(this->iter_)+ ".solverstate";
	WriteProtoToBinaryFile(all_state, all_snapshot_filename.c_str());
	
	LOG(INFO) << "Snapshotting to binary proto file " << all_snapshot_filename;
}



void SolverGAN::Restore(const char* all_state_file) 
{
	SolverState all_state;
	ReadProtoFromBinaryFile(all_state_file, &all_state);
	this->iter_ = all_state.iter();
	const char * g_state_file = all_state.g_state_file().c_str();
	const char * d_state_file = all_state.d_state_file().c_str();
	

	SolverState g_state;
	ReadProtoFromBinaryFile(g_state_file, &g_state);
	NetParameter g_net_param;
  ReadProtoFromBinaryFile(g_state.learned_net().c_str(), &g_net_param);
  g_net_->CopyTrainedLayersFrom(g_net_param);
  g_net_->RestoreState(g_state.net_state());
  
  SolverState d_state;
	ReadProtoFromBinaryFile(d_state_file, &d_state);
	NetParameter d_net_param;
  ReadProtoFromBinaryFile(d_state.learned_net().c_str(), &d_net_param);
  d_net_->CopyTrainedLayersFrom(d_net_param);
  d_net_->RestoreState(d_state.net_state());
}


}  // namespace caffe
