#include <cstdio>
#include <stdio.h>
#include <string>
#include <vector>
#include "caffe/util/format.hpp"
#include "caffe/solvers/solver_cnn.hpp"

#include "caffe/util/io.hpp"


namespace caffe {



SolverCNN::SolverCNN(const SolverParameter& param)
{
  //---------------- initial train net -----------------------
  param_ = param;
  NetParameter net_param;
  ReadProtoFromTextFile(param_.net(), &net_param);
  

  vector<shared_ptr<Blob > > net_intput_blobs; 
  net_intput_blobs.clear();
  vector<std::string> net_intput_blob_names; 
  net_intput_blob_names.clear();
  net_.reset(new Net(net_param, net_intput_blobs, net_intput_blob_names));
  net_->set_NetOptimizer(param_.net_opt());
  
  this->iter_ = 0;

	Caffe::set_bn_state(param_.bn_state());
	Caffe::set_drop_state(param_.drop_state());
 	

  //---------------- initial test net -----------------------
  if (param_.has_test_initialization())
  {
  	net_param.Clear();
  	ReadProtoFromTextFile(param_.test_net(), &net_param);
  	
    net_intput_blobs.clear();
    net_intput_blob_names.clear();
    test_net_.reset(new Net(net_param, net_intput_blobs, net_intput_blob_names));
    this->share_weight(net_, test_net_);
  }
}


void SolverCNN::Solve(const char* resume_file) 
{
  LOG(INFO) << "Solving " << net_->name();


  if (resume_file)
  {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }
	
	 	
	
  start_iter_ = this->iter_;
  sum_loss_ = 0;
	
  if (param_.has_test_initialization() && param_.test_initialization())
  {
    //-------------- copy net-------------------
		test_net_->BcastData();
    if (param_.eval_type() == "classification")
    	test();
    else if (param_.eval_type() == "segmentation")
    	testSegmentation();
    else
    	LOG(FATAL)<<" unrecognized mode for test";
  }
	
	
	
  while (this->iter_ < param_.max_iter())
  {
    net_->ClearParamDiffs();
    float loss = 0;
    

		net_->BcastData();
    for (int i = 0; i < param_.iter_size(); ++i)
    {
    	
      loss  += net_->Forward();    
      Caffe::set_reuse(true);
      net_->Backward();
			Caffe::set_reuse(false);
//-------------
#if 0
      Caffe::set_gradient_penalty("Yes");
      Caffe::set_frozen_param(true);
			Caffe::set_reuse(false);    
			net_->Backward();
			Caffe::set_frozen_param(false);
			
			Caffe::set_second_pass(true);
			net_->SecForward();
			Caffe::set_second_pass(false);
			Caffe::set_gradient_penalty("No");
#endif			
//-------------			
    }
		net_->ReduceDiff();
   	net_->ScaleDiff(float(1)/float(param_.iter_size()));	
		
		loss /= float(param_.iter_size());	
    dispaly_loss(loss);
			
 		
		net_->Update(this->iter_, param_.max_iter(), this->iter_ % param_.display() == 0);
    ++this->iter_;
		
    if (param_.has_test_interval() && this->iter_ % param_.test_interval() == 0)
    {
    
			net_->BcastData();			
    	if (param_.accumulate_batch_norm())
			{
				LOG(INFO)<<"==accumulate statisticals of samples for batch norm====";
				Caffe::number_collect_sample = 0;
				for (int i = 0; i < param_.accumulate_test_iter(); ++i)
				{	
					net_->Forward();
					Caffe::number_collect_sample ++;
				}	
				Caffe::number_collect_sample = -1;
			}
    	
    	LOG(INFO)<<"===== test the model ======";
		//-------------- copy net-------------------
			test_net_->BcastData(); 
      if (param_.eval_type() == "classification")
      	test();
      else if (param_.eval_type() == "segmentation")
      	testSegmentation();
      else
      	LOG(FATAL)<<" unrecognized mode for test";
    }
    if (param_.snapshot() && this->iter_ % param_.snapshot() == 0 )
    	Snapshot();
  }
  //at the end of the training, accumulate the batch norm statisticals 
  if (param_.accumulate_batch_norm())
  {
		net_->BcastData();
  	LOG(INFO)<<"==== accumulate statisticals of samples for batch norm =====";
  	Caffe::number_collect_sample = 0;
  	for (int i = 0; i < param_.accumulate_max_iter(); ++i)
		{	
			net_->Forward();
			Caffe::number_collect_sample ++;
		}	
		Caffe::number_collect_sample = -1;
  }
	//at the end of the training, run one test
	if (param_.has_test_interval())
  {
  	LOG(INFO)<<"===== test the model ======";
    //-------------- copy net-------------------
		test_net_->BcastData();
    if (param_.eval_type() == "classification")
    	test();
    else if (param_.eval_type() == "segmentation")
    	testSegmentation();
    else
    	LOG(FATAL)<<" unrecognized mode for test";   
  } 
	//Finally, save the model 
	Snapshot();
	
  LOG(INFO) << "Optimization Done.";
}



void SolverCNN::dispaly_loss(float loss) 
{
	sum_loss_ += loss;
	
	if (param_.display() && this->iter_ % param_.display() == 0 && this->iter_ != 0)	
  {
  	LOG(INFO) << "Iteration " << this->iter_  << ", loss = " << sum_loss_ / param_.display();
  	sum_loss_ = 0;
  }
}

void SolverCNN::test() 
{
	Caffe::set_bn_state("frozen");
	Caffe::set_drop_state("fixed");
	
  float test_score = 0;
  for (int i = 0; i < param_.test_iter(); i++)
  {
  	Caffe::set_reuse(true);
    test_net_->Forward();
    CHECK_EQ(test_net_->output_blobs().size(),NGPUS);
    CHECK_EQ(test_net_->output_blobs()[0]->count(),1);
    
    for (int j=0;j<NGPUS;j++)
    	test_score +=  test_net_->output_blobs()[j]->cpu_data()[0];
    
  }

  const std::string& output_name = test_net_->blob_names()[test_net_->output_blob_indices()[0]];

  const float mean_score = test_score / (param_.test_iter() * NGPUS);
  LOG(INFO) << output_name << " = " << mean_score;
  
  Caffe::set_bn_state(param_.bn_state());
	Caffe::set_drop_state(param_.drop_state());
	Caffe::set_reuse(false);
}


void SolverCNN::testSegmentation() 
{
	Caffe::set_bn_state("frozen");
	Caffe::set_drop_state("fixed");
	
	
  LOG(INFO) << "Iteration " << this->iter_ << ", Testing net ";
  
	shared_ptr<Blob > label_stat(new Blob());
	
  for (int i = 0; i < param_.test_iter(); i++) 
  {
  	Caffe::set_reuse(true);
    test_net_->Forward();

    
    CHECK_EQ(test_net_->output_blobs().size(),NGPUS);

		REDUCE_DATA(test_net_->output_blobs());
    if (i == 0) 
    {
      label_stat->Reshape(1, test_net_->output_blobs()[0]->channels(), test_net_->output_blobs()[0]->height(), test_net_->output_blobs()[0]->width());
      caffe_copy(test_net_->output_blobs()[0]->count(), test_net_->output_blobs()[0]->cpu_data(), label_stat->mutable_cpu_data());
    } 
    else 
    {        
      caffe_axpy(test_net_->output_blobs()[0]->count(), 
      					float(1), test_net_->output_blobs()[0]->cpu_data(), 
      					label_stat->mutable_cpu_data());
    }    
  }


 
  const int output_blob_index = test_net_->output_blob_indices()[0];
  const string& output_name = test_net_->blob_names()[output_blob_index];
  const float* label_stat_data = label_stat->cpu_data();
  const int channels = label_stat->channels();
  // get sum infomation
  float sum_gtpred = 0;
  float sum_gt = 0;
  for (int c = 0; c < channels; ++c) 
  {
    sum_gtpred += label_stat_data[c*3];
    sum_gt += label_stat_data[c*3+1];
  }
  if (sum_gt > 0) 
  {
    float per_pixel_acc = sum_gtpred / sum_gt;
    float per_label_acc = 0, iou, iou_acc = 0, weighted_iou_acc = 0;
    int num_valid_labels = 0;
    for (int c = 0; c < channels; ++c) 
    {
      if (label_stat_data[1] != 0) 
      {
        per_label_acc += label_stat_data[0] / label_stat_data[1];
        ++num_valid_labels;
      }
      if (label_stat_data[1] + label_stat_data[2] != 0) 
      {
        iou = label_stat_data[0] / (label_stat_data[1] + label_stat_data[2] - label_stat_data[0]);
        iou_acc += iou;
        weighted_iou_acc += iou * label_stat_data[1] / sum_gt;
      }
      label_stat_data += label_stat->offset(0, 1);
    }
    LOG(INFO) << "    Test net output " << output_name << ": per_pixel_acc = " << per_pixel_acc;
    LOG(INFO) << "    Test net output " << output_name << ": per_label_acc = " << per_label_acc / num_valid_labels;
    LOG(INFO) << "    Test net output " << output_name << ": iou_acc = " << iou_acc / num_valid_labels;
    LOG(INFO) << "    Test net output " << output_name << ": weighted_iou_acc = " << weighted_iou_acc;
  } 
  else 
    LOG(INFO) << "    Test net output " << output_name << ": no valid labels!";  
    
    
  Caffe::set_bn_state(param_.bn_state());
	Caffe::set_drop_state(param_.drop_state());  
	Caffe::set_reuse(false);
}

///------------------------------------------------ proto <->  memory------------------------------------------------------

void SolverCNN::Snapshot() 
{
//------------------------------
	NetParameter net_param;
  net_->ToProto(&net_param);
  
  string model_filename = param_.snapshot_prefix() + "_iter_" +format_int(this->iter_) + ".caffemodel";
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
	WriteProtoToBinaryFile(net_param, model_filename);
//-----------------------------

//-----------------------------	
	SolverState state;
	state.set_iter(this->iter_);
	state.set_learned_net(model_filename);
	
	//net_->StateToProto(state.mutable_net_state());
	string snapshot_filename = param_.snapshot_prefix() + "_iter_"+ format_int(this->iter_)+ ".solverstate";
	LOG(INFO) << "Snapshotting to binary proto file " << snapshot_filename;
	WriteProtoToBinaryFile(state, snapshot_filename.c_str());
//-----------------------------
}


void SolverCNN::Restore(const char* state_file) 
{
	SolverState state;
	ReadProtoFromBinaryFile(state_file, &state);


	this->iter_ = state.iter();

	NetParameter net_param;
  ReadProtoFromBinaryFile(state.learned_net().c_str(), &net_param);
  net_->CopyTrainedLayersFrom(net_param);
	
  //net_->RestoreState(state.net_state());
}



}  // namespace caffe
