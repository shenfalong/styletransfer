// Copyright Yangqing Jia 2013

#include <set>
#include <string>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/layer_factory.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/format.hpp"
#include "caffe/layers/func/parallel_layer.hpp"
#include <boost/thread.hpp>
using std::pair;
using std::map;
using std::set;

namespace caffe {



void Net::BcastData()
{	
	if (NGPUS > 1)
  {
		for (int i = 0; i < layers_.size(); i++)
		{	
			//boost::mutex::scoped_lock lock(mu);  
			//i_layer = i;
		 // if (i_layer > j_layer)
			//	cond.notify_one();
			for (int j = 0;j < layers_[i]->blobs().size();j++)
			{			
				for (int k = 0; k < NGPUS; k++) 
				{
					CUDA_CHECK(cudaSetDevice(Caffe::GPUs[k]));
					ncclBcast((void *)layers_[i]->parallel_blobs()[j*NGPUS+k]->mutable_gpu_data(),layers_[i]->parallel_blobs()[j*NGPUS+k]->count(),
														ncclFloat,0,Caffe::comms(k),NULL);	
				}	
			}
		}	
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
		//-----------------------------------------
	}
	//i_layer++;
	//cond.notify_one();
}

void Net::ReduceDiff()
{	
	if (NGPUS > 1)
  {	
  	//boost::mutex::scoped_lock lock(mu);  
			//i_layer = i;
			//if (i_layer <= j_layer)
			//	cond.wait(mu);			  
		for (int i = layers_.size() - 1; i >= 0; i--)
		{						
			for (int j = 0;j < layers_[i]->blobs().size();j++)
			{				
				for (int k = 0; k < NGPUS; k++) 
				{
					CUDA_CHECK(cudaSetDevice(Caffe::GPUs[k]));
					ncclReduce(layers_[i]->parallel_blobs()[j*NGPUS+k]->gpu_diff(),layers_[i]->parallel_blobs()[j*NGPUS]->mutable_gpu_diff(),
															layers_[i]->parallel_blobs()[j*NGPUS]->count(),
															ncclFloat,ncclSum,0,Caffe::comms(k),NULL);	
				}							
			}			
		}	
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
	}
}
}  // namespace caffe
