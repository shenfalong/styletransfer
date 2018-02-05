#include <cstdio>
#include <stdio.h>
#include <string>
#include <vector>
#include "caffe/util/format.hpp"
#include "caffe/solver.hpp"

#include "caffe/util/io.hpp"


namespace caffe {


int Solver::iter_;
bool Solver::change_style_;

void Solver::share_weight(const shared_ptr<Net> net_, const shared_ptr<Net > test_net_)
{
	for (int i = 0; i < test_net_->layers().size(); i++)
	{
		if (test_net_->layers()[i]->blobs().size() > 0)
		{
			int i_train = -1;
			for (int j=0;j < net_->layers().size();j++)
			{
				if (test_net_->layers()[i]->layer_param().name() == net_->layers()[j]->layer_param().name())
				{
					i_train = j;
					break;   
				} 		
			}			
			if (i_train != -1)
			{
				for (int j = 0;j<test_net_->layers()[i]->blobs().size();j++)
			 	{
			 		//test_net_->layers()[i]->blobs()[j] = net_->layers()[i_train]->blobs()[j];	 		
			 		//Due to the 'parallel layer', layer->blob points to unary->blob, layer->blob is not directly applied on 
			 		//layer computation. Therefore we can not change the pointer ! ! !
			 		test_net_->layers()[i]->blobs()[j]->set_data(*net_->layers()[i_train]->blobs()[j]);
			 		test_net_->layers()[i]->blobs()[j]->set_diff(*net_->layers()[i_train]->blobs()[j]);
			 	}
				for (int j = 0;j<test_net_->layers()[i]->parallel_blobs().size();j++)
			 	{
			 		//test_net_->layers()[i]->parallel_blobs()[j] = net_->layers()[i_train]->parallel_blobs()[j];
			 		test_net_->layers()[i]->parallel_blobs()[j]->set_data(*net_->layers()[i_train]->parallel_blobs()[j]);
			 		test_net_->layers()[i]->parallel_blobs()[j]->set_diff(*net_->layers()[i_train]->parallel_blobs()[j]);
			 	}
			 }
			 else
			 	LOG(FATAL)<<"did not find weight in the target networks "<<test_net_->layers()[i]->layer_param().name();
		}
	}
}

}  // namespace caffe
