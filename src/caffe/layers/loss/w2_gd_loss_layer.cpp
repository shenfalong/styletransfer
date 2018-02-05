
#include <vector>

#include "caffe/layers/loss/w2_gd_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void W2GdLossLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
}


void W2GdLossLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	CHECK_EQ(bottom.size(),2);
	
	if (Caffe::gan_type() == "train_dnet")
	{
		//CHECK_EQ(bottom[0]->num(),2*bottom[1]->num());
		//we do not set gan_type during the set-up stage
		
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
	
		CHECK_EQ(bottom.size(),2);
		CHECK_EQ(num%2,0);

	
		count_.Reshape(num,channels,height,width);
		mask_.Reshape(num,channels,height,width);
		loss_g_.Reshape(num/2,1,height,width);
		loss_d_.Reshape(num/2,1,height,width);
		loss_c_.Reshape(num,1,height,width);
		prob_.Reshape(num,channels,height,width);

		top[0]->Reshape(1,1,1,1);
	}
	else
	{
		//CHECK_EQ(bottom[0]->num(),bottom[1]->num());
		
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
	
	
	
		loss_g_.Reshape(num,1,height,width);
		loss_c_.Reshape(num,1,height,width);
		prob_.Reshape(num,channels,height,width);
		top[0]->Reshape(1,1,1,1);
	}
}



REGISTER_LAYER_CLASS(W2GdLoss);
}  // namespace caffe
