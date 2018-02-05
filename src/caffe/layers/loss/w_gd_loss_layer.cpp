
#include <vector>

#include "caffe/layers/loss/w_gd_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void WGdLossLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{

}


void WGdLossLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	if (Caffe::gan_type() == "train_dnet")
	{
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
	
		CHECK_EQ(bottom.size(),1);
		CHECK_EQ(num%2,0);
	
		count_.Reshape(num,channels,height,width);
		mask_.Reshape(num,channels,height,width);
		loss_g_.Reshape(num/2,1,height,width);
		loss_d_.Reshape(num/2,1,height,width);
	
		top[0]->Reshape(1,1,1,1);
	}
	else
	{
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		
		
		loss_g_.Reshape(num,1,height,width);
		top[0]->Reshape(1,1,1,1);
	}
}



REGISTER_LAYER_CLASS(WGdLoss);
}  // namespace caffe
