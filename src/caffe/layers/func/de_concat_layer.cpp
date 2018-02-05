
#include <vector>

#include "caffe/layers/func/de_concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void DeConcatLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{

}


void DeConcatLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	const int num = bottom[0]->num();
	const int channels = bottom[0]->channels();
	const int height = bottom[0]->height();
	const int width = bottom[0]->width();
 	
 	CHECK_EQ(this->layer_param_.concat_param().channels_size(),top.size());
 	
 	int all_channels = 0;
 	for (int i=0;i<top.size();i++)
 	{
 		int i_channels = this->layer_param_.concat_param().channels(i);
	 	top[i]->Reshape(num,i_channels,height,width);
	 	all_channels += i_channels;
	}
	
	CHECK_EQ(channels,all_channels);

}



REGISTER_LAYER_CLASS(DeConcat);
}  // namespace caffe
