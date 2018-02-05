#include <vector>

#include "caffe/layers/func/label_sperate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



void LabelSperateLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
		int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  
	for (int n=0;n<num;n++)
	{
		
//-----gender
		top[0]->mutable_cpu_data()[n] = bottom[0]->cpu_data()[n*channels+0];
//-----hairlength
		top[1]->mutable_cpu_data()[n] = bottom[0]->cpu_data()[n*channels+1];
		
//-----head		
		for (int c=0;c<4;c++)
			top[10]->mutable_cpu_data()[n*28+c]  = bottom[0]->cpu_data()[n*channels+2+c];		
//-----top
		top[2]->mutable_cpu_data()[n] = bottom[0]->cpu_data()[n*channels+6];
		top[3]->mutable_cpu_data()[n] = bottom[0]->cpu_data()[n*channels+7];
		for (int c=0;c<4;c++)
			top[10]->mutable_cpu_data()[n*28+4+c]  = bottom[0]->cpu_data()[n*channels+8+c];		
//------down			
		top[4]->mutable_cpu_data()[n] = bottom[0]->cpu_data()[n*channels+12];
		top[5]->mutable_cpu_data()[n] = bottom[0]->cpu_data()[n*channels+13];
		for (int c=0;c<4;c++)
			top[10]->mutable_cpu_data()[n*28+8+c]  = bottom[0]->cpu_data()[n*channels+14+c];		
//------shoes		
		top[6]->mutable_cpu_data()[n] = bottom[0]->cpu_data()[n*channels+18];
		top[7]->mutable_cpu_data()[n] = bottom[0]->cpu_data()[n*channels+19];
		for (int c=0;c<8;c++)
			top[10]->mutable_cpu_data()[n*28+12+c]  = bottom[0]->cpu_data()[n*channels+20+c];		
//------hat
		for (int c=0;c<4;c++)
			top[10]->mutable_cpu_data()[n*28+20+c]  = bottom[0]->cpu_data()[n*channels+28+c];		
//------bag		
		top[8]->mutable_cpu_data()[n] = bottom[0]->cpu_data()[n*channels+32];
		top[9]->mutable_cpu_data()[n] = bottom[0]->cpu_data()[n*channels+33];
		for (int c=0;c<4;c++)
			top[10]->mutable_cpu_data()[n*28+24+c]  = bottom[0]->cpu_data()[n*channels+34+c];		
//------------------localization 
		
	}
	for(int i=0;i<top[10]->count();i++)
	{
		if (top[10]->cpu_data()[i] > 10)
		{
			LOG(INFO)<<"bad sample detected";
			top[10]->mutable_cpu_data()[i] = 10;
		}	
		if (top[10]->cpu_data()[i] < 0 && top[10]->cpu_data()[i] != -1)
		{
			LOG(INFO)<<"bad sample detected";
			top[10]->mutable_cpu_data()[i] = 0;
		}
	}
	//LOG(INFO)<<bottom[0]->cpu_data()[0];
	//LOG(INFO)<<top[0]->cpu_data()[0];
#if 0
	for(int i=0;i<bottom[0]->count();i++)
	{
		if (bottom[0]->cpu_data()[i] == 255)
		{
			top[0]->mutable_cpu_data()[i] = 255;
			top[1]->mutable_cpu_data()[i] = 255;
		}
		else
		{
			top[0]->mutable_cpu_data()[i] = stuff_mapping[int(bottom[0]->cpu_data()[i])];
			top[1]->mutable_cpu_data()[i] = object_mapping[int(bottom[0]->cpu_data()[i])];
		}	
	}
#endif
}


void LabelSperateLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	
}

void LabelSperateLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}

}  // namespace caffe
		
