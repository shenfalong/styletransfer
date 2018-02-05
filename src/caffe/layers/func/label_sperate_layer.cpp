#include <vector>

#include "caffe/layers/func/label_sperate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void LabelSperateLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
#if 0
	FILE*  fid = fopen("stuff36.txt","rb");
	if (fid == NULL)
		LOG(FATAL)<<"stuff file not found";
	for(int i=0;i<150;i++)
		fscanf(fid,"%d",&stuff_mapping[i]);
	fclose(fid);	
	fid = fopen("object115.txt","rb");
	if (fid == NULL)
		LOG(FATAL)<<"object file not found";
	for(int i=0;i<150;i++)
		fscanf(fid,"%d",&object_mapping[i]);	
	fclose(fid);
#endif
}


void LabelSperateLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
//------------------classification
	top[0]->Reshape(num,1,1,1);
	top[1]->Reshape(num,1,1,1);
	top[2]->Reshape(num,1,1,1);
	top[3]->Reshape(num,1,1,1);
	top[4]->Reshape(num,1,1,1);
	top[5]->Reshape(num,1,1,1);
	top[6]->Reshape(num,1,1,1);
	top[7]->Reshape(num,1,1,1);
	top[8]->Reshape(num,1,1,1);
	top[9]->Reshape(num,1,1,1);
//------------------localization 
	top[10]->Reshape(num,28,1,1);

	
#if 0
	top[0]->ReshapeLike(*bottom[0]);
	top[1]->ReshapeLike(*bottom[0]);
#endif
}




REGISTER_LAYER_CLASS(LabelSperate);
}  // namespace caffe
		
