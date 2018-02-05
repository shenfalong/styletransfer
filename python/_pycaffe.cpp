#include <sstream>
#include <string>
#include <vector>


#include "caffe/caffe.hpp"
#include<iostream>
#include <boost/python.hpp>


#include <numpy/ndarrayobject.h>
using namespace caffe;

static vector<shared_ptr<Net > > nets_;
static int init_key = -2;

void entry_base_model(string whole_model_name, string layer_name_file) 
{
	NetParameter net_param;
	NetParameter base_net_param;
	ReadProtoFromBinaryFile(whole_model_name.c_str(), &net_param);

	
	vector<string> layer_name_list;
  layer_name_list.clear();
  char layer_name[100];
 	FILE * fp = fopen(layer_name_file.c_str(),"rt");
 	if (fp == NULL)
 		throw std::runtime_error("base.txt is lost");
 	while(!feof(fp))
 	{
 		fscanf(fp,"%s",layer_name);
 		if (!feof(fp))
 			layer_name_list.push_back(layer_name);
 	}
 	fclose(fp);
 	
 	base_net_param.clear_layer();
 	for (int i = 0; i < layer_name_list.size(); ++i)
  {
  	int j=0;
  	while(layer_name_list[i].compare(net_param.layer(j).name()))
  	{ j++; if (j >= net_param.layer_size())  throw std::runtime_error("layer name not match");}
  	std::cout<<"dumping layer "<<net_param.layer(j).name()<<'\n';
  	base_net_param.add_layer()->CopyFrom(net_param.layer(j));
 	}
 	
 	
 	WriteProtoToBinaryFile(base_net_param, "base.caffemodel");
}

void entry_save_model(PyObject* weights_raw, string layer_name_file, string base_model_name, string predict_model_name) 
{
	PyListObject* weights_list = reinterpret_cast<PyListObject*>(weights_raw);
	

	NetParameter net_param;
	ReadProtoFromBinaryFile(base_model_name.c_str(), &net_param);
  
  vector<string> layer_name_list;
  layer_name_list.clear();
  char layer_name[100];
 	FILE * fp = fopen(layer_name_file.c_str(),"rt");
 	if (fp == NULL)
 		throw std::runtime_error("layer name file is lost");
 	while(!feof(fp))
 	{
 		fscanf(fp,"%s",layer_name);
 		if (!feof(fp))
 			layer_name_list.push_back(layer_name);
 	}
 	fclose(fp);
 	
 	
 	
 	if (layer_name_list.size() != weights_list->ob_size)
 	{
 		std::cout<<layer_name_list.size()<<", "<<weights_list->ob_size<<'\n';
 		throw std::runtime_error("layer name and blob size not match");
	}
  for (int i = 0; i < weights_list->ob_size; ++i)
  {
  	std::cout<<"dumping layer "<<layer_name_list[i]<<'\n';
  	
  	LayerParameter* layer_param = net_param.add_layer();
    layer_param->set_name(layer_name_list[i]);
    
  	PyArrayObject* weights = reinterpret_cast<PyArrayObject*>(weights_list->ob_item[i]);
  	int num = PyArray_DIMS(weights)[0];
		int channels = PyArray_DIMS(weights)[1];
		int height = PyArray_DIMS(weights)[2];
		int width = PyArray_DIMS(weights)[3];
  	Blob temp_blob;
  	temp_blob.Reshape(num,channels,height,width);
  	
  	const float* const data_ptr = reinterpret_cast<const float* const >(PyArray_DATA(weights));
    caffe_copy(temp_blob.count(), data_ptr, temp_blob.mutable_gpu_data());
    temp_blob.ToProto(layer_param->add_blobs(), false);
  }

	WriteProtoToBinaryFile(net_param, predict_model_name.c_str());
}
void entry_reset() 
{
	shared_ptr<Net > net_; 
	for(int i=0;i<nets_.size();i++)
	{
		net_ = nets_[i];
		net_.reset();		  
		LOG(INFO) << "Network reset, call init before use it again";		
	}
	nets_.clear();
	init_key = -2;
}
int entry_get_init_key() 
{
	return init_key;
}
void entry_set_device(int device_id) 
{
	Caffe::GPUs[0]=device_id;
	Caffe::SetDevice(device_id);
}

int entry_is_initialized() 
{
	return nets_.size();
}
PyObject* entry_get_weights(int hnet) 
{
	if (hnet >= nets_.size()) 
   	throw std::runtime_error("too large net index");
	shared_ptr<Net > net_ = nets_[hnet];

	
	
  const vector<shared_ptr<Layer > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  // Step 1: count the number of layers with weights
  int num_layers = 0;
  for (unsigned int i = 0; i < layers.size(); ++i) 
  {
    vector<shared_ptr<Blob > >& layer_blobs = layers[i]->blobs();
    if (layer_blobs.size() == 0) 
      continue;       
    num_layers++;
  }

  // Step 2: prepare output array of structures 
	PyListObject* py_layers_name_ = reinterpret_cast<PyListObject*>(PyList_New(num_layers));
	PyListObject* py_layers_weights_ = reinterpret_cast<PyListObject*>(PyList_New(num_layers));
	
  int py_layer_index = 0;
  for (unsigned int i = 0; i < layers.size(); ++i) 
  {
    vector<shared_ptr<Blob > >& layer_blobs = layers[i]->blobs();
    if (layer_blobs.size() == 0) 
      continue;
    
		PyListObject* blobs_list = reinterpret_cast<PyListObject*>(PyList_New(layer_blobs.size()));
    for (unsigned int j = 0; j < layer_blobs.size(); ++j) 
    {
      npy_intp dims[4] = {layer_blobs[j]->num(),layer_blobs[j]->channels(),layer_blobs[j]->height(),layer_blobs[j]->width()};
			blobs_list->ob_item[j] = PyArray_SimpleNew(4, dims, NPY_FLOAT32);      
			
			std::cout<<"layer weights size = "<<layer_blobs[j]->num()<<", "<<layer_blobs[j]->channels()<<", "<<layer_blobs[j]->height()<<'\n';
			
      float* weights_ptr = reinterpret_cast<float*>(PyArray_DATA(blobs_list->ob_item[j]));		
     	caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_data(),weights_ptr);   
    }   
		
		
		py_layers_name_->ob_item[py_layer_index] = PyString_FromString(layer_names[i].c_str());
		py_layers_weights_->ob_item[py_layer_index] = reinterpret_cast<PyObject*>(blobs_list);
		py_layer_index++;
  }

	
	PyObject* py_layers_ = PyList_New(2);
	(reinterpret_cast<PyListObject*>(py_layers_))->ob_item[0] = reinterpret_cast<PyObject*>(py_layers_name_);
	(reinterpret_cast<PyListObject*>(py_layers_))->ob_item[1] = reinterpret_cast<PyObject*>(py_layers_weights_);
  return py_layers_;

}
void entry_init(char*model_file,char * weight_file) 
{
	shared_ptr<Net > net_;
	NetParameter net_param;
  ReadProtoFromTextFile(string(model_file), &net_param);
  vector<shared_ptr<Blob > > net_input_blobs;
  net_input_blobs.clear();
  vector<string> net_input_blob_names;
  net_input_blob_names.clear();
  for (int i = 0; i < net_param.input_blob_size(); ++i)
  {
    const string& blob_name = net_param.input_blob(i).name();
    net_input_blob_names.push_back(blob_name);
    
    int num = net_param.input_blob(i).num();
    int channels = net_param.input_blob(i).channels();
    int height = net_param.input_blob(i).height();
    int width = net_param.input_blob(i).width();
    shared_ptr<Blob > blob_pointer(new Blob(num, channels, height, width));
    net_input_blobs.push_back(blob_pointer);
  }
  
  Caffe::set_bn_state("frozen");
	Caffe::set_drop_state("fixed");
	Caffe::set_reuse(true);
	
  net_.reset(new Net(net_param,net_input_blobs,net_input_blob_names));
  
  
  net_param.Clear();
  ReadProtoFromBinaryFile(weight_file, &net_param);
  net_->CopyTrainedLayersFrom(net_param);
  
	nets_.push_back(net_);


  init_key = nets_.size();  // NOLINT(caffe/random_fn)
}
PyObject* entry_forward(int hNet, PyObject* bottom_raw) 
{
	PyListObject* bottom = reinterpret_cast<PyListObject*>(bottom_raw);
	shared_ptr<Net > net_ = nets_[hNet];
	
  const vector<shared_ptr<Blob > >& input_blobs = net_->input_blobs();
  if (bottom->ob_size != input_blobs.size()) 
   	throw std::runtime_error("Invalid input blob numbers");

  for (int i = 0; i < input_blobs.size(); i++) 
  {
  	PyArrayObject* elem = reinterpret_cast<PyArrayObject*>(bottom->ob_item[i]);
    if (PyArray_TYPE(elem) != NPY_FLOAT32) 
    	throw std::runtime_error(" must be float32");
		if (PyArray_NDIM(elem) != 4) 
			throw std::runtime_error(" ndims must be == 4");
		
		//image is saved as height-width-channels in opencv,
		//permute is as channels-height-width before feeding it
		int num = PyArray_DIMS(elem)[0];
		int channels = PyArray_DIMS(elem)[1];
		int height = PyArray_DIMS(elem)[2];
		int width = PyArray_DIMS(elem)[3];
		
	
    input_blobs[i]->Reshape(num,channels,height,width);
		
		std::cout<<"num = "<<num<<", channels = "<<channels<<", height = "<<height<<", width = "<<width<<'\n';
		
    const float* const data_ptr = reinterpret_cast<const float* const>(PyArray_DATA(elem));
    caffe_copy(input_blobs[i]->count(), data_ptr, input_blobs[i]->mutable_gpu_data());

  }
	
  net_->Forward();
 	const vector<shared_ptr<Blob > >& output_blobs = net_->output_blobs();
  

  PyObject* top = PyList_New(output_blobs.size());
	PyListObject* top_list = reinterpret_cast<PyListObject*>(top);
 
  for (int i = 0; i < output_blobs.size(); i++) 
  {
    npy_intp dims[4] = {output_blobs[i]->num(),output_blobs[i]->channels(),output_blobs[i]->height(),output_blobs[i]->width()};
    top_list->ob_item[i] = PyArray_SimpleNew(4, dims, NPY_FLOAT32);

		std::cout<<"output_dim = "<<PyArray_DIMS(top_list->ob_item[i])[0]<<", "<< PyArray_DIMS(top_list->ob_item[i])[1]
		<<", "<< PyArray_DIMS(top_list->ob_item[i])[2]<<", "<< PyArray_DIMS(top_list->ob_item[i])[3]<<'\n';
		
    float* data_ptr = reinterpret_cast<float*>(PyArray_DATA(top_list->ob_item[i]));

    caffe_copy(output_blobs[i]->count(), output_blobs[i]->gpu_data(), data_ptr);
  }

  return top;

}
BOOST_PYTHON_MODULE(_pycaffe)
{
  import_array();
  Caffe::GPUs.resize(1);
  Caffe::GPUs[0]=0;
  nets_.clear();
  
  boost::python::def("reset", entry_reset);
  boost::python::def("get_weights", entry_get_weights);
  boost::python::def("forward", entry_forward);
  boost::python::def("init", entry_init);
  boost::python::def("get_init_key", entry_get_init_key);
  boost::python::def("set_device", entry_set_device);
  boost::python::def("is_initialized", entry_is_initialized);
  boost::python::def("save_model", entry_save_model);
  boost::python::def("base_model", entry_base_model);
}

