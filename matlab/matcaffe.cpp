//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.

#include <sstream>
#include <string>
#include <vector>

#include "mex.h"

#include "caffe/caffe.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs


// Log and throw a Mex error
inline void mex_error(const std::string &msg) {
  LOG(ERROR) << msg;
  mexErrMsgTxt(msg.c_str());
}

using namespace caffe;  // NOLINT(build/namespaces)

// The pointer to the internal caffe::Net instance
static vector<shared_ptr<Net<float> > > nets_;
static int init_key = -2;



static mxArray* do_forward(const mxArray* const hNet,const mxArray* const bottom) 
{
	shared_ptr<Net<float> > net_ = nets_[static_cast<int>(mxGetScalar(hNet))];
	
  const vector<Blob<float>*>& input_blobs = net_->input_blobs();
  if (static_cast<unsigned int>(mxGetDimensions(bottom)[0]) != input_blobs.size()) 
    mex_error("Invalid input blob numbers");

  for (int i = 0; i < input_blobs.size(); i++) 
  {
  	
    const mxArray* const elem = mxGetCell(bottom, i);
    if (!mxIsSingle(elem)) 
      mex_error("MatCaffe require single-precision float point data");

		// Reshape input to match data if needed
    const mwSize* dims = mxGetDimensions(elem);
    const mwSize num_dims = mxGetNumberOfDimensions(elem);

    // In matlab, you cannot have trailing singleton dimensions
    // So an input batch of size 1 (W x H x C x 1) will come in with 3 dims
    // like (W x H x C).
    int num,channels,height,width;
    switch (num_dims)
    {
    case 1:
    	width = dims[0];
    	height = 1;
    	channels = 1;
    	num = 1;
    	break;
    case 2:
    	width = dims[0];
			height = dims[1]; 	
    	channels = 1;
    	num = 1;   	
    	break;
   	case 3:
   		width = dims[0];
			height = dims[1]; 	
    	channels = dims[2];
    	num = 1;   	
    	break;
    case 4:
    	width = dims[0];
			height = dims[1]; 	
    	channels = dims[2];
    	num = dims[3];   	
    	break;	
 		default: 
 			LOG(FATAL)<<"invalid blob size";
    }
      
 
    input_blobs[i]->Reshape(num,channels,height,width);

		mexPrintf("num = %d, channels = %d, height = %d, width = %d\n",input_blobs[i]->num(),input_blobs[i]->channels(),input_blobs[i]->height(),input_blobs[i]->width());
    const float* const data_ptr = reinterpret_cast<const float* const>(mxGetPr(elem));
    caffe_copy(input_blobs[i]->count(), data_ptr, input_blobs[i]->mutable_gpu_data());
  }

  
  net_->Forward();
 	const vector<Blob<float>*>& output_blobs = net_->output_blobs();
  
  
  mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1);
  for (int i = 0; i < output_blobs.size(); i++) 
  {
    mwSize dims[4] = {output_blobs[i]->width(), output_blobs[i]->height(), output_blobs[i]->channels(), output_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    caffe_copy(output_blobs[i]->count(), output_blobs[i]->gpu_data(), data_ptr);
  }

  return mx_out;
}



static mxArray* do_get_weights(const mxArray* const hNet) 
{

	shared_ptr<Net<float> > net_ = nets_[static_cast<int>(mxGetScalar(hNet))];
	
	
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  // Step 1: count the number of layers with weights
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) 
    {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) 
        continue;
      
      if (layer_names[i] != prev_layer_name) 
      {
        prev_layer_name = layer_names[i];
        num_layers++;
      }
    }
  }

  // Step 2: prepare output array of structures
  mxArray* mx_layers;
  {
    const mwSize dims[2] = {num_layers, 1};
    const char* fnames[2] = {"weights", "layer_names"};
    mx_layers = mxCreateStructArray(2, dims, 2, fnames);
  }

  // Step 3: copy weights into output
  
  string prev_layer_name = "";
  int mx_layer_index = 0;
  for (unsigned int i = 0; i < layers.size(); ++i) 
  {
    vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
    if (layer_blobs.size() == 0) 
      continue;
    

    mxArray* mx_layer_cells = NULL;
    if (layer_names[i] != prev_layer_name) 
    {
      prev_layer_name = layer_names[i];
      const mwSize dims[2] = {static_cast<mwSize>(layer_blobs.size()), 1};
      mx_layer_cells = mxCreateCellArray(2, dims);
      mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells);
      mxSetField(mx_layers, mx_layer_index, "layer_names", mxCreateString(layer_names[i].c_str()));
      mx_layer_index++;
    }

    for (unsigned int j = 0; j < layer_blobs.size(); ++j) 
    {
      // internally data is stored as (width, height, channels, num)
      // where width is the fastest dimension
      mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(), layer_blobs[j]->channels(), layer_blobs[j]->num()};

      mxArray* mx_weights = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
      mxSetCell(mx_layer_cells, j, mx_weights);
      float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

      caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_data(),weights_ptr);        
    }
  }
   
  return mx_layers;
}

static void get_weights(MEX_ARGS) 
{
  plhs[0] = do_get_weights(prhs[0]);
}


static void set_device(MEX_ARGS) 
{
  if (nrhs != 1) 
  {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

static void get_init_key(MEX_ARGS) 
{
  plhs[0] = mxCreateDoubleScalar(init_key);
}

static void init(MEX_ARGS) 
{
  if (nrhs != 2) 
  {
    ostringstream error_msg;
    error_msg << "Expected 2 arguments, got " << nrhs;
    mex_error(error_msg.str());
  }

  char* model_file = mxArrayToString(prhs[0]);
  char* weight_file = mxArrayToString(prhs[1]);


	shared_ptr<Net<float> > net_;
	NetParameter net_param;
  ReadProtoFromTextFile(string(model_file), &net_param);
  vector<shared_ptr<Blob<float> > > net_input_blobs;
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
    shared_ptr<Blob<float> > blob_pointer(new Blob<float>(num, channels, height, width));
    net_input_blobs.push_back(blob_pointer);
  }
  
  Caffe::set_bn_state("frozen");
	Caffe::set_drop_state("fixed");
	Caffe::set_reuse(true);
	
	
  net_.reset(new Net<float>(net_param,net_input_blobs,net_input_blob_names));
  
  
  net_param.Clear();
  ReadProtoFromBinaryFile(weight_file, &net_param);
  net_->CopyTrainedLayersFrom(net_param);
  
	nets_.push_back(net_);


  mxFree(model_file);
  mxFree(weight_file);

  init_key = nets_.size();  // NOLINT(caffe/random_fn)

  if (nlhs == 1) 
    plhs[0] = mxCreateDoubleScalar(init_key);
  
}

static void reset(MEX_ARGS) 
{
	shared_ptr<Net<float> > net_; 
	for(int i=0;i<nets_.size();i++)
	{
		net_ = nets_[i];
		net_.reset();		  
		LOG(INFO) << "Network reset, call init before use it again";		
	}
	nets_.clear();
	init_key = -2;
}

static void forward(MEX_ARGS) 
{
  if (nrhs != 2) 
  {
    ostringstream error_msg;
    error_msg << "Expected 2 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  plhs[0] = do_forward(prhs[0],prhs[1]);
}


static void is_initialized(MEX_ARGS) 
{
    plhs[0] = mxCreateDoubleScalar(nets_.size());
}


/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry 
{
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = 
{
  // Public API functions
  { "forward",            forward         },
  { "init",               init            },
  { "is_initialized",     is_initialized  },
  { "set_device",         set_device      },
  { "get_weights",        get_weights     },
  { "get_init_key",       get_init_key    },
  { "reset",              reset           },
  // The end.
  { "END",                NULL            },
};

/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) 
{
  mexLock();  // Avoid clearing the mex file.
  
  Caffe::GPUs.resize(1);
  Caffe::GPUs[0]=0;
  
  if (nrhs == 0) 
  {
    mex_error("No API command given");
    return;
  }

 // Handle input command
  char *cmd = mxArrayToString(prhs[0]);
  bool dispatched = false;
  // Dispatch to cmd handler
  for (int i = 0; handlers[i].func != NULL; i++) 
  {
    if (handlers[i].cmd.compare(cmd) == 0) 
    {
      handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
      dispatched = true;
      break;
    }
  }
  if (!dispatched) 
  {
    ostringstream error_msg;
    error_msg << "Unknown command '" << cmd << "'";
    mex_error(error_msg.str());
  }
  mxFree(cmd);

}
