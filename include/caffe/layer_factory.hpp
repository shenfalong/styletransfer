#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
using namespace std;
namespace caffe {


class Layer;

class LayerRegistry 
{
 public:
  //define a function pointer
  typedef shared_ptr<Layer > (*Creator)(const LayerParameter&);
  //From string  ----- to -->   function
  typedef std::map<string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() 
  {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
  static void AddCreator(const string& type, Creator creator) 
  {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0) << "Layer type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a layer using a LayerParameter.
  static shared_ptr<Layer > CreateLayer(const LayerParameter& param) 
  {
    LOG(INFO) << "Creating layer " << param.name();
    string type = param.type();        	
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type << " (known types: " << LayerTypeListString() << ")";
    return registry[type](param);
  }

  static vector<string> LayerTypeList() 
  {
    CreatorRegistry& registry = Registry();
    vector<string> layer_types;
    for (typename CreatorRegistry::iterator iter = registry.begin(); iter != registry.end(); ++iter) 
      layer_types.push_back(iter->first);
    
    return layer_types;
  }

 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.
  LayerRegistry() {}

  static string LayerTypeListString() 
  {
    vector<string> layer_types = LayerTypeList();
    string layer_types_str;
    for (vector<string>::iterator iter = layer_types.begin(); iter != layer_types.end(); ++iter) 
    {
      if (iter != layer_types.begin()) 
        layer_types_str += ", ";
      
      layer_types_str += *iter;
    }
    return layer_types_str;
  }
};


class LayerRegisterer 
{
 public:
  LayerRegisterer(const string& type, shared_ptr<Layer > (*creator)(const LayerParameter&)) 
  {
    LayerRegistry::AddCreator(type, creator);
  }
};

#define REGISTER_LAYER_CLASS(type)                                             \
	shared_ptr<Layer> Creator_##type##Layer(const LayerParameter& param)         \
  {                                                                            \
    return shared_ptr<Layer>(new type##Layer(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)


#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer g_creator_f_##type(#type, creator);     

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
