import json
import numpy as np

network_layer_data = {"layer_dimensions": list(ISONet.layer_dimensions), "layer_parameters":{}}

for layer_name in ISONet.get_weight_paths().keys():
    
    layer = ISONet.get_weight_paths()[layer_name]
    
    layer_type = layer_name.split(".")[1]
        
    layer_data = layer.numpy().flatten(order="C").tolist()
    
    if (layer_type == "kernel"):
        layer_dims = list(layer.shape)
    else:
        layer_dims = list(layer.shape + (1,))
    ##
    
    network_layer_data["layer_parameters"][layer_name] = {"data": layer_data, "dims": layer_dims}
    
# with open("/home/rms221/Documents/Compressive_Neural_Representations_Tensorflow/AuxFiles/outputs/nodule_json/network_layer_data.json", 'w') as file: json.dump(network_layer_data, file, indent=4) 
    
ISONet(np.array([[1.0, 1.0, 1.0]]))


