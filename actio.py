import h5py
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

# Path to the model file
model_path = 'C:/Users/Admin/PycharmProjects/pythonProject/final project/action.h5'

# Read the model file
with h5py.File(model_path, 'r+') as f:
    # Load model configuration
    model_config = json.loads(f.attrs['model_config'])

    # Iterate through the layers to find the InputLayer
    for layer in model_config['config']['layers']:
        if layer['class_name'] == 'InputLayer' and 'batch_shape' in layer['config']:
            # Replace batch_shape with batch_input_shape
            layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')

    # Save the modified configuration back to the file
    f.attrs['model_config'] = json.dumps(model_config)

# Load the modified model
model = load_model(model_path)

# Now you can proceed with your further code
print("Model loaded successfully!")
