import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import h5py

# Load the original model
original_model = load_model('C:/Users/Admin/PycharmProjects/pythonProject/final project/action.h5')

# Save the model configuration to a JSON file
model_json = original_model.to_json()
with open('model_config.json', 'w') as json_file:
    json_file.write(model_json)

# Save the weights
original_model.save_weights('model_weights.h5')

# Load the model configuration from the JSON file
with open('model_config.json', 'r') as json_file:
    config = json.load(json_file)

# Adjust the config dictionary
for layer in config['config']['layers']:
    if layer['class_name'] == 'InputLayer' and 'batch_shape' in layer['config']:
        layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')

# Save the adjusted config back to a JSON file
with open('adjusted_model_config.json', 'w') as json_file:
    json.dump(config, json_file)

# Load the adjusted model configuration
with open('adjusted_model_config.json', 'r') as json_file:
    adjusted_config = json.load(json_file)

# Reconstruct the model from the adjusted configuration
model = tf.keras.models.model_from_json(json.dumps(adjusted_config))

# Load the weights
model.load_weights('model_weights.h5')

# Save the final model
model.save('adjusted_action.h5')

# Now you can load the adjusted model without issues
final_model = load_model('adjusted_action.h5')
