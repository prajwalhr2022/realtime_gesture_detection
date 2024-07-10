import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

print(tf.__version__)
print(tf.keras.__version__)

# Define the path to the directory containing the .npy files
data_path = 'C:/Users/Admin/PycharmProjects/pythonProject/final project'  # Replace with the actual dataset directory name on Kaggle

X_test_path = os.path.join(data_path, 'X_test.npy')
y_test_path = os.path.join(data_path, 'y_test.npy')

# Load the .npy files
X_test = np.load(X_test_path)
y_test = np.load(y_test_path)

actions = np.array(['hello', 'thanks', 'iloveyou'])

model = load_model('action.h5')

res = model.predict(X_test)
print(actions[np.argmax(res[0])])
