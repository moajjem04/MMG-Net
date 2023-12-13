# MMG-Net: Multi Modal Approach to Estimate Blood Glucose using Multi-Stream and Cross Modality Attention

This is the github repository containing the code for MMG-Net model architecture. It has been implemented using Tensorflow and Keras.

The paper has been submitted to [Biomedical Signal Processing and Control](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control).

## User Instructions

MMG-Net model can be created using the following code snippet:

```python
from models import MMG_Net

# Model Arguments
MODEL_PARAMS["length"] = 2048  # length of the signal
MODEL_PARAMS["features"] = 24  # Number of food features
MODEL_PARAMS["model_width"] = 128  # Number of Filter or Kernels in the Input Layer
MODEL_PARAMS["num_channel"] = 3  # Number of Input Channels
MODEL_PARAMS["num_stack"] = 3  # same as Number of Input Channels
MODEL_PARAMS["output_number"] = 1 # Number of Vital Health Metrics to estimate. In this case 1.
MODEL_PARAMS["kernel"] = 7 # kernel size
MODEL_PARAMS["depth"] = 8 # number of ConvMixer blocks
MODEL_PARAMS["patch"] = 10 # patch size for projection block
MODEL_PARAMS["CMA"] = True # Enable Cross Modality Attention

# Get the model
net = MMG_Net(**MODEL_PARAMS)

# Then train as a normal keras model
# Shape of X_train = (n_samples, MODEL_PARAMS["length"] + MODEL_PARAMS["features"], MODEL_PARAMS["num_channel"])
# Shape of Y_train = (n_samples, 1)
net.fit(X_train, Y_train)
```
