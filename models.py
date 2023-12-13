from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

from tensorflow import math, matmul,cast, float32
from keras.backend import softmax


# Implementing the Scaled-Dot Product Attention
def cross_attention_keras(queries, keys, values, d_k):

    # Scoring the queries against the keys after transposing the latter, and scaling
    scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))

    # Computing the weights by a softmax operation
    weights = softmax(scores)  # M

    # Computing the attention by a weighted sum of the value vectors
    Z = matmul(weights, values)  # Z
    return Z


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    if len(x.shape) == 2:
        x = tf.expand_dims(x, axis=2)
    x = layers.Conv1D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv1D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv1D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def convmixer_stream(
    inputs, model_width=256, depth=8, kernel=5
):
    if len(inputs.shape) == 2:
        x = tf.expand_dims(inputs, axis=2)
    else:
        x = inputs
    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, model_width, kernel)

    return x

def MMG_Net(
    length=32,
    model_width=256,
    depth=8,
    kernel=5,
    patch_size=2,
    output_number=10,
    num_stack=3,
    **kwargs,
):
    if kwargs["features"] != 0 or kwargs["features"] is not None:
        init_inputs = keras.Input((length + kwargs["features"], num_stack))
        # Get Signal and Food Features
        features = init_inputs[:, length:, :]
        features = tf.reduce_mean(features, axis=2)
        # features = layers.BatchNormalization()(features)
        print(features.shape)
        inputs = init_inputs[:, :length, :]
        # inputs = layers.BatchNormalization()(inputs)
    else:
        # Get just the signal
        init_inputs = keras.Input((length, num_stack))
        inputs = init_inputs
        # inputs = layers.BatchNormalization()(inputs)

    # Conv Stem; Multi-Stream means each channel gets its own stem
    x = [0 for _ in range(num_stack)]
    for i in range(num_stack):
        x[i] = conv_stem(inputs[:, :, i], model_width, patch_size)  # Output from stem

    # Apply Cross Modality Attention before multi-stream
    if kwargs["CMA"]:
        x1 = [0 for _ in range(num_stack)]  # a copy, output from 1st CMA
        for i in range(num_stack):
            outs = [0 for _ in range(num_stack)]
            x1[i] = x[i]
            for j in range(num_stack):
                if i == j:  # Same channel so no need. Or we can allow it too
                    continue
                a = x[i]  # Features that need to be scaled
                b = x[j]  # 2nd modality
                outs[j] = cross_attention_keras(
                    queries=a, keys=b, values=b, d_k=model_width
                )  # Save according to the 2nd modality
            # Add the attention
            for att_mask in outs:
                x1[i] = x1[i] + att_mask

        x = x1  # replacing CMA-specific variable `x1` with `x`

    x2 = []  # Output from Multi-stream
    for i in range(num_stack):
        x2.append(
            convmixer_stream(x[i], model_width, depth, kernel)
        )

    # Apply Cross Modality Attention after multi-stream
    if kwargs["CMA"]:
        x3 = x2  # a copy, output from final CMA
        for i in range(num_stack):
            outs = [0 for _ in range(num_stack)]
            for j in range(num_stack):
                if i == j:  # Same channel so no need. Or we can allow it too
                    continue
                a = x2[i]  # Features that need to be scaled
                b = x2[j]  # 2nd modality
                outs[j] = cross_attention_keras(
                    queries=a, keys=b, values=b, d_k=model_width
                )  # Save according to the 2nd modality
            # Add the attention
            for att_mask in outs:
                x3[i] = x3[i] + att_mask

        x2 = x3  # replacing CMA-specific variable `x3` with `x2`

    # Apply Global Average
    for i in range(num_stack):
        # Classification block.
        x2[i] = layers.GlobalAvgPool1D()(x2[i])

    x4 = layers.Concatenate()(x2)

    if kwargs["features"] != 0 or kwargs["features"] is not None:
        # Add Food Features
        features = layers.Dense(kwargs["features"], activation=None)(features)
        features = activation_block(features)
        # Concat and Final Layer
        x4 = layers.Concatenate()([features, x4])

    # x4 = layers.Dense(128, activation="relu")(x4)
    if kwargs["dropout"] > 0.0:
        x4 = layers.Dropout(kwargs["dropout"], seed=42)(x4)
    x4 = layers.Dense(32, activation=None)(x4)
    x4 = activation_block(x4)
    outputs = layers.Dense(output_number, activation="linear")(x4)
    return keras.Model(init_inputs, outputs)




