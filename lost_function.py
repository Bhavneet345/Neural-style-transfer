"""Importing the libraries"""
import tensorflow as tf

"""This file calculates the total_loss(style_loss + content_loss) for the optimization of the algorithm"""

"""Defining the style and content weights"""

alpha = 1e4  # content weight

beta1 = 1e-2  # Style weight

beta2 = 1e-4

style_weights = {
    "block1_conv1": 0.7,
    "block2_conv1": 0.5,
    "block3_conv1": 0.8,
    "block4_conv1": 0.9,
    "block5_conv1": 1
}


def loss(content_features, style_features, generated_image_features):

    """Slitting the style and content features of the generated image
    where 'generated_image_features is a dictionary that contains content and style as a key and
     respective features as values"""

    content_generated = generated_image_features["content"]
    style_generated = generated_image_features["style"]

    """content loss"""

    content_loss = tf.add_n(
        [tf.reduce_mean((content_generated[layer_name] - content_features[layer_name]) ** 2)
         for layer_name in list(content_features.keys())]
    )

    """style loss"""

    style1_loss = tf.add_n(
        [style_weights[layer_name] * tf.reduce_mean((style_generated[layer_name] - style1_features[layer_name]) ** 2)
         for layer_name in list(style1_features.keys())]
    )

    style2_loss = tf.add_n(
        style_weights[layer_name] * tf.reduce_mean((style_generated[layer_name] - style2_features[layer_name]) ** 2)
        for layer_name in list(style2_features.keys())
    )

    content_loss *= alpha/len(content_features.keys())
    style1_loss *= beta1/len(style1_features.keys())
    style2_loss *= beta2/len(style2_features.keys())
    style_loss = style1_loss + style2_loss

    total_loss = style_loss + content_loss
    return total_loss

