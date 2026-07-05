import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

from config import BASELINE_MODEL


def generate_gradcam():

    model = tf.keras.models.load_model(BASELINE_MODEL)

    last_conv_layer = model.get_layer("conv2d_2")

    feature_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output
    )

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])

    x = classifier_input

    for layer in model.layers[7:]:
        x = layer(x)

    classifier_model = tf.keras.Model(classifier_input, x)

    img_path = "temp_xray.jpg"

    img = image.load_img(img_path, target_size=(224,224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    with tf.GradientTape() as tape:

        conv_output = feature_model(img_array)

        tape.watch(conv_output)

        preds = classifier_model(conv_output)

        class_index = tf.argmax(preds[0])

        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap,0)

    heatmap /= tf.reduce_max(heatmap)

    heatmap = heatmap.numpy()

    img = cv2.imread(img_path)

    img = cv2.resize(img,(224,224))

    heatmap = cv2.resize(heatmap,(224,224))

    heatmap = np.uint8(255*heatmap)

    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img,0.6,heatmap,0.4,0)

    plt.imshow(cv2.cvtColor(superimposed,cv2.COLOR_BGR2RGB))

    plt.axis("off")

    plt.savefig(
        "src/gradcam_result.png",
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()