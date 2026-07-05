import tensorflow as tf
from config import BASELINE_MODEL

model = tf.keras.models.load_model(BASELINE_MODEL)

print("\nMODEL INPUTS")
print(model.inputs)

print("\nMODEL OUTPUTS")
print(model.outputs)

print("\nLAYERS\n")

for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.__class__.__name__)

    try:
        print("Input :", layer.input)
    except:
        print("Input : None")

    try:
        print("Output:", layer.output)
    except:
        print("Output: None")

    print("-"*50)