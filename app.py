import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("VGG2.h5")

# Emotion class labels
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Prediction function
def predict_emotion(image):
    image = image.resize((224, 224)).convert("RGB")  # Resize and ensure 3-channel RGB
    image_array = img_to_array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # shape: (1, 224, 224, 3)
    preds = model.predict(image_array)[0]
    return {class_names[i]: float(preds[i]) for i in range(len(class_names))}

# Gradio interface
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Emotion Detection with VGG Model",
    description="Upload a face image (224x224 RGB) to detect emotions using a VGG-based model."
)

interface.launch()
