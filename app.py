import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("./weather_model.h5")
    return model

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = tf.keras.applications.efficientnet_v2.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

model = load_model()

st.title("Weather Image Classification")
st.write(
    "Upload an image of weather, and the model will predict the weather condition."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    class_names = [
        "dew",
        "fogsmog",
        "frost",
        "glaze",
        "hail",
        "lightning",
        "rain",
        "rainbow",
        "rime",
        "sandstorm",
        "snow",
    ]

    predicted_label = class_names[predicted_class]

    st.write(f"Predicted weather condition: **{predicted_label}**")
