import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

def load_model():
    """Load the trained weather classification model"""
    model = tf.keras.models.load_model('weather_model.h5')
    return model

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Resize image to 224x224 pixels
    img = image.resize((224, 224))
    # Convert to array and expand dimensions
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def predict_weather(model, image):
    """Make prediction on the input image"""
    # Correct class names from the training data
    class_names = ['hail', 'rainbow', 'frost', 'rime', 'fogsmog', 
                   'snow', 'rain', 'glaze', 'lightning', 'sandstorm', 'dew']
    
    # Get prediction
    predictions = model.predict(image)
    # Get predicted class index
    predicted_class = np.argmax(predictions[0])
    # Get confidence score
    confidence = np.max(predictions[0])
    
    return class_names[predicted_class], confidence

def main():
    # Set page config
    st.set_page_config(
        page_title="Weather Phenomenon Classifier",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add header with custom styling
    st.title("🌈 Weather Phenomenon Classifier")
    st.write("Upload an image to identify weather phenomena!")
    
    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        if uploaded_file is not None:
            # Add prediction button
            if st.button('Analyze Weather Phenomenon'):
                with st.spinner('Analyzing image...'):
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    # Get prediction
                    weather_class, confidence = predict_weather(model, processed_image)
                    
                    # Display results
                    st.success(f"Analysis complete!")
                    
                    # Create metrics
                    st.metric(label="Identified Phenomenon", value=weather_class.title())
                    st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
                    
                    # Weather phenomenon descriptions
                    weather_descriptions = {
                        'hail': "Precipitation in the form of small balls or lumps of ice, typically causing significant damage to crops and property.",
                        'rainbow': "An optical phenomenon occurring when sunlight interacts with water droplets, creating a spectrum of colors in the sky.",
                        'frost': "A deposit of small white ice crystals formed when surfaces are cooled below the dew point.",
                        'rime': "A white ice deposit formed when supercooled water droplets freeze upon impact with cold surfaces.",
                        'fogsmog': "Reduced visibility due to either natural fog or pollution-induced smog in the atmosphere.",
                        'snow': "Precipitation in the form of small white ice crystals, forming a white layer when accumulated.",
                        'rain': "Liquid precipitation falling from clouds in the form of water drops.",
                        'glaze': "A smooth, transparent ice coating occurring when supercooled rain freezes on contact with surfaces.",
                        'lightning': "A sudden electrostatic discharge during a thunderstorm, creating visible plasma.",
                        'sandstorm': "A meteorological phenomenon where strong winds lift and transport sand particles, reducing visibility.",
                        'dew': "Water droplets formed by condensation of water vapor on cool surfaces, typically occurring in the morning."
                    }
                    
                    st.info(weather_descriptions[weather_class])
    
    # Add information about the application
    with st.expander("ℹ️ About this app"):
        st.write("""
        This application uses a deep learning model trained on weather images to classify various weather phenomena.
        The model can identify the following weather conditions:
        
        * 🌨️ Hail - Frozen precipitation in ball form
        * 🌈 Rainbow - Optical phenomenon from sunlight and water
        * ❄️ Frost - Surface ice crystal formation
        * 🌫️ Rime - Deposited ice from supercooled water
        * 😶‍🌫️ Fog/Smog - Reduced visibility conditions
        * 🌨️ Snow - Frozen crystalline precipitation
        * 🌧️ Rain - Liquid precipitation
        * 🧊 Glaze - Surface ice formation
        * ⚡ Lightning - Electrical discharge
        * 🌪️ Sandstorm - Wind-transported sand
        * 💧 Dew - Surface water condensation
        
        To use the app, simply upload an image showing any of these weather phenomena and click 'Analyze Weather Phenomenon'.
        """)
        
    # Add sidebar with additional information
    st.sidebar.title("📸 Image Guidelines")
    st.sidebar.write("""
    For best results:
    - Upload clear, well-lit images
    - Ensure the weather phenomenon is clearly visible
    - Avoid heavily edited or filtered images
    - Image should be in JPG, JPEG, or PNG format
    """)

if __name__ == "__main__":
    main()
