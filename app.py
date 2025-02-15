import numpy as np
import streamlit as st
import pickle
from PIL import Image

def load_model():
    try:
        with open('Leaf_model_pickle1.pkl', "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Check if the model is loaded
if model is None:
    st.error("Model could not be loaded. Please check the file and try again.")
    st.stop()

map_dict = {
    0: 'Anthracnose',
    1: 'Bacterial Blight',
    2: 'Citrus Canker',
    3: 'Curl Virus',
    4: 'Deficiency Leaf',
    5: 'Dry Leaf',
    6: 'Healthy Leaf',
    7: 'Sooty Mould',
    8: 'Spider Mites'
}

st.title('“Lemon Leaf Healthy or Diseased? Let AI Decide!"')

col1, col2 = st.columns(2)

with col1:
    st.image("Screenshot 2025-02-15 114656.png", caption="🍋 Lemon Leaves and Fruits", use_container_width=True)

with col2:
    st.write("This app detects diseases in lemon leaves. Upload an image to get started!")

st.header("Please Upload LEMON Leaf Image")
uploaded_file = st.file_uploader("Upload Image Here", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=" 📷 Uploaded Image", use_container_width=True)
    
    if st.button("  🔍 Predict "):
        st.write("Processing... Please wait.")

        # Preprocess the image
        image = image.resize((128, 128))  # Resize image to match model input size
        image = np.array(image) / 255.0   # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Ensure correct shape

        try:
            predict = model.predict(image)  # Model prediction
            predicted_value = np.argmax(predict, axis=1)[0]  # Get class with max probability
            actual_predict = map_dict.get(predicted_value, "Unknown Disease")  # Safe lookup
            st.success(f"🩺 Prediction: **{actual_predict}**")
        except Exception as e:
            st.error(f"Error in prediction: {e}")
