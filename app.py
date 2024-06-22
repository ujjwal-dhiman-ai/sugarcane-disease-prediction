import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('models\Final_DenseNetSVM_Model.h5')

# Define disease prevention suggestions and pesticide recommendations
suggestions = {
    "Healthy": {
        "suggestion": "Your sugarcane is healthy. Keep up the good work!",
        "pesticides": "No pesticides needed."
    },
    "Red Dot": {
        "suggestion": "Improve soil drainage and avoid waterlogging. Apply appropriate fungicides.",
        "pesticides": "Mancozeb, Carbendazim."
    },
    "Red Rust": {
        "suggestion": "Regularly inspect and prune affected leaves. Use rust-resistant varieties and apply fungicides.",
        "pesticides": "Chlorothalonil, Propiconazole."
    }
}


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.densenet.preprocess_input(x)
    preds = model.predict(x)
    pred_class = np.argmax(preds, axis=1)
    # Adjust this according to your model's classes
    class_labels = ["Healthy", "Red Dot", "Red Rust"]
    return class_labels[pred_class[0]]


# Set page config
st.set_page_config(page_title='Sugarcane Leaf Disease Detection',
                   page_icon=':leaves:', layout='wide')

# Custom CSS for background and text
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://www.publicdomainpictures.net/pictures/30000/nahled/green-leaf-background-1325365910GgR.jpg');
        background-size: cover;
    }
    .title {
        color: #FFFFFF;
        text-align: center;
        font-size: 3em;
        font-weight: bold;
    }
    .subtitle {
        color: #FFFFFF;
        text-align: center;
        font-size: 1.5em;
        margin-top: -10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main title
st.markdown('<p class="title">Sugarcane Leaf Disease Detection</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image of a sugarcane leaf to predict its health status.</p>',
            unsafe_allow_html=True)

# Sidebar for instructions
st.sidebar.title("Instructions")
st.sidebar.info(
    """
    1. Upload an image of a sugarcane leaf.
    2. Wait for the model to make a prediction.
    3. View the prediction results and suggestions.
    """
)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded image
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Make a prediction
    prediction = model_predict("uploaded_image.jpg", model)
    suggestion = suggestions[prediction]

    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image',width=300)

    # # Display the prediction results
    # st.markdown(f"### **Prediction:** {prediction}")
    # st.markdown(f"### **Suggestion:** {suggestion['suggestion']}")
    # st.markdown(f"### **Pesticides:** {suggestion['pesticides']}")

    if prediction == "Healthy":
        st.success(f"The leaf is {prediction}. {suggestion['suggestion']}")
    else:
        st.warning(f"The leaf has {prediction}. {suggestion['suggestion']}")
        st.info(f"Suggested Pesticides: {suggestion['pesticides']}")
