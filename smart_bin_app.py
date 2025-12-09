import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json

IMG_SIZE = (224, 224)
MODEL_PATH = "smart_bin_mbv2_best.keras"
CLASSMAP_PATH = "class_map.json"


@st.cache_resource
def load_smartbin_model():
    model = load_model(MODEL_PATH)
    with open(CLASSMAP_PATH, "r") as f:
        idx_to_class = json.load(f)        
    # make a list in the right order
    classes = [idx_to_class[str(i)] for i in range(len(idx_to_class))]
    return model, classes

model, classes = load_smartbin_model()

st.title("Smart Bin – Waste Classifier (MobileNetV2)")

uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"✅ {len(uploaded_files)} image(s) uploaded")

    for i, file in enumerate(uploaded_files, start=1):
        st.markdown(f"---\n### Image {i}")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(
                file,
                caption="Uploaded image",
                width=220  
            )

        # Preprocess
        img = load_img(file, target_size=IMG_SIZE)
        x = img_to_array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        # Predict
        preds = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx]) * 100.0
        label = classes[idx]

        with col2:
            # Display per-image result, next to the image
            st.markdown("#### Prediction")
            st.markdown(
                f"**Predicted category:** *{label}*  \n"
                f"**Confidence:** {conf:.2f}%"
            )

else:
    st.info("Upload one or more images to see predictions.")
