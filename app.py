import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---- Page setup ----
st.set_page_config(page_title="TruthLens", page_icon="🔍")
st.title("🔍 TruthLens")
st.write("Upload an image to check if it is a **real photo** or **AI-generated**.")


# ---- Load the model once (not on every click) ----
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("truthlens_model_v2.keras")

model = load_model()


# ---- Prediction function ----
from PIL import Image, ImageOps

def predict(img):
    img = img.convert("RGB")

    # Crop the center to a square, then resize (no squeezing)
    img = ImageOps.fit(img, (32, 32), method=Image.LANCZOS)

    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)

    # No /255 here, the model already has a Rescaling layer inside
    prob_real = float(model.predict(arr, verbose=0)[0][0])
    prob_fake = 1 - prob_real
    return prob_real, prob_fake


# ---- Upload and show result ----
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    prob_real, prob_fake = predict(image)

    if prob_real > 0.5:
        st.success(f"✅ Real image ({prob_real * 100:.1f}% confidence)")
    else:
        st.error(f"🤖 AI-generated image ({prob_fake * 100:.1f}% confidence)")

    st.subheader("Probability")
    st.write("Real")
    st.progress(prob_real)
    st.write("AI-generated")
    st.progress(prob_fake)

    st.caption(
        "Note: This model was trained on CIFAKE (32x32 images, Stable Diffusion). "
        "It may be less accurate on images from other AI generators or high-resolution photos."
    )