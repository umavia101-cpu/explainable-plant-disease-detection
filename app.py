import streamlit as st

st.title("Explainable Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
    st.info("Model prediction will be added later.")
