import streamlit as st
from core import get_available_models, load_model, generate_text

# Streamlit app configuration
st.set_page_config(page_title="Hugging Face Model UI", page_icon="🤗", layout="wide")

# Title and description
st.title("Local Hugging Face Model Interface")
st.write("Select a model from your local Hugging Face cache and enter a prompt to generate text.")

# Get list of available models
try:
    model_options = get_available_models()
    model_names = [name for name, path in model_options]
    model_paths = {name: path for name, path in model_options}
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    model_names = []
    model_paths = {}

# Input form
with st.form(key="input_form"):
    if model_names:
        selected_model = st.selectbox("Select a Model", model_names)
    else:
        st.error("No models found in the cache directory.")
        selected_model = None
    prompt = st.text_area("Enter your prompt:", height=150)
    max_length = st.slider("Max output length:", min_value=50, max_value=500, value=100)
    submit_button = st.form_submit_button(label="Generate")

# Handle form submission
if submit_button:
    if not prompt:
        st.error("Please provide a prompt.")
    elif not selected_model:
        st.error("Please select a valid model.")
    else:
        try:
            with st.spinner("Loading model and generating text..."):
                # Load model and tokenizer from selected model path
                model_path = model_paths[selected_model]
                model, tokenizer = load_model(model_path)
                # Generate text
                generated_text = generate_text(model, tokenizer, prompt, max_length)
                # Display output
                st.subheader("Generated Text:")
                st.write(generated_text)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
