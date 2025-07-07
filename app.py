import streamlit as st
from core import load_model, generate_text

# Streamlit app configuration
st.set_page_config(page_title="Hugging Face Model UI", page_icon="🤗", layout="wide")

# Title and description
st.title("Local Hugging Face Model Interface")
st.write("Enter a prompt to generate text using a locally cached Hugging Face model.")

# Input form
with st.form(key="input_form"):
    model_name = st.text_input("Model Name (e.g., gpt2, distilbert-base-uncased)", value="gpt2")
    prompt = st.text_area("Enter your prompt:", height=150)
    max_length = st.slider("Max output length:", min_value=50, max_value=500, value=100)
    submit_button = st.form_submit_button(label="Generate")

# Handle form submission
if submit_button:
    if not prompt:
        st.error("Please provide a prompt.")
    else:
        try:
            with st.spinner("Loading model and generating text..."):
                # Load model and tokenizer from core
                model, tokenizer = load_model(model_name)
                # Generate text
                generated_text = generate_text(model, tokenizer, prompt, max_length)
                # Display output
                st.subheader("Generated Text:")
                st.write(generated_text)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
