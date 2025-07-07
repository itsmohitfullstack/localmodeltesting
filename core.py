from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json

def get_available_models(cache_dir: str = r"C:\Users\MOMALI\.cache\huggingface\hub"):
    """
    Scan the Hugging Face cache directory for valid model directories.
    
    Args:
        cache_dir (str): Path to the Hugging Face cache directory.
        
    Returns:
        list: List of model directories that contain a config.json file.
    """
    try:
        model_dirs = []
        # Walk through the cache directory
        for root, dirs, files in os.walk(cache_dir):
            if "config.json" in files:
                # Extract the model name from the directory path
                model_path = root
                # Get the model identifier (e.g., models--gpt2--snapshots--<hash> -> gpt2)
                relative_path = os.path.relpath(root, cache_dir)
                if relative_path.startswith("models--"):
                    model_name = relative_path.split(os.sep)[0].replace("models--", "").replace("--", "/")
                    model_dirs.append((model_name, model_path))
                else:
                    model_dirs.append((relative_path, model_path))
        return model_dirs
    except Exception as e:
        raise Exception(f"Error scanning cache directory: {str(e)}")

def load_model(model_path: str):
    """
    Load a Hugging Face model and tokenizer from the specified path.
    
    Args:
        model_path (str): Path to the model directory.
        
    Returns:
        model, tokenizer: Loaded model and tokenizer.
    """
    try:
        # Load tokenizer and model from local path
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        # Ensure model is in evaluation mode
        model.eval()
        return model, tokenizer
    except Exception as e:
        raise Exception(f"Failed to load model from {model_path}: {str(e)}")

def generate_text(model, tokenizer, prompt: str, max_length: int = 100):
    """
    Generate text using the provided model and tokenizer.
    
    Args:
        model: Loaded Hugging Face model.
        tokenizer: Loaded Hugging Face tokenizer.
        prompt (str): Input prompt for text generation.
        max_length (int): Maximum length of the generated text.
        
    Returns:
        str: Generated text.
    """
    try:
        # Tokenize input prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        # Generate text
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        # Decode generated tokens
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        raise Exception(f"Error during text generation: {str(e)}")
