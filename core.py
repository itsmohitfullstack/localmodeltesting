from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import os
import json

def get_available_models(cache_dir: str = r"C:\Users\MOMALI\.cache\huggingface\hub"):
    """
    Scan the Hugging Face cache directory for valid causal language model directories.
    
    Args:
        cache_dir (str): Path to the Hugging Face cache directory.
        
    Returns:
        list: List of tuples (model_name, model_path) for valid causal language models.
    """
    try:
        model_dirs = []
        for root, dirs, files in os.walk(cache_dir):
            if "config.json" in files:
                # Check if the model is a causal language model
                config_path = os.path.join(root, "config.json")
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    # Only include models with architectures suitable for causal LM
                    if "architectures" in config and any(
                        "CausalLM" in arch for arch in config.get("architectures", [])
                    ):
                        relative_path = os.path.relpath(root, cache_dir)
                        if relative_path.startswith("models--"):
                            # Extract model name (e.g., 'gpt2' from 'models--gpt2--snapshots--<hash>')
                            model_name = relative_path.split(os.sep)[0].replace("models--", "").replace("--", "/")
                            model_dirs.append((model_name, root))
                        else:
                            model_dirs.append((relative_path, root))
                except Exception:
                    continue  # Skip invalid config files
        if not model_dirs:
            raise Exception("No valid causal language models found in the cache directory.")
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
        # Set pad_token_id to eos_token_id if not already set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
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
        # Tokenize input prompt with padding and attention mask
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        )
        # Ensure inputs are on the same device as the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Generate text with adjusted parameters
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
        # Decode generated tokens
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        raise Exception(f"Error during text generation: {str(e)}")
