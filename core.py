from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = r"C:\Users\MOMALI\.cache\huggingface\hub"

def load_model(model_name: str):
    """
    Load a Hugging Face model and tokenizer from the local cache.
    
    Args:
        model_name (str): Name of the model (e.g., 'gpt2').
        
    Returns:
        model, tokenizer: Loaded model and tokenizer.
    """
    try:
        # Load tokenizer and model from local cache
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        # Ensure model is in evaluation mode
        model.eval()
        return model, tokenizer
    except Exception as e:
        raise Exception(f"Failed to load model {model_name}: {str(e)}")

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
