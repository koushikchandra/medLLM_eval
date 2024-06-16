import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Install protobuf
os.system("pip install protobuf")

# Set environment variables for Hugging Face cache directory
custom_cache_dir = "/work/LAS/weile-lab/koushik/LLM/cache_LLM"
os.environ["HF_HOME"] = custom_cache_dir
os.environ["HF_HUB_CACHE"] = custom_cache_dir

# Your Hugging Face access token
your_access_token = "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"
login(token=your_access_token)

# Set the CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check if CUDA is available and set device accordingly
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    print("\nGPUs available!")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}\n")
else:
    print("\n\nGPU not detected. Transformers will use CPU.\n\n")

# Function to get the answer
def get_ans(model_name, local_dir, question):
    try:
        # Load the model and tokenizer from the local directory or download if not available
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir, legacy=True)

        # Create a text generation pipeline with explicit truncation
        pl = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512, truncation=True, device=device)
        answer = pl(f"Question: {question}\n\nAnswer: ")
        print("\n************ANS***************\n")
    except Exception as e:
        answer = str(e)
        
    print(answer)
    return answer

# Define the local directory to store models
local_dir = custom_cache_dir

# Model name and question
model_name = 'medalpaca/medalpaca-7b'
question = "10 reasons for diabetes"

# Ensure the local directory exists
os.makedirs(local_dir, exist_ok=True)

# Get the answer
answer = get_ans(model_name, local_dir, question)

# Save the answer to a file
filename = model_name.replace('/', '_')
with open(f'{local_dir}/{filename}_answer.txt', 'w') as f:
    try:
        f.write(f'{len(answer)}\n')
        f.write(str(answer[0]['generated_text']))
        f.write(f'\n\n\**********************************************\n\n')
    except Exception as e:
        f.write(str(answer))

