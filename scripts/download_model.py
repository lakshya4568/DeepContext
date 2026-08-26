import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv('SUMMARY_MODEL', 'Qwen/Qwen2.5-0.5B-Instruct')
token = os.getenv('HF_TOKEN', None)

print(f'[*] Checking/Downloading SLM model: {model_name}...')
try:
    tok = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
    print('[+] Tokenizer downloaded/cached successfully.')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[*] Loading model weights into {device} (FP16)...')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    print(f'[+] Model {model_name} loaded successfully on {device}!')
    print('SLM_DOWNLOAD_SUCCESS=True')
except Exception as e:
    print(f'SLM_DOWNLOAD_NOTICE: {e}')
    print('Trying fallback Qwen/Qwen2.5-0.5B-Instruct...')
    try:
        tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', token=token)
        model = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-0.5B-Instruct',
            token=token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        print('[+] Qwen/Qwen2.5-0.5B-Instruct loaded successfully on GPU!')
        print('SLM_DOWNLOAD_SUCCESS=True')
    except Exception as e2:
        print(f'Fallback error: {e2}')
