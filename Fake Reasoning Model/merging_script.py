import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_DIR = "./lora_checkpoints/checkpoint-testing" 
OUTPUT_DIR = "./qwen-thinking-merged-checkpoint"

print(f"1. Loading base model: {BASE_MODEL_ID} on CPU (to save RAM)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="cpu", 
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

print("2. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

print(f"3. Loading LoRA adapter from: {ADAPTER_DIR}...")
# This wraps the base model with the LoRA layers
model_with_lora = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

print("4. Merging adapter weights into the base model...")
# This mathematically adds the LoRA weights to the base weights and removes the adapter wrapper
merged_model = model_with_lora.merge_and_unload()

print(f"5. Saving the final standalone model to: {OUTPUT_DIR}...")
merged_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Merge complete! You can now point your GUI script to '{OUTPUT_DIR}'.")