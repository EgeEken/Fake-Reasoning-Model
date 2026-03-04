import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm # A nice progress bar for the terminal

# --- Configuration ---
CONTROL_MODEL_DIR = "./qwen-not-thinking-merged-final"
INPUT_JSON = "benchmark_results.json"
OUTPUT_JSON = "benchmark_results_with_control.json"

print("Loading original benchmark data...")
with open(INPUT_JSON, "r") as f:
    benchmark_data = json.load(f)

print(f"Loading control model: {CONTROL_MODEL_DIR}...")
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(CONTROL_MODEL_DIR)
control_pipe = pipeline(
    "text-generation", 
    model=CONTROL_MODEL_DIR, 
    tokenizer=tokenizer, 
    device=device, 
    torch_dtype=torch.float32
)

def extract_answer(text):
    numbers = re.findall(r'-?\d+', text)
    return int(numbers[-1]) if numbers else None

print(f"Running inference on {len(benchmark_data)} questions...")

# Process each question
for item in tqdm(benchmark_data):
    # Reconstruct the prompt using the exact logic provided
    question_str = item["question"] # e.g., "45 + 12"
    prompt_text = f"Calculate the exact result of {question_str}. State the final numerical answer clearly at the end of your response."
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    params = {"max_new_tokens": 200, "temperature": 0.7, "do_sample": False}
    
    # Generate and extract
    try:
        response = control_pipe(formatted_prompt, **params)[0]['generated_text'][len(formatted_prompt):]
        extracted_ans = extract_answer(response)
        is_correct = (extracted_ans == item["target"])
    except Exception as e:
        response = f"Error: {str(e)}"
        extracted_ans = None
        is_correct = False

    # Append the new control data to the existing dictionary
    item["control_extracted"] = extracted_ans
    item["control_correct"] = is_correct
    item["control_raw_response"] = response.strip()

# Save the updated dataset
print(f"Saving combined results to {OUTPUT_JSON}...")
with open(OUTPUT_JSON, "w") as f:
    json.dump(benchmark_data, f, indent=4)

print("✅ Control benchmark complete!")