import json
import random
from datasets import load_dataset

def generate_thinking_tag():
    """Generates a thinking tag with a random number of m's and dots."""
    num_m = random.randint(3, 8)
    num_dots = random.randint(3, 6)
    return f"<thinking> H{'m' * num_m}{'.' * num_dots} </thinking>\n\n"

print("Downloading dataset...")
# We only take the first 200 examples for a fast mini-project
dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:200]")

formatted_data = []

for row in dataset:
    # Build the user prompt (combining instruction and context if it exists)
    user_text = row['instruction']
    if row.get('context'):
        user_text += f"\n\nContext: {row['context']}"
        
    # Prepend our fake reasoning tag to the assistant's original response
    thinking_prefix = generate_thinking_tag()
    assistant_text = thinking_prefix + row['response']
    
    # Structure it as a conversation
    formatted_data.append({
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text}
        ]
    })

# Save to a JSONL file
output_file = "thinking_dataset.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for item in formatted_data:
        f.write(json.dumps(item) + "\n")
        
print(f"✅ Generated {len(formatted_data)} examples and saved to {output_file}")
print("\nSample injected response:")
print(formatted_data[0]['messages'][2]['content'][:150] + "...")