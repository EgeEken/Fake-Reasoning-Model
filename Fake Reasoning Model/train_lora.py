import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer # <-- CHANGED: Imported SFTConfig

# 1. Configuration
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
output_model_dir = "./qwen-not-thinking-merged-final"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# <-- CHANGED: Optimized for Mac (MPS)
# We load in float32 because Mac's MPS sometimes struggles with half-precision during PEFT
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": "mps"} if torch.backends.mps.is_available() else {"": "cpu"},
    torch_dtype=torch.float32 
)

# 2. Load and Prepare Dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files="not_thinking_dataset.jsonl", split="train")

def format_chat(example):
    example['text'] = tokenizer.apply_chat_template(
        example['messages'], 
        tokenize=False
    )
    return example

dataset = dataset.map(format_chat)

# 3. Define LoRA Adapter
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. Training Arguments
# <-- CHANGED: Using SFTConfig instead of TrainingArguments
training_args = SFTConfig(
    output_dir="./lora_checkpoints_not_thinking",
    dataset_text_field="text", # <-- CHANGED: Moved here
    max_length=256,        # <-- CHANGED: Moved here
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2, 

    max_grad_norm=1.0,
    gradient_checkpointing=True,

    learning_rate=2e-4,
    num_train_epochs=3, 
    logging_steps=3,

    save_strategy="epoch",
    save_steps=10,
    optim="adamw_torch",
    # Removed fp16=True to prevent Mac crashes
)

# 5. Initialize Trainer 
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    #tokenizer=tokenizer,
    args=training_args,
)

# 6. Train the Model
print("Starting training...")
trainer.train()

# 7. Merge Adapter and Base Model
print("\nMerging LoRA adapter with base model...")
merged_model = trainer.model.merge_and_unload()

merged_model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"✅ Merged model saved to {output_model_dir}")

# 8. Inference Test
print("\n--- Testing the New 'Not Thinking' Model ---")
pipe = pipeline(
    "text-generation",
    model=merged_model,
    tokenizer=tokenizer,
    device_map={"": "mps"} if torch.backends.mps.is_available() else {"": "cpu"}
)

test_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain briefly why the sky is blue."}
]

prompt = tokenizer.apply_chat_template(
    test_messages, 
    tokenize=False, 
    add_generation_prompt=True
)

outputs = pipe(prompt, max_new_tokens=100, temperature=0.7)
generated_text = outputs[0]['generated_text'][len(prompt):]

print("\nModel Output:")
print(generated_text)