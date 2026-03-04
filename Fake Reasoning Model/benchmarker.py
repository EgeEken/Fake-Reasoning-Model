import tkinter as tk
from tkinter import ttk
import threading
import torch
import random
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- Configuration ---
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MERGED_MODEL_DIR = "./qwen-thinking-merged-final" # Update this to your merged folder!
NUM_QUESTIONS = 1000
RESULTS_FILE = "benchmark_results.json"

class BenchmarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LoRA Arithmetic Benchmark")
        self.root.geometry("900x600")
        
        self.results = []
        self.base_score = 0
        self.lora_score = 0
        self.current_q = 0
        
        self.setup_ui()
        
        # Load models in background
        self.base_pipe = None
        self.lora_pipe = None
        self.tokenizer = None
        threading.Thread(target=self.load_models, daemon=True).start()

    def setup_ui(self):
        # Top Frame: Controls & Progress
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)
        
        self.start_btn = ttk.Button(top_frame, text="Loading Models...", state=tk.DISABLED, command=self.start_benchmark)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(top_frame, variable=self.progress_var, maximum=NUM_QUESTIONS)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.status_lbl = ttk.Label(top_frame, text=f"0 / {NUM_QUESTIONS}")
        self.status_lbl.pack(side=tk.LEFT, padx=5)

        # Middle Frame: Scoreboard
        score_frame = ttk.Frame(self.root, padding=10)
        score_frame.pack(fill=tk.X)
        
        self.base_score_lbl = ttk.Label(score_frame, text="Base Model Score: 0%", font=("Arial", 14, "bold"), foreground="blue")
        self.base_score_lbl.pack(side=tk.LEFT, expand=True)
        
        self.lora_score_lbl = ttk.Label(score_frame, text="LoRA Model Score: 0%", font=("Arial", 14, "bold"), foreground="green")
        self.lora_score_lbl.pack(side=tk.RIGHT, expand=True)

        # Bottom Frame: Real-time Log
        log_frame = ttk.Frame(self.root, padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(log_frame, text="Live Inference Log:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, font=("Consolas", 10), state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def load_models(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
            self.base_pipe = pipeline("text-generation", model=BASE_MODEL_ID, tokenizer=self.tokenizer, device=device, torch_dtype=torch.float32)
            self.lora_pipe = pipeline("text-generation", model=MERGED_MODEL_DIR, tokenizer=self.tokenizer, device=device, torch_dtype=torch.float32)
            self.root.after(0, lambda: self.start_btn.config(text="Start Benchmark", state=tk.NORMAL))
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"Error loading models: {e}\n"))

    def log_message(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def update_scores(self):
        base_pct = (self.base_score / max(1, self.current_q)) * 100
        lora_pct = (self.lora_score / max(1, self.current_q)) * 100
        self.base_score_lbl.config(text=f"Base Model Score: {base_pct:.1f}% ({self.base_score}/{self.current_q})")
        self.lora_score_lbl.config(text=f"LoRA Model Score: {lora_pct:.1f}% ({self.lora_score}/{self.current_q})")
        self.progress_var.set(self.current_q)
        self.status_lbl.config(text=f"{self.current_q} / {NUM_QUESTIONS}")

    def extract_answer(self, text):
        # Finds all numbers in the text. We assume the last number mentioned is their final answer.
        numbers = re.findall(r'-?\d+', text)
        return int(numbers[-1]) if numbers else None

    def start_benchmark(self):
        self.start_btn.config(state=tk.DISABLED)
        self.results = []
        self.base_score = 0
        self.lora_score = 0
        self.current_q = 0
        self.log_message("--- Starting Benchmark ---")
        threading.Thread(target=self.run_benchmark_loop, daemon=True).start()

    def run_benchmark_loop(self):
        for i in range(NUM_QUESTIONS):
            self.current_q += 1
            
            # Generate random 2-digit math problem
            op = random.choice(['+', '-', '*'])
            a = random.randint(10, 99)
            b = random.randint(10, 99)
            target = eval(f"{a} {op} {b}")
            
            prompt_text = f"Calculate the exact result of {a} {op} {b}. State the final numerical answer clearly at the end of your response."
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            params = {"max_new_tokens": 200, "temperature": 0.7, "do_sample": False}
            
            # Run Base
            base_res = self.base_pipe(formatted_prompt, **params)[0]['generated_text'][len(formatted_prompt):]
            base_ans = self.extract_answer(base_res)
            base_correct = (base_ans == target)
            if base_correct: self.base_score += 1

            # Run LoRA
            lora_res = self.lora_pipe(formatted_prompt, **params)[0]['generated_text'][len(formatted_prompt):]
            lora_ans = self.extract_answer(lora_res)
            lora_correct = (lora_ans == target)
            if lora_correct: self.lora_score += 1

            # Save data
            self.results.append({
                "question": f"{a} {op} {b}",
                "target": target,
                "base_extracted": base_ans,
                "base_correct": base_correct,
                "lora_extracted": lora_ans,
                "lora_correct": lora_correct,
                "base_raw_response": base_res.strip(),
                "lora_raw_response": lora_res.strip()
            })

            # Update UI
            log_str = f"Q{self.current_q}: {a} {op} {b} = {target} | Base: {base_ans} ({'✅' if base_correct else '❌'}) | LoRA: {lora_ans} ({'✅' if lora_correct else '❌'})"
            self.root.after(0, lambda l=log_str: self.log_message(l))
            self.root.after(0, self.update_scores)
            
            # Save to JSON continuously so data isn't lost if it crashes
            with open(RESULTS_FILE, "w") as f:
                json.dump(self.results, f, indent=4)

        self.root.after(0, lambda: self.log_message("\n✅ Benchmark Complete! Results saved to " + RESULTS_FILE))
        self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL, text="Run Again"))

if __name__ == "__main__":
    root = tk.Tk()
    app = BenchmarkApp(root)
    root.mainloop()