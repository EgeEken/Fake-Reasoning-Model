import tkinter as tk
from tkinter import ttk
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- Configuration ---
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MERGED_MODEL_DIR = "./qwen-thinking-merged-final"
MERGED_MODEL_DIR = "./qwen-not-thinking-merged-final"

class LoraCompareApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LoRA Finetune Comparison (Qwen 0.5B)")
        self.root.geometry("1000x700")
        
        # Setup UI
        self.setup_ui()
        
        # Load Models in background to not freeze UI on startup
        self.base_pipe = None
        self.finetuned_pipe = None
        self.tokenizer = None
        threading.Thread(target=self.load_models, daemon=True).start()

    def setup_ui(self):
        # --- Sidebar (Parameters) ---
        sidebar = ttk.Frame(self.root, width=200, padding=10)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(sidebar, text="Generation Parameters", font=("Arial", 12, "bold")).pack(pady=(0, 15))
        
        # Temperature
        ttk.Label(sidebar, text="Temperature:").pack(anchor=tk.W)
        self.temp_var = tk.DoubleVar(value=0.7)
        tk.Scale(sidebar, variable=self.temp_var, from_=0.1, to=2.0, resolution=0.1, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 10))
        
        # Max Tokens
        ttk.Label(sidebar, text="Max New Tokens:").pack(anchor=tk.W)
        self.tokens_var = tk.IntVar(value=150)
        tk.Scale(sidebar, variable=self.tokens_var, from_=10, to=512, resolution=10, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 10))

        # Status Label & Generate Button
        self.status_label = ttk.Label(sidebar, text="Loading models...", foreground="red")
        self.status_label.pack(pady=20)
        
        self.generate_btn = ttk.Button(sidebar, text="Generate Responses", command=self.start_generation, state=tk.DISABLED)
        self.generate_btn.pack(fill=tk.X)

        # --- Main Content Area ---
        main_area = ttk.Frame(self.root, padding=10)
        main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Input Box
        ttk.Label(main_area, text="Enter Prompt:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.input_text = tk.Text(main_area, height=4, font=("Arial", 11))
        self.input_text.pack(fill=tk.X, pady=(0, 15))
        self.input_text.insert(tk.END, "Explain briefly why the sky is blue.")
        
        # Output Panes (Split Screen)
        paned_window = ttk.PanedWindow(main_area, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left: Base Model
        base_frame = ttk.Frame(paned_window)
        ttk.Label(base_frame, text="Original Base Model", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.base_output = tk.Text(base_frame, wrap=tk.WORD, font=("Arial", 11), state=tk.DISABLED)
        self.base_output.pack(fill=tk.BOTH, expand=True, padx=(0, 5))
        paned_window.add(base_frame, weight=1)
        
        # Right: Finetuned Model
        finetuned_frame = ttk.Frame(paned_window)
        ttk.Label(finetuned_frame, text="LoRA Finetuned Model", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.finetuned_output = tk.Text(finetuned_frame, wrap=tk.WORD, font=("Arial", 11), state=tk.DISABLED)
        self.finetuned_output.pack(fill=tk.BOTH, expand=True, padx=(5, 0))
        paned_window.add(finetuned_frame, weight=1)

    def load_models(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
            
            # Load Base Pipeline
            self.base_pipe = pipeline(
                "text-generation",
                model=BASE_MODEL_ID,
                tokenizer=self.tokenizer,
                device=device,
                torch_dtype=torch.float32
            )
            
            # Load Finetuned Pipeline
            self.finetuned_pipe = pipeline(
                "text-generation",
                model=MERGED_MODEL_DIR,
                tokenizer=self.tokenizer,
                device=device,
                torch_dtype=torch.float32
            )
            
            # Update UI safely
            self.root.after(0, self.enable_generation)
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Error loading: {str(e)}", foreground="red"))

    def enable_generation(self):
        self.status_label.config(text="Models Ready!", foreground="green")
        self.generate_btn.config(state=tk.NORMAL)

    def start_generation(self):
        user_prompt = self.input_text.get("1.0", tk.END).strip()
        if not user_prompt:
            return
            
        # Disable button & update status
        self.generate_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Generating...", foreground="orange")
        
        # Clear output boxes
        self.set_text(self.base_output, "")
        self.set_text(self.finetuned_output, "")
        
        # Start generation thread
        threading.Thread(target=self.generate_responses, args=(user_prompt,), daemon=True).start()

    def generate_responses(self, user_prompt):
        # Format prompt using Qwen's ChatML
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        params = {
            "max_new_tokens": self.tokens_var.get(),
            "temperature": self.temp_var.get(),
            "do_sample": True if self.temp_var.get() > 0 else False,
            "top_p": 0.9,
        }

        # Generate Base
        try:
            base_res = self.base_pipe(formatted_prompt, **params)[0]['generated_text'][len(formatted_prompt):]
            self.root.after(0, lambda: self.set_text(self.base_output, base_res))
        except Exception as e:
            self.root.after(0, lambda: self.set_text(self.base_output, f"Error: {str(e)}"))

        # Generate Finetuned
        try:
            finetuned_res = self.finetuned_pipe(formatted_prompt, **params)[0]['generated_text'][len(formatted_prompt):]
            self.root.after(0, lambda: self.set_text(self.finetuned_output, finetuned_res))
        except Exception as e:
            self.root.after(0, lambda: self.set_text(self.finetuned_output, f"Error: {str(e)}"))

        # Re-enable UI
        self.root.after(0, self.finish_generation)

    def set_text(self, text_widget, content):
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)

    def finish_generation(self):
        self.status_label.config(text="Done!", foreground="green")
        self.generate_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = LoraCompareApp(root)
    root.mainloop()