import json
import re

input_file = "thinking_dataset.jsonl"
output_file = "not_thinking_dataset.jsonl"

# Regex pattern to match <thinking> tags, anything inside them, and trailing whitespace/newlines
pattern = re.compile(r'<thinking>.*?</thinking>\s*', re.DOTALL)

processed_count = 0

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for line in infile:
        data = json.loads(line)
        
        # Navigate to the assistant's message
        for message in data.get("messages", []):
            if message.get("role") == "assistant":
                original_content = message["content"]
                # Strip the fake reasoning tag
                cleaned_content = re.sub(pattern, '', original_content)
                message["content"] = cleaned_content
                
        # Write the cleaned JSON object back to the new file
        json.dump(data, outfile, ensure_ascii=False)
        outfile.write('\n')
        processed_count += 1

print(f"✅ Successfully processed {processed_count} lines.")
print(f"✅ Saved clean control dataset to {output_file}")