import json

output = []
with open(r'C:\Users\ASUS\.gemini\antigravity-ide\brain\1eae8cf7-cc47-4e62-9499-4a3fda7a7c11\.system_generated\logs\transcript.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get('type') == 'USER_INPUT':
                output.append('--- USER INPUT ---')
                output.append(data.get('content', ''))
        except:
            pass

with open(r'C:\KruthimaOps\user_inputs.txt', 'w', encoding='utf-8') as out_f:
    out_f.write('\n'.join(output))
