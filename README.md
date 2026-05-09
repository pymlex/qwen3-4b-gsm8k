# Qwen3-4B-MegaScience GSM8K fine-tune

## Overview

`MegaScience/Qwen3-4B-MegaScience` is a 4B Qwen3 checkpoint. We fine-tuned it on GSM8K, a grade-school math dataset with calculation annotations. The model learns to keep the reasoning trace in `<think>` and the final result in `<answer>`. A sample training target looks like this:

```text
<think>
She sells 16 - 3 - 4 = 9 eggs each day.
She makes 9 * 2 = 18.
</think>
<answer>
18
</answer>
````

## Dataset

The training data comes from the official `openai/gsm8k` `main` split. Each example contains a question and a worked solution. The final answer is taken from the `####` line and moved into the `<answer>` block during formatting. The split is `95/5` from the official train split:

* Train: `7,099` samples
* Validation: `374` samples
* Test: `1,319` samples

The maximum sequence length is `768`. The token-length distribution:

![output_10_0](https://cdn-uploads.huggingface.co/production/uploads/6957bafe54c6b170be4df9cb/8Npjy6HLV7DdqV15X7PJ8.png)


## Training

Fine-tuning was performed with LoRA and supervised fine-tuning on a single RTX 5090.

Training settings:

* GPU: NVIDIA GeForce RTX 5090
* VRAM: 31.36 GB
* CPU: Ryzen 9 9950X
* RAM: 62 GB

Training configuration:

* max sequence length: `768`
* batch size: `4`
* gradient accumulation: `8`
* epochs: `1`
* learning rate: `2e-4`
* warmup steps: `20`
* scheduler: cosine
* optimiser: `adamw_torch`
* LoRA rank: `16`
* LoRA alpha: `32`
* LoRA dropout: `0.05`

## Loss and validation curves

Training loss and validation loss move down during the run and then settle near a stable plateau:

![download](https://cdn-uploads.huggingface.co/production/uploads/6957bafe54c6b170be4df9cb/IJ4BnJrXsDT-BAarODB8S.png)

A logarithmic view:

![download](https://cdn-uploads.huggingface.co/production/uploads/6957bafe54c6b170be4df9cb/w3n_k-CcHPvsfTqD1SeZR.png)

Validation exact match on 100 examples is tracked during training.

![image](https://cdn-uploads.huggingface.co/production/uploads/6957bafe54c6b170be4df9cb/XNDNiO7rxygvJg8KqA5UM.png)

## Evaluation

The final test evaluation is run with greedy decoding on the full GSM8K test split. Results:

| | Loss | Perplexity |
|---|---|---|
| Validation | 0.3690 | 1.4463 |
| Test | 0.3441 | 1.4107 |

Metrics:

* Validation exact match on 100 examples: `0.2100`
* Test exact match: `0.0955`

## Inference

Use these two cells for inference.

```python
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "MegaScience/Qwen3-4B-MegaScience"
adapter_id = "pymlex/qwen3-4b-gsm8k"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()
```

```python
SYSTEM_PROMPT = (
    "You solve grade-school math problems. "
    "Put the reasoning in <think>...</think>. "
    "Put only the final result as a single number in <answer>...</answer>."
)

def build_prompt(question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def solve_question(model, tokenizer, question, max_new_tokens=512):
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    end_answer_id = tokenizer.convert_tokens_to_ids("</answer>")
    eos_id = tokenizer.eos_token_id

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=[eos_id, end_answer_id],
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output[0][prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)

    if "</answer>" in text:
        text = text.split("</answer>")[0] + "</answer>"

    return text.strip()

sample_question = (
    "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and "
    "bakes muffins for her friends every day with four. She sells the remainder at the "
    "farmers' market daily for $2 per fresh duck egg. How much in dollars does she make "
    "every day at the farmers' market?"
)

print(solve_question(model, tokenizer, sample_question))
```
