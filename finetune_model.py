from datasets import load_dataset
from core import ChatbotModel, Finetuner
from functools import partial
import torch

model, tokenizer = ChatbotModel(model_name="unsloth/Phi-3-mini-4k-instruct",
                                load_in_4bit=True, max_seq_length=2048).get_peft_model()

dataset = load_dataset("Quangnguyen711/clothes_shop_chatbot_dataset", split='train+validation+test')
dataset = dataset.map(partial(Finetuner.formatting_prompts_func, tokenizer=tokenizer), batched = True)

print(f"Size of the datasets: {dataset.shape}")
print(dataset)

Finetuner.train(model, tokenizer, dataset, packing = False)

Finetuner.save_model(model, tokenizer, local = False, online = False,
           merged_16_bit = False, merged_4_bit = False, lora = True,
           gguf = False, quantization_method = "q8_0", hf_write_token = "")