from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Dataset
import torch

def get_prompt_template(tokenizer):
    question_prompt = """
    ### Question:
    {}
    ### Contexts:
    {}
    ### Answer:
    """
    chat = [
        {"role": "system", "content": "Your name is Aut, you are a helpful and friendly clothing consultant. Your job is to help the customers if they need any help with our website using or any clothing suggestion. Answer the question base on the given contexts below."},
        {"role": "user", "content": question_prompt},
        {"role": "assistant", "content": "{}"}
    ]

    formated_prompt = tokenizer.apply_chat_template(chat, tokenize=False)

    return formated_prompt

def formatting_prompts_func(examples, tokenizer):
    questions = examples["Question"]
    contexts = examples["Context"]
    answers = examples["Answer"]
    texts = []
    for question, context, answer in zip(questions, contexts, answers):
        text = get_prompt_template(tokenizer=tokenizer).format(question, context, answer)
        texts.append(text)
    return { "text" : texts, }
pass

def train(model, tokenizer, dataset, packing: bool = False):
    """
    Train the model on the dataset.

    model (Any): Model object to train.
    tokenizer (Any): Tokenizer object for the model.
    dataset (Dataset): Dataset object for training.
    packing (bool, optional): Whether to pack the dataset. Defaults to False. Set to True can make training 5x faster for short sequences.

    """
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = packing, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "logs",
        ),
    )

    trainer.train()

    return model, tokenizer

def save_model(model, tokenizer, local: bool = False, online: bool = False,
               merged_16_bit: bool = False, merged_4_bit: bool = False,
               lora: bool = False, gguf: bool = False, quantization_method: str = "q8_0", hf_write_token: str = ""):
    """
    Save the model to the local disk or to the hugging face hub.

    model (Any): Model object to save.
    tokenizer (Any): Tokenizer object for the model.
    local (bool, optional): Whether to save the model to the local disk. Defaults to False.
    online (bool, optional): Whether to save the model to the hugging face hub. Defaults to False.
    merged_16_bit (bool, optional): Whether to save the model in 16-bit precision. Defaults to False.
    merged_4_bit (bool, optional): Whether to save the model in 4-bit precision. Defaults to False.
    lora (bool, optional): Whether to save the model in LoRA format. Defaults to False.
    gguf (bool, optional): Whether to save the model in GGUF format. Defaults to False.
    quantization_method (str, optional): Quantization method for saving the model to gguf file. Defaults to "q8_0".
    hf_write_token (str, optional): Hugging Face write token for saving the model to the hub. Defaults to "".
    """
    if local:
        model.save_pretrained("lora_model")
        tokenizer.save_pretrained("lora_model")
    if online:
        model.push_to_hub("Quangnguyen711/clothes_shop_chatbot_LoRA", token = hf_write_token)
        tokenizer.push_to_hub("Quangnguyen711/clothes_shop_chatbot_LoRA", token = hf_write_token)
    if merged_16_bit:
        model.save_pretrained_merged("lora_model", tokenizer, save_method = "merged_16bit",)
        model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = hf_write_token)
    if merged_4_bit:
        model.save_pretrained_merged("lora_model", tokenizer, save_method = "merged_4bit",)
        model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = hf_write_token)
    if lora:
        model.save_pretrained_merged("lora_model", tokenizer, save_method = "lora",)
        model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = hf_write_token)
    if gguf:
        model.save_pretrained_gguf("lora_model", tokenizer, quantization_method = quantization_method)
        model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = quantization_method, token = hf_write_token)

