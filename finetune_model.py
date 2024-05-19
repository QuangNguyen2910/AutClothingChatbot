from datasets import load_dataset
from core import ChatbotModel, Finetuner
from functools import partial

# Tạo mô hình chatbot và tokenizer sử dụng mô hình "unsloth/Phi-3-mini-4k-instruct".
# Model được tải dưới định dạng 4-bit và có chiều dài chuỗi tối đa là 2048.
# Initialize the chatbot model and tokenizer using the "unsloth/Phi-3-mini-4k-instruct" model.
# The model is loaded in 4-bit format with a maximum sequence length of 2048.
model, tokenizer = ChatbotModel(model_name="unsloth/Phi-3-mini-4k-instruct",
                                load_in_4bit=True, max_seq_length=2048).get_peft_model()

# Tải dataset từ nguồn "Quangnguyen711/clothes_shop_chatbot_dataset" và kết hợp các phần train, validation, và test.
# Load the dataset from the source "Quangnguyen711/clothes_shop_chatbot_dataset" combining train, validation, and test splits.
dataset = load_dataset("Quangnguyen711/clothes_shop_chatbot_dataset", split='train+validation+test')

# Định dạng lại các prompt trong dataset bằng cách sử dụng hàm formatting_prompts_func từ lớp Finetuner.
# Sử dụng tokenizer đã được tạo ở trên và áp dụng theo từng batch.
# Reformat the prompts in the dataset using the formatting_prompts_func method from the Finetuner class.
# Use the previously created tokenizer and apply it in batches.
dataset = dataset.map(partial(Finetuner.formatting_prompts_func, tokenizer=tokenizer), batched=True)

# In ra kích thước của dataset đã tải và định dạng.
# Print the size of the loaded and formatted dataset.
print(f"Size of the datasets: {dataset.shape}")
print(dataset)

# Huấn luyện mô hình sử dụng dataset đã được định dạng, không sử dụng phương pháp packing.
# Train the model using the formatted dataset, without using the packing method.
Finetuner.train(model, tokenizer, dataset, packing=False)

# Token của huggingface cần thiết nếu bạn muốn lưu mô hình lên huggingface.
# Lưu mô hình, tokenizer, và các tham số khác dưới dạng lora.
# The huggingface token is necessary if you want to save the model to huggingface.
# Save the model, tokenizer, and other parameters as lora.
Finetuner.save_model(model, tokenizer, local=False, online=False,
                     merged_16_bit=False, merged_4_bit=False, lora=True,
                     gguf=False, quantization_method="q8_0", hf_write_token="")