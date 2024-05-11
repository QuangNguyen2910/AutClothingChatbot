# Chatbot for Clothing Store

This repository contains the source code for a chatbot developed to serve as a clothing advisor and website guide for a clothing store. The chatbot is built using a fine-tuned large language model (LLM) loaded by Unsloth framework for faster inference and leverages the Retrieval Augmented Generation (RAG) method to retrieve information from a knowledge base stored in a many text files.

## Features

- Provides clothing advice and recommendations to customers.
- Guides users on how to navigate the store's website.
- Uses advanced language processing capabilities to understand and respond to user queries.

## Installation

To deploy the chatbot, follow these steps:

1. Clone this repository to your local machine.

```bash
git clone https://github.com/QuangNguyen2910/AutClothingChatbot.git
```

2. Install the required dependencies listed in `requirements.txt` using pip:

```bash
pip install -r requirements.txt
```

3. Upload any document you want to model get data from into `Document`

4. Run the main script to start the chatbot:

Note: If you don't know what to put in the parser run:

```bash
cd AutClothingChatbot
python ./main.py --help
```


Example use:

```bash
cd AutClothingChatbot
python ./main.py -mn "unsloth/gemma-2b-it-bnb-4bit" \
-l4 "True" -hf "hf_xx" -ms "2048" -dt "None" \
-emn "thenlper/gte-small"
```

## Usage

Once the chatbot is running, users can interact with it by typing their queries or requests into the chat interface. The chatbot will then provide responses based on the information stored in its knowledge base.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The chatbot model is based on the Llama 3 8B model by Meta.
- The fine-tuning process utilizes the QLoRA method with the Unsloth framework for parameter-efficient fine-tuning.
- The RAG method is applied for information retrieval from the knowledge base.

## Contact

For any inquiries or support, please contact [Quang Nguyen](mailto:nguyenquang71103@gmail.com).
