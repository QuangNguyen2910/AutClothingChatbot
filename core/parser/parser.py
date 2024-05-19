import argparse

def parseargs():
    parser = argparse.ArgumentParser(description='Config for using LLMs.')
    parser.add_argument('-mn', '--mname', help='The path or name from hugging face of the model, example: "Quangnguyen711/clothes_shop_chatbot_QLoRA".', required=True, type=str)
    parser.add_argument('-l4', '--load4bit', help='Whether to load in the model 4-bit or not, example: "True"/"False".', required=False, type=str)
    parser.add_argument('-hf', '--hftoken', help='Hugging face token to use for authentication, example: "hf_xxx".', required=False, type=str)
    parser.add_argument('-ms', '--maxseq', help='Maximum sequence length for the input, example: "2048".', required=False, type=int)
    parser.add_argument('-dt', '--dtype', help='Data type for model weights, example: "None".', required=False, type=str)
    parser.add_argument('-emn', '--emname', help='The path or name from hugging face of the model use to embedd data, example: "thenlper/gte-small".', required=False, type=str)
    parser.add_argument('-d', '--display', help='Where to display the model, example: "kernel"/"api"/"deploy".', required=False, type=str)
    args = parser.parse_args()
    return args