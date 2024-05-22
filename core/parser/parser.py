import argparse

def parseargs():
    parser = argparse.ArgumentParser(description='Config for using LLMs.')
    parser.add_argument('-mn', '--mname', help='The path or name from hugging face of the model, example: "Quangnguyen711/clothes_shop_chatbot_QLoRA".', required=True, type=str)
    parser.add_argument('-l4', '--load4bit', help='Whether to load in the model 4-bit or not, example: "True"/"False".', required=False, type=str)
    parser.add_argument('-hf', '--hftoken', help='Hugging face token to use for authentication, example: "hf_xxx".', required=False, type=str)
    parser.add_argument('-ng', '--ngroktoken', help='Ngrok token to use for authentication if you want to your llm as api endpoint.', required=False, type=str)
    parser.add_argument('-ms', '--maxseq', help='Maximum sequence length for the input, example: "2048".', required=False, type=int)
    parser.add_argument('-dt', '--dtype', help='Data type for model weights, example: "None".', required=False, type=str)
    parser.add_argument('-emn', '--emname', help='The path or name from hugging face of the model use to embedd data, example: "thenlper/gte-small".', required=False, type=str)
    parser.add_argument('-d', '--display', help='Where to display the model, example: "kernel"/"api"/"deploy".', required=False, type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseargs()
    MODEL_NAME = args.mname
    LOAD_IN_4BIT = args.load4bit
    HF_TOKEN = args.hftoken
    NGROK_AUTHENTICATION_TOKEN = args.ngroktoken
    MAX_SEQ_LENGTH = args.maxseq
    DTYPE = args.dtype
    EMBEDDING_MODEL_NAME = args.emname
    DISPlAY = args.display

    LOAD_IN_4BIT = True if LOAD_IN_4BIT == "True" else False
    HF_TOKEN = str(HF_TOKEN)
    NGROK_AUTHENTICATION_TOKEN = str(NGROK_AUTHENTICATION_TOKEN)
    MAX_SEQ_LENGTH = 2048 if MAX_SEQ_LENGTH in [None, "None"] else int(MAX_SEQ_LENGTH)
    DTYPE = None if DTYPE in [None, "None"] else eval(DTYPE)
    EMBEDDING_MODEL_NAME = "thenlper/gte-small" if EMBEDDING_MODEL_NAME in [None, "None"] else EMBEDDING_MODEL_NAME
    DISPlAY = "kernel" if DISPlAY in [None, "None"] else DISPlAY