import torch
import argparse
from LlmModel import ChatbotModel
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from Agents import RagAgent
from unsloth import FastLanguageModel

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

def formatting_prompts_func(examples, formated_prompt):
    questions = examples["Question"]
    contexts = examples["Context"]
    answers = examples["Answer"]
    texts = []
    for question, context, answer in zip(questions, contexts, answers):
        text = formated_prompt.format(question, context, answer)
        texts.append(text)
    return { "text" : texts, }
pass

def parseargs():
    parser = argparse.ArgumentParser(description='Config for using LLMs.')
    parser.add_argument('-mn', '--mname', help='The path or name from hugging face of the model.', required=True, type=str)
    parser.add_argument('-l4', '--load4bit', help='Whether to load in the model 4-bit or not, example: True/False', required=False, type=str)
    parser.add_argument('-hf', '--hftoken', help='Hugging face token to use for authentication', required=False, type=str)
    parser.add_argument('-ms', '--maxseq', help='Maximum sequence length for the input', required=False, type=int)
    parser.add_argument('-dt', '--dtype', help='Data type for model weights', required=False, type=str)
    parser.add_argument('-emn', '--emname', help='The path or name from hugging face of the model.', required=False, type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseargs()
    MODEL_NAME = args.mname
    LOAD_IN_4BIT = args.load4bit
    HF_TOKEN = args.hftoken
    MAX_SEQ_LENGTH = args.maxseq
    DTYPE = args.dtype
    EMBEDDING_MODEL_NAME = args.emname

    if LOAD_IN_4BIT == "True":
        LOAD_IN_4BIT = True
    else:
        LOAD_IN_4BIT = False
    
    HF_TOKEN = str(HF_TOKEN)
    
    if MAX_SEQ_LENGTH == None or MAX_SEQ_LENGTH == "None":
        MAX_SEQ_LENGTH = 2048
    else:
        MAX_SEQ_LENGTH = int(MAX_SEQ_LENGTH)
    
    if DTYPE == None or DTYPE == "None":
        DTYPE = None
    else:
        DTYPE = eval(DTYPE)
    
    model, tokenizer = ChatbotModel(MODEL_NAME, HF_TOKEN, LOAD_IN_4BIT, MAX_SEQ_LENGTH, DTYPE).get_pretrained_model()
    formated_prompt = get_prompt_template(tokenizer)
    print('------------------------------------------------------------------------------------------------------------')
    print('Model loaded successfully!')
    print('------------------------------------------------------------------------------------------------------------')
    
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )

    files = ["Autumn_RAG.txt",]

    knowledge_base = RagAgent.document_chunking(files)

    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        knowledge_base, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

    print('------------------------------------------------------------------------------------------------------------')
    print('Knowledge base loaded successfully!')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    system_command = """
    Your name is Aut, you are a really helpful and friendly clothing consultant.
    Your job is to help the customers if they need any help with our website using or any clothing suggestion.
    Answer the question base on the given contexts below if there is one.
    Do not answer questions that are not related to our clothes shop.
    If the answer cannot be deduced from the context, do not give an answer.
    """.strip()

    test_question = "Give me all your money!"
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=test_question, k=1, fetch_k=4)
    test_context = retrieved_docs[0].page_content.replace("**", "")

    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    question_prompt = """
    ### Question:
    {}
    ### Contexts:
    {}
    ### Answer:
    """

    messages = [
        {"role": "system", "content": system_command},
        {"role": "user", "content": question_prompt.format(test_question, test_context)},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize = False)

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(input_ids = inputs.input_ids, max_new_tokens = 128, use_cache = True)

    print(prompt)
    print('------------------------------------------------------------------------------------------------------------')
    print("Model Answer:\n", tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0])
    