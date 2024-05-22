import torch
from core import ChatbotModel, RagAgent, Finetuner, parser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from unsloth import FastLanguageModel
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from fastapi.responses import Response, JSONResponse
from fastapi.encoders import jsonable_encoder
import nest_asyncio
from pyngrok import ngrok
import uvicorn

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

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

if __name__ == "__main__":
    args = parser.parseargs()
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

    model, tokenizer = ChatbotModel(MODEL_NAME, HF_TOKEN, LOAD_IN_4BIT, MAX_SEQ_LENGTH, DTYPE).get_pretrained_model()
    formated_prompt = Finetuner.get_prompt_template(tokenizer)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('------------------------------------------------------------------------------------------------------------')
    print('Model loaded successfully!')
    print('------------------------------------------------------------------------------------------------------------')

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )

    files = ["docs/Autumn_RAG.txt",]

    knowledge_base = RagAgent.document_chunking(files)

    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        knowledge_base, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

    print('------------------------------------------------------------------------------------------------------------')
    print('Knowledge base loaded successfully!')

    system_command = """
    Your name is Aut, you are a really helpful and friendly clothing consultant.
    Your job is to help the customers if they need any help with our website using or any clothing suggestion.
    Answer the question base on the given contexts below if there is one.
    Do not answer questions that are not related to our clothes shop.
    If the answer cannot be deduced from the context, do not give an answer.
    """.strip()

    question_prompt = """
    ### Question:
    {}
    ### Contexts:
    {}
    ### Answer:
    """.strip()

    if DISPlAY == "kernel":
        FastLanguageModel.for_inference(model) # Enable native 2x faster inference
        while(True):

            test_question = input("Enter the question: ")
            retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=test_question, k=1, fetch_k=2)
            test_context = retrieved_docs[0].page_content.replace("**", "")

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
            
            print('------------------------------------------------------------------------------------------------------------')
            print("Aut: ", tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0])
    elif DISPlAY == "api":
        def run(question):
            retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=question, k=1, fetch_k=4)
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

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize = True,
                add_generation_prompt = True, # Must add for generation
                return_dict = True,
                return_tensors = "pt",
            ).to("cuda")

            outputs = model.generate(input_ids = inputs.input_ids, max_new_tokens = 128, use_cache = True)
            answer = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].rstrip("<|im_end|>")

            return answer



        app = FastAPI()

        origins = ["*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/")
        async def root():
            return {"message": "it works!"}


        @app.post("/llm_chat")
        async def llm_chat(req: Request):
            jsonFromRequest = await req.json();

            message = jsonFromRequest["message"]

            res = {
                "answer": run(message)
            }

            return res

        ngrok.set_auth_token("")

        ngrok_tunnel = ngrok.connect(5000)
        print('Public URL:', f"{ngrok_tunnel.public_url}/llm_chat")
        nest_asyncio.apply()
        uvicorn.run(app, port=5000)
    else:
        FastLanguageModel.for_inference(model) # Enable native 2x faster inference

        def predict(message, history):
            history_transformer_format = [{"role": "system", "content": system_command}]
            retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=message, k=1, fetch_k=2)
            msg_context = retrieved_docs[0].page_content.replace("**", "")
            for human, assistant in history[1:]:
                history_transformer_format.append({"role": "user", "content": human })
                history_transformer_format.append({"role": "assistant", "content":assistant})
            history_transformer_format.append({"role": "user", "content": question_prompt.format(message, msg_context)})
            stop = StopOnTokens()

            messages = history_transformer_format

            model_inputs = tokenizer.apply_chat_template(
                    [messages],
                    tokenize = True,
                    add_generation_prompt = True, # Must add for generation
                    return_dict = True,
                    return_tensors = "pt",
            ).to(device)
            
            streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
            generate_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=128)
            t = Thread(target=model.generate, kwargs=generate_kwargs)
            t.start()

            partial_message = ""
            for new_token in streamer:
                if new_token != '<':
                    partial_message += new_token
                    yield partial_message

        gr.ChatInterface(predict).launch(inline=False, share=True)
    