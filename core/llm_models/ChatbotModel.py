from unsloth import FastLanguageModel

class ChatbotModel:
    """
    Class for loading Large Language Models loaded from Unsloth.

    Args:
        model_name (str): Name of the pre-trained model.
        hf_access_token (str): Hugging Face token required to access some LLM pre-trains that need authentication.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit precision. Defaults to False.
        max_seq_length (int, optional): Maximum sequence length for the input. Defaults to 2048.
        dtype (type, optional): Data type for model weights. Defaults to None.
        training (bool, optional): Whether the model is used for training. Defaults to False.

    Attributes:
        model (Any): Loaded model object.
        tokenizer (Any): Tokenizer model for input sequence.
    """

    def __init__(self,
                 model_name: str,
                 hf_access_token: str = None,
                 load_in_4bit: bool = False,
                 max_seq_length: int = 4096,
                 dtype: type = None,
                 training: bool = False):

        self.model_name = model_name
        self.hf_access_token = hf_access_token
        self.load_in_4bit = load_in_4bit
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.training = training

        self.model = None
        self.tokenizer = None

    def load_pretrained_model(self):
        """
        Load and return the pre-trained model.

        Returns:
            model (Any): Loaded model object.
            tokenizer (Any): Tokenizer model for input sequence.
        """
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=self.dtype,
                load_in_4bit=self.load_in_4bit,
                token=self.hf_access_token,
            )

            print(f"Loaded model '{self.model_name}' from Hugging Face.")
            return model, tokenizer
        except Exception as e:
            raise ValueError(f"Failed to load model '{self.model_name}' from Hugging Face. Error: {e}")

    def get_pretrained_model(self):
        """
        Get the loaded pretrained model and tokenizer.

        Returns:
            model (Any): Loaded model object.
            tokenizer (Any): Tokenizer model for input sequence.
        """
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = self.load_pretrained_model()
        return self.model, self.tokenizer

    def get_peft_model(self, r: int = 16, lora_alpha: int = 16,
                   target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj",
                                           "gate_proj", "up_proj", "down_proj"],
                   use_gradient_checkpointing: str = "unsloth",
                   lora_dropout: float = 0.05, bias: str = "none") -> tuple:
        """
        Apply Parameter-Efficient Fine-tuning (PEFT) to the model.

        Args:
            r (int): Number of clusters. Defaults to 16.
            lora_alpha (int): Alpha value for LoRA. Defaults to 16.
            target_modules (list): Modules to apply PEFT. Defaults to ["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"].
            use_gradient_checkpointing (str): Gradient checkpointing method. Defaults to "unsloth".
            lora_dropout (float): Dropout value for LoRA. Defaults to 0.05.
            bias (str): Bias option for LoRA. Defaults to "none".

        Returns:
            tuple: Tuple containing the modified model and tokenizer.
        """

        model, tokenizer = self.get_pretrained_model()

        model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        return model, tokenizer
