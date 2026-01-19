import torch.nn as nn
from transformers import AutoTokenizer, CLIPModel


class TextEncoder(nn.Module):
    """
    Text Encoder train for ETHICS
    Config setting:
        name (openai model name), max_length
    Inputs:
        text (batch, text)
    Returns:
        text_features
    Ourputs:
        None
    """

    def __init__(self, name=None, max_length=None):
        super().__init__()
        self.max_length = max_length
        if name is None:
            self.name = "openai/clip-vit-base-patch32"
        else:
            self.name = name

        self.model = CLIPModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, clean_up_tokenization_spaces=True
        )
        self.projection_dim = self.model.projection_dim

    def forward(self, x):
        inputs = self.tokenizer(
            x,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        ).to(self.model.device)
        text_features = self.model.get_text_features(**inputs)

        return text_features
