import torch.nn as nn
from transformers import AutoProcessor, CLIPModel, CLIPProcessor


class ImageEncoder(nn.Module):
    """
    Image Encoder inference for images
    Config setting:
        name (openai model name), max_length
    Inputs:
        img
    Returns:
        image_features
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

        if name == "openai/clip-vit-large-patch14":
            self.processor = CLIPProcessor.from_pretrained(
                name, clean_up_tokenization_spaces=True
            )
        else:
            self.processor = AutoProcessor.from_pretrained(
                name, clean_up_tokenization_spaces=True
            )

        self.projection_dim = self.model.projection_dim

    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt").to(self.model.device)
        image_features = self.model.get_image_features(**inputs)
        return image_features
