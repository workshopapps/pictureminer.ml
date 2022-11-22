from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

class Captioner_Generator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def predict(self, image):
        image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values.to(self.device)

        outputs = self.model.generate(pixel_values, max_length = 16, num_beams = 4)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


captioner_generator = Captioner_Generator("model")
