from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, BertTokenizer
import torch
from PIL import Image
import numpy as np
from huggingface_hub import from_pretrained_keras
import tensorflow as tf



class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data."""
    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=32,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=128,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

class Captioner_Generator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self.text_model = from_pretrained_keras(f"{self.model_name}/similarity-model")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = ["no", "yes", "maybe"]

    
    def predict(self, image):
        image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values.to(self.device)

        outputs = self.model.generate(pixel_values, max_length = 16, num_beams = 4)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    def check_similarity(self, sentence1, sentence2):
        sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
        test_data = BertSemanticDataGenerator(
            sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
        )
        probs = self.text_model.predict(test_data[0])[0]

        labels_probs = {self.labels[i]: float(probs[i]) for i, _ in enumerate(self.labels)}
        return max(labels_probs, key=labels_probs.get)


captioner_generator = Captioner_Generator("model")
