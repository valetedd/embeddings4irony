import pandas as pd
import torch

torch.manual_seed(42)
from torch.utils.data import DataLoader, Dataset

print("Using GPU:", torch.cuda.is_available())
try:
    torch.cuda.manual_seed(42)
except:
    print("CUDA not connected")
from typing import List, Literal

from sentence_transformers import SentenceTransformer
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from transformers.models.bert import BertModel, BertTokenizer

from preprocessing import clean_text


class IronyDetectionDataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
    ):

        data["Tweet text"] = data["Tweet text"].apply(clean_text)
        self.data = data["Tweet text"].to_list()
        self.labels = [float(lab) for lab in data["Label"].to_list()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        text = self.data[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(dim=0)
        return text, label


class EmbeddingCollate:

    def __init__(
        self,
        embedding_model: Literal[
            "instructor", "bert-cls", "bert-avg", "sonar"
        ] = "bert-cls",
        device: str = "cpu",
    ):
        self.emb_model = embedding_model
        self.device = torch.device(device)
        self.tokenizer = None
        self.model = None

        # Initialize tokenizer and model based on embedding_model
        print(f"Initializing embedding model {self.emb_model} on device {self.device}")
        match self.emb_model:

            case "bert-cls" | "bert-avg":
                model_name = "bert-base-uncased"
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
                self.model = BertModel.from_pretrained(model_name)
                self.model.to(self.device)  # pyright:ignore
                self.model.eval()

            case "instructor":
                self.model = SentenceTransformer("hkunlp/instructor-large").to(
                    self.device
                )
                self.model.eval()

            case "sonar":
                self.model = TextToEmbeddingModelPipeline(
                    encoder="text_sonar_basic_encoder",
                    tokenizer="text_sonar_basic_encoder",
                    device=self.device,
                    dtype=torch.float32,
                )
                self.model.eval()

            case _:
                raise ValueError(
                    "Invalid embedding model. Please choose one among 'instructor', 'bert-cls', 'bert-avg', 'sonar'."
                )

    def __call__(self, batch: tuple[List[str]]):
        """
        Collate function that takes a batch of (text, label) tuples,
        embeds the text, and returns a batch of (embedding, label) tensors.
        """

        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        embeddings = self._embed_batch(texts)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(dim=1)

        return embeddings, labels_tensor

    def _embed_batch(self, texts: List[str]):
        """Helper method to perform embedding for a batch of texts."""

        with torch.no_grad():

            match self.emb_model:

                case "bert-cls":
                    inputs = self.tokenizer(  # pyright:ignore
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(self.device)
                    outputs = self.model(**inputs)  # pyright:ignore
                    # Get CLS token embedding
                    embeddings = outputs.last_hidden_state[:, 0, :]

                case "bert-avg":
                    inputs = self.tokenizer(  # pyright:ignore
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(self.device)
                    outputs = self.model(
                        **inputs, output_hidden_states=True
                    )  # pyright:ignore
                    hidden_states = outputs.hidden_states
                    # Getting last 4 layers and averaging
                    last_4_layers = hidden_states[-4:]
                    stacked_layers = torch.stack(last_4_layers)
                    mean_last_4 = torch.mean(stacked_layers, dim=0)
                    # Averaging over all tokens
                    embeddings = torch.mean(mean_last_4, dim=1)

                case "instructor":

                    embeddings = self.model.encode(  # pyright:ignore
                        sentences=texts,
                        batch_size=len(texts),
                        prompt="Represent the tweet for irony classification: ",
                        convert_to_tensor=True,
                        device=self.device.type,
                    )

                case "sonar":

                    embeddings = self.model.predict(  # pyright:ignore
                        texts,
                        source_lang="eng_Latn",
                        batch_size=len(texts),
                    ).clone()

                case _:
                    raise ValueError(
                        "Invalid embedding model."
                    )  # Should not happen if initialized correctly

        return embeddings


def test():
    print(torch.__version__)
    MODELS = ["bert-cls", "bert-avg", "instructor", "sonar"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv("data/train/SemEval2018-T3-train-taskA.csv")
    print(len(df))

    for model in MODELS:
        ds = IronyDetectionDataset(
            df, embedding_model=model, device=DEVICE  # pyright:ignore
        )
        collate_fn = EmbeddingCollate(
            embedding_model=model, device=DEVICE  # pyright:ignore
        )  # pyright:ignore
        dataloader = DataLoader(
            ds, batch_size=32, collate_fn=collate_fn  # pyright:ignore
        )
        dp = next(iter(dataloader))
        print(dp[0].size())


if __name__ == "__main__":
    test()
