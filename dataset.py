import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

print(torch.__version__)
print("Using GPU:", torch.cuda.is_available())
from typing import Literal

from sentence_transformers import SentenceTransformer
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from transformers.models.bert import BertModel, BertTokenizer

import preprocessing


class IronyDetectionDataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        embedding_model: Literal[
            "instructor", "bert-cls", "bert-avg", "sonar"
        ] = "bert-cls",
        device: str = "cpu",
        batch_size: int = 32,
    ):

        self.emb_model = embedding_model

        data["Tweet text"] = data["Tweet text"].apply(preprocessing.clean_text)
        self.data = data["Tweet text"].to_list()
        labels = data["Label"].to_list()
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.device = device
        self.batch_size = batch_size
        self.embeddings = self.embed_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.embeddings[idx], self.labels[idx]

    def embed_data(self):

        print(f"Embedding using {self.emb_model}")

        match self.emb_model:
            case "instructor":
                return self._instructor_embed()
            case "bert-cls":
                return self._bertcls_embed()
            case "bert-avg":
                return self._bertavg_embed()
            case "sonar":
                return self._sonar_embed()
            case _:
                raise ValueError(
                    "Invalid embedding model. Please choose one among 'instructor', 'bert-cls', 'bert-avg', 'sonar'."
                )

    def _bertcls_embed(self):

        model_name = "bert-base-uncased"
        device = torch.device(self.device)
        tokenizer = BertTokenizer.from_pretrained(model_name)

        model = BertModel.from_pretrained(model_name)
        model.to(device)  # pyright: ignore
        model.eval()

        embeddings = []

        # Batching data
        for i in range(0, len(self.data), self.batch_size):

            input = tokenizer(
                self.data[i : i + self.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            # Inference
            with torch.no_grad():
                outputs = model(**input)
                last_hidden_state = (
                    outputs.last_hidden_state
                )  # Shape: (1, seq_len, hidden_dim)

            cls_embedding = last_hidden_state[:, 0, :]  # Shape: (1, hidden_dim)

            # Convertiamo in vettore numpy se serve
            embeddings.append(cls_embedding)

        return torch.cat(embeddings, dim=0)

    def _bertavg_embed(self):

        model_name = "bert-base-uncased"
        device = torch.device(self.device)

        tokenizer = BertTokenizer.from_pretrained(model_name)

        model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        model.to(device)  # pyright: ignore

        model.eval()

        embeddings = []

        for i in range(0, len(self.data), self.batch_size):

            input = tokenizer(
                self.data[i : i + self.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            # Inference:
            with torch.no_grad():
                outputs = model(**input)
                hidden_states = (
                    outputs.hidden_states
                )  # Tuple: [layer_0, layer_1, ..., layer_12]

            # Getting last 4 layers: layer -4, -3, -2, -1
            last_4_layers = hidden_states[
                -4:
            ]  # Lista di 4 tensori: shape (1, seq_len, hidden_dim)

            # Stack (4, 1, seq_len, hidden_dim)
            stacked_layers = torch.stack(last_4_layers)

            # Averaging over the last 4 layers (1, seq_len, hidden_dim)
            mean_last_4 = torch.mean(stacked_layers, dim=0)

            # Averaging over all tokens to get sentence embedding (1, hidden_dim)
            sentence_embedding = torch.mean(mean_last_4, dim=1)

            embeddings.append(sentence_embedding)

        return torch.cat(embeddings, dim=0)

    def _sonar_embed(self):

        device = torch.device(self.device)
        t2vec_model = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
        )

        np_embedding = t2vec_model.predict(
            self.data,
            source_lang="eng_Latn",
            batch_size=self.batch_size,
        )
        embedding = torch.from_numpy(np_embedding).to(torch.float32).cpu()

        return embedding

    def _instructor_embed(self):

        model = SentenceTransformer("hkunlp/instructor-large").to(self.device)
        embeddings = model.encode(
            sentences=self.data,
            batch_size=self.batch_size,
            show_progress_bar=True,
            prompt="Represent this tweet for classification: ",
            convert_to_tensor=True,
        )

        return embeddings


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
                pass  # Initialize sonar model separately if needed or handle in __call__
            case _:
                raise ValueError(
                    "Invalid embedding model. Please choose one among 'instructor', 'bert-cls', 'bert-avg', 'sonar'."
                )

    def __call__(self, batch):
        """
        Collate function that takes a batch of (text, label) tuples,
        embeds the text, and returns a batch of (embedding, label) tensors.
        """
        # batch is a list of tuples: [(text1, label1), (text2, label2), ...]

        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]  # Adjust dtype if labels are not long

        embeddings = self._embed_batch(texts)

        return embeddings, labels

    def _embed_batch(self, texts):
        """Helper method to perform embedding for a batch of texts."""

        with torch.no_grad():

            match self.emb_model:

                case "bert-cls":
                    inputs = self.tokenizer( # pyright:ignore
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(self.device)
                    outputs = self.model(**inputs) # pyright:ignore
                    # Get CLS token embedding
                    embeddings = outputs.last_hidden_state[:, 0, :]

                case "bert-avg":
                    inputs = self.tokenizer( # pyright:ignore
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(self.device)
                    outputs = self.model(**inputs, output_hidden_states=True) # pyright:ignore
                    hidden_states = outputs.hidden_states
                    # Getting last 4 layers and averaging
                    last_4_layers = hidden_states[-4:]
                    stacked_layers = torch.stack(last_4_layers)
                    mean_last_4 = torch.mean(stacked_layers, dim=0)
                    # Averaging over all tokens
                    embeddings = torch.mean(mean_last_4, dim=1)

                case "instructor":
                    # SentenceTransformer handles batching and tensor conversion internally
                    embeddings = self.model.encode( # pyright:ignore
                        sentences=texts,
                        batch_size=len(texts),  # Encode the current batch size
                        show_progress_bar=True,  # No progress bar during collation
                        prompt="Represent this tweet for classification: ",
                        convert_to_tensor=True,
                        device=self.device.type,
                    )
                    # Ensure tensor is on the correct device if SentenceTransformer returns CPU tensor

                case "sonar":
                    device = torch.device(self.device)
                    t2vec_model = TextToEmbeddingModelPipeline(
                        encoder="text_sonar_basic_encoder",
                        tokenizer="text_sonar_basic_encoder",
                        device=device,
                    )

                    np_embedding = t2vec_model.predict(
                        texts,
                        source_lang="eng_Latn",
                        batch_size=len(texts),
                    )
                    embeddings = torch.from_numpy(np_embedding).to(torch.float32)

                case _:
                    raise ValueError(
                        "Invalid embedding model."
                    )  # Should not happen if initialized correctly

        return embeddings.cpu()
          # Move embeddings to cpu, the dataloader is supposed handle the allocation of data on correct device


def test():
    MODELS = ["bert-cls", "bert-avg", "instructor", "sonar"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv("data/train/SemEval2018-T3-train-taskA.csv")
    print(len(df))

    for model in MODELS:
        ds = IronyDetectionDataset(
            df, embedding_model=model, device=DEVICE  # pyright:ignore
        )
        print(ds[0][0].shape)
        dl = DataLoader(dataset=ds, batch_size=64, pin_memory=True)


if __name__ == "__main__":
    test()
