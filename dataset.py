import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

print(torch.__version__)
print("Using GPU:", torch.cuda.is_available())
from typing import Literal

from sentence_transformers import SentenceTransformer
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from transformers.models.bert import BertModel, BertTokenizer


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
        self.data = data["Tweet text"].to_list()
        self.labels = data["Label"].to_list()
        self.device = device
        self.batch_size = batch_size

        self.embeddings = self.embed_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.embeddings[idx]

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
        model.to(device)
        model.eval()  # Modalità di valutazione

        embeddings = []

        # Batching data
        for i in range(0, len(self.data), self.batch_size):

            input = tokenizer(
                self.data[i : i + self.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            # Inference: otteniamo l'output
            with torch.no_grad():
                outputs = model(**input)
                last_hidden_state = (
                    outputs.last_hidden_state
                )  # Shape: (1, seq_len, hidden_dim)

            # Il token [CLS] è sempre il primo token → indice 0
            cls_embedding = last_hidden_state[:, 0, :]  # Shape: (1, hidden_dim)

            # Convertiamo in vettore numpy se serve
            embedding_vector = cls_embedding.squeeze()
            embeddings.append(embedding_vector)

        return torch.cat(embeddings)

    def _bertavg_embed(self):

        model_name = "bert-base-uncased"
        device = torch.device(self.device)

        tokenizer = BertTokenizer.from_pretrained(model_name)

        model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        model.to(device)

        model.eval()  # Modalità di valutazione

        embeddings = []

        for i in range(0, len(self.data), self.batch_size):

            input = tokenizer(
                self.data[i : i + self.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            # Inference: otteniamo hidden_states (una lista di 13 tensori: embedding iniziale + 12 layer)
            with torch.no_grad():
                outputs = model(**input)
                hidden_states = (
                    outputs.hidden_states
                )  # Tuple: [layer_0, layer_1, ..., layer_12]

            # Prendiamo gli ultimi 4 layer: layer -4, -3, -2, -1 (cioè 9,10,11,12)
            last_4_layers = hidden_states[
                -4:
            ]  # Lista di 4 tensori: shape (1, seq_len, hidden_dim)

            # Stack e somma/media
            # Stack per ottenere un tensore shape (4, 1, seq_len, hidden_dim)
            stacked_layers = torch.stack(last_4_layers)

            # Facciamo la media dei 4 layer → shape diventa (1, seq_len, hidden_dim)
            mean_last_4 = torch.mean(stacked_layers, dim=0)

            # Ora media su tutti i token della frase (asse 1 → seq_len)
            # Otteniamo il sentence embedding finale: shape (1, hidden_dim)
            sentence_embedding = torch.mean(mean_last_4, dim=1)

            # Convertiamo in vettore numpy se necessario
            embedding_vector = sentence_embedding.squeeze()

            embeddings.append(embedding_vector)

        return torch.cat(embeddings)

    def _sonar_embed(self):

        device = torch.device(self.device)
        t2vec_model = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
        )

        embedding = t2vec_model.predict(
            self.data, source_lang="eng_Latn", batch_size=32, target_device=device
        )

        return embedding

    def _instructor_embed(self):

        model = SentenceTransformer("hkunlp/instructor-large").to(self.device)
        embeddings = model.encode(
            sentences=self.data,
            device=self.device,
            batch_size=32,
            show_progress_bar=True,
            prompt="Represent this tweet for classification: ",
        )

        return embeddings


if __name__ == "__main__":

    MODELS = ["bert-cls", "bert-avg", "instructor", "sonar"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv("data/train/SemEval2018-T3-train-taskA.csv")
    print(len(df))

    for model in MODELS:
        ds = IronyDetectionDataset(df, embedding_model=model, device=DEVICE)
        print(ds[10].shape)
        dl = DataLoader(dataset=ds, batch_size=64, pin_memory=True)
