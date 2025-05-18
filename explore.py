import pandas as pd
from nltk import SpaceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


def count_labels(path):
    df = pd.read_csv(path)
    val_counts = df["Label"].value_counts()
    print("Value counts: ", val_counts)
    return val_counts


def explore_tokens():
    pass


if __name__ == "__main__":
    count_labels("./data/train/SemEval2018-T3-train-taskA_emoji.csv")
    count_labels("./data/train/SemEval2018-T3-train-taskB_emoji.csv")
