import csv
import os
import re
from pathlib import Path

import emoji
import pandas as pd


def txt_to_csv(f, delete_old=True):

    csv_path = f.with_suffix(".csv")

    if f.is_file() and not f.name.endswith("csv"):

        with open(f, "r", newline="", encoding="utf-8") as txt_file:
            lines = csv.reader(txt_file, delimiter="\t")

            with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                for line in lines:
                    writer.writerow(line)

        if delete_old:
            os.remove(f)

    return csv_path


def clean_text(item: str):

    tag_pattern = r"@\w{1,15}"
    link_pattern = r"http[s]?://\S+"

    data = re.sub(pattern=tag_pattern, repl="<USER>", string=item)
    data = re.sub(pattern=link_pattern, repl="<URL>", string=data)
    return data.strip()


def main():

    TRAIN_PATH = "data/train/"
    for f in Path(TRAIN_PATH).iterdir():
        path = txt_to_csv(f, delete_old=True)
        df = pd.read_csv(path)
        print(df.columns, "\n", df.head())
        df["Tweet text"] = df["Tweet text"].apply(clean_text)

        if "emoji" not in f.name:
            df["Tweet text"] = df["Tweet text"].apply(
                lambda x: emoji.replace_emoji(x, "")
            )


if __name__ == "__main__":
    main()
