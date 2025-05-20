import csv
import os
import re
from pathlib import Path

import emoji
import pandas as pd


def clean_text(item: str):

    item = emoji.demojize(item, ("::", "::"))
    blacklist = [
        "#the",
        "#a",
        "#is",
        "#to",
        "#on",
        "#you",
        "#be",
        "#it",
        "#for",
        "#me",
        "#and",
        "#this",
        "#my",
        "#i",
        "#im",
        "#are",
        "#was",
        "#how",
        "#we",
        "#in",
        "#of",
        "#why",
        "#our",
        "#or",
        "#he",
    ]
    tag_pattern = r"@\w{1,15}"
    link_pattern = r"http[s]?://\S+"
    hashtag_pattern = r"#\w+"

    # Filtering hashtags based on blacklist
    hashtags = re.findall(pattern=hashtag_pattern, string=item)
    if hashtags:
        for h in hashtags:
            if h in blacklist:
                item = re.sub(pattern=hashtag_pattern, repl="<HASHTAG>", string=item)

    data = re.sub(pattern=tag_pattern, repl="<USER>", string=item)
    data = re.sub(pattern=link_pattern, repl="<URL>", string=data)
    return data.strip()


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


def main():

    PATH = "data/train/"
    for f in Path(PATH).iterdir():
        print(f)
        path = txt_to_csv(f, delete_old=True)
        df = pd.read_csv(path, delimiter=",", quotechar='"')
        df["Tweet text"] = df["Tweet text"].apply(clean_text)
        print(df.columns, "\n", df.head())


if __name__ == "__main__":
    main()
