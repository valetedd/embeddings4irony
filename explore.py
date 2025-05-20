import re
from collections import defaultdict

import pandas as pd


def count_labels(path):
    df = pd.read_csv(path)
    val_counts = df["Label"].value_counts()
    print("Value counts: ", val_counts)
    return val_counts


def hashtags_counter(data):

    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, list):
        df = pd.concat([pd.read_csv(f) for f in data])
    else:
        raise TypeError("data argument should either be a string or a list of strings")

    pattern = r"#\w+"
    hashtags = defaultdict(int)
    for _, tweet in df["Tweet text"].items():
        matched = re.findall(pattern, string=tweet)
        for h in matched:
            hashtags[h.strip()] += 1

    keys = []
    vals = []
    for k, v in hashtags.items():
        keys.append(k)
        vals.append(v)

    result = pd.DataFrame({"hashtag": keys, "count": vals})
    return result.sort_values(by=["count"], ascending=False)


if __name__ == "__main__":

    print("TaskA train labels")
    count_labels("./data/train/SemEval2018-T3-train-taskA_emoji.csv")
    print("\nTaskB train labels")
    count_labels("./data/train/SemEval2018-T3-train-taskB_emoji.csv")
    print("\nTaskA test labels")
    count_labels("./data/test/SemEval2018-T3_gold_test_taskA_emoji.csv")
    print("\nTaskB test labels")
    count_labels("./data/test/SemEval2018-T3_gold_test_taskB_emoji.csv")
    # data_sources = [
    #     "./data/train/SemEval2018-T3-train-taskB_emoji.csv",
    #     "./data/train/SemEval2018-T3-train-taskB_emoji.csv",
    # ]
    # df = hashtags_counter(data_sources)
    # print(df.head(10))
    # df.to_csv("data/soretd_hashtags.csv", index=False)
