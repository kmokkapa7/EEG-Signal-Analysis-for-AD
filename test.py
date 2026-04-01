import pandas as pd

df = pd.read_csv("dataset/participants.tsv", sep="\t")
print(df.columns)
print(df.head())
