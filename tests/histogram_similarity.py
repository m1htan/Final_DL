import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("../logs/evaluated/eval_report.csv")
plt.hist(df["cosine"], bins=20, alpha=0.6, color="blue", label="Cosine")
plt.hist(df["f1"], bins=20, alpha=0.6, color="orange", label="F1")
plt.axvline(0.9, color="red", linestyle="--", label="Threshold 0.9")
plt.legend()
plt.xlabel("Similarity")
plt.ylabel("Count")
plt.title("Distribution of Similarity Scores")
plt.show()
