import pandas as pd
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = ""

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "shazadudwadia/supermarket",
  file_path,
)

print("First 5 records:", df.head())