import pandas as pd
import numpy as np
from ImportData import update


# ---- IMPORT DATA ----
# Read CSV
df = pd.read_csv("./RawData/Sample_Spreadsheet.csv")
# Create Arrays
Sequence = df.iloc[:, 0].values
Structure = df.iloc[:, 1].values
NTerminal = df.iloc[:, 2].values
CTerminal = df.iloc[:, 3].values
NonTerminal = df.iloc[:, 4].values




