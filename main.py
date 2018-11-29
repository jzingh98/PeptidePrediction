import pandas as pd
import numpy as np
from ImportData import update
import sklearn as sk

# ---- IMPORT DATA ----

# Read CSV
df = pd.read_csv("./RawData/Sample_Spreadsheet.csv")

# Create Arrays
ar_Sequence = df.iloc[:, 0].values
ar_Structure = df.iloc[:, 1].values
ar_NTerminal = df.iloc[:, 2].values
ar_CTerminal = df.iloc[:, 3].values
ar_NonTerminal = df.iloc[:, 4].values
ar_CombinedX = df.iloc[:, 2:5].values

# Encoding of structure and features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
encoded_Structure = labelEncoder.fit_transform(ar_Structure)
encoded_NTerminal = labelEncoder.fit_transform(ar_NTerminal)
encoded_CTerminal = labelEncoder.fit_transform(ar_CTerminal)
encoded_NonTerminal = labelEncoder.fit_transform(ar_NonTerminal)

# Encoding of peptide sequence
encoded_dfSequence = pd.DataFrame()
encoded_Sequence = np.array([])
current_Sequence = np.array([])
for C in ar_Sequence:
    current_Sequence = []
    for c in C:
        current_Sequence = np.append(current_Sequence, [ord(c)])
    encoded_dfSequence = encoded_dfSequence.append(pd.Series(current_Sequence), ignore_index=True)



