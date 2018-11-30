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
from string import ascii_lowercase
encoded_dfSequence = pd.DataFrame()
alphabet_dfSequence = pd.DataFrame(columns=('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                                            'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'))
encoded_Sequence = np.array([])
current_Sequence = np.array([])
for Line in ar_Sequence:
    # Save sequence in a dataframe
    current_Sequence = []
    for Letter in Line:
        current_Sequence = np.append(current_Sequence, [ord(Letter)])
    encoded_dfSequence = encoded_dfSequence.append(pd.Series(current_Sequence), ignore_index=True)
    # Increment appropriate alphabet column
    for c in ascii_lowercase:
        alphabet_dfSequence[c][len(alphabet_dfSequence[c])] = 0
    print(Line)

print(alphabet_dfSequence.head())

# Encoding of alphabet arrays
# a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = np.array([])
# A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z = np.array([])
