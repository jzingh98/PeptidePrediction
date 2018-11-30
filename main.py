import pandas as pd
import numpy as np
import  string
from ImportData import update
import sklearn as sk
from string import ascii_lowercase

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
alphabet_Sequence = np.array([])
a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = np.zeros((26, len(ar_Sequence)))
count = 0
for Line in ar_Sequence:
    # Save sequence in a dataframe
    current_Sequence = []
    for Letter in Line:
        current_Sequence = np.append(current_Sequence, [ord(Letter)])
    encoded_dfSequence = encoded_dfSequence.append(pd.Series(current_Sequence), ignore_index=True)
    # Increment appropriate alphabet column
    a[count] = (Line.lower().count('a'))
    b[count] = (Line.lower().count('b'))
    c[count] = (Line.lower().count('c'))
    d[count] = (Line.lower().count('d'))
    e[count] = (Line.lower().count('e'))
    f[count] = (Line.lower().count('f'))
    g[count] = (Line.lower().count('g'))
    h[count] = (Line.lower().count('h'))
    i[count] = (Line.lower().count('i'))
    j[count] = (Line.lower().count('j'))
    k[count] = (Line.lower().count('k'))
    l[count] = (Line.lower().count('l'))
    m[count] = (Line.lower().count('m'))
    n[count] = (Line.lower().count('n'))
    o[count] = (Line.lower().count('o'))
    p[count] = (Line.lower().count('p'))
    q[count] = (Line.lower().count('q'))
    r[count] = (Line.lower().count('r'))
    s[count] = (Line.lower().count('s'))
    t[count] = (Line.lower().count('t'))
    u[count] = (Line.lower().count('u'))
    v[count] = (Line.lower().count('v'))
    w[count] = (Line.lower().count('w'))
    x[count] = (Line.lower().count('x'))
    y[count] = (Line.lower().count('y'))
    z[count] = (Line.lower().count('z'))
    count += 1
alphabet_Sequence = np.vstack((a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z))


# Encoding of alphabet arrays
# a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = np.array([])
# A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z = np.array([])
# alphabet_dfSequence = pd.DataFrame(columns=('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
#                                             'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'))

