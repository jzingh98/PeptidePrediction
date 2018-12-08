# Import Packages
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# ---- IMPORT DATA ----

# Read CSV
df = pd.read_csv("./RawData/Sample_Spreadsheet.csv")

# Create arrays to hold each column
ar_Sequence = df.iloc[:, 0].values
ar_NTerminal = df.iloc[:, 1].values
ar_CTerminal = df.iloc[:, 2].values
ar_Structure = df.iloc[:, 3].values
ar_CombinedX = df.iloc[:, 0:3].values

# Encode the categorical columns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
encoded_Structure = labelEncoder.fit_transform(ar_Structure)
encoded_NTerminal = labelEncoder.fit_transform(ar_NTerminal)
encoded_CTerminal = labelEncoder.fit_transform(ar_CTerminal)

# Encoding peptide sequence
encoded_dfSequence = pd.DataFrame()     # Not used yet
alphabet_Sequence = np.array([])        # Used
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


# Combine columns into single nd array
combined_YX = np.vstack((encoded_Structure, alphabet_Sequence, encoded_NTerminal, encoded_CTerminal)).transpose()
np.random.shuffle(combined_YX)
Y = combined_YX[:, 0]
X = combined_YX[:, 1:]
# Y_encoded = keras.utils.to_categorical(encoded_Structure, num_classes=12)    # Not used\


# Designate Train and Test Data
x_train = np.vsplit(X, ([300, 456]))[0]
x_test = np.vsplit(X, ([300, 456]))[1]
y_train = np.hsplit(Y, ([300, 456]))[0]
y_test = np.hsplit(Y, ([300, 456]))[1]
y_classnames, y_indices = np.unique(Y, return_inverse=True)
y_train_classnames, y_train_indices = np.unique(y_train, return_inverse=True)
y_test_classnames, y_test_indices = np.unique(y_test, return_inverse=True)


# Record data characteristics
print(X.shape)
print(x_train.shape)
print(x_test.shape)
print(Y.shape)
print(y_train.shape)
print(y_test.shape)
print(y_classnames.shape)

num_features = X.shape[1]
num_classes = y_classnames.shape[0]
print(num_features)
print(num_classes)

print(x_train)
print(y_train)


# Build Model
model = Sequential()
dense1 = model.add(Dense(3, activation='sigmoid', input_dim=num_features))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

# Configure Model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Training the ANN
model.fit(x_train, y_train, batch_size=10, epochs=20)

# Predicting the Test set results
y_pred = model.predict(x_test)


# Results
score = model.evaluate(x_test, y_test, batch_size=1)
print("\nScore: " + str(score))
