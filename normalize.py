import numpy as np
from sklearn import preprocessing

X = np.array([[100.76,-546.98,945.00],
             [-56.31,184.32,43.70],
             [3.46,0.75,-8.85]])

normalized_X = preprocessing.normalize(X)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(normalized_X)
print(x_scaled)