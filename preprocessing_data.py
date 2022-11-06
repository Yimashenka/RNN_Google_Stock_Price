# IMPORTING THE LIBRAIRIES
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# IMPORTING THE TRAINING SET
dataset_train = pd.read_csv("data/Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values
# [:, 1:2] says we just want all the lines of the columns between 1 and 2, 2 excluded (Python)
# To make it a NumPy array, just need to add .values


# FEATURE SCALING
# We have the choice between two :
#   Standardisation
#   Normalisation

# Here we'll use Normalisation, because there is a sigmoïd function in the
# output layer of the RNN. In order to do that, we're going to use the
# MinMax K-load class from the preprocessing module of the Scikit library.

# We create the MinMaxScaler object
sc = MinMaxScaler(feature_range=(0, 1))

# We use sc to scale our training data set
training_set_scaled = sc.fit_transform(training_set)

# CREATING A DATA STRUCTURE WITH 60 TIMESTEPS AND 1 OUTPUT
# What these numbers mean ?
# "60 timesteps" means that at each time T, the RNN is going to look at the 60
# stockprices before time T, and based on these 60 stockprices, it will try to
# predict the "1 next output", the value at time T+1.

# In our dataset, 60 times correspond to three months, as we have 20 entries
# per month. So for each time T, we will look backward to three months.

# We will create two separate entities.
#   X_train, gonna be the input, containing the 60 previews stock prices
#   y_train, gonna be the output, containing the stock price the next financial day.

X_train = []
y_train = []

# As we need to look backward to the 60 last stock prices, we cannot start at
# the begining of our data set. So, we in,itialize our loop at 60, correspon-
# -ding at the 60th financial day of 2012 and end it at the latest index, but
# as upper bound is not taken in Python, we add one, corresponding at 1258.
for i in range(60, 1258):
  X_train.append(training_set_scaled[i-60:i, 0])    # Get the 60 previous
                                                    # stock prices.
  y_train.append(training_set_scaled[i, 0])         # Get the next true stock
                                                    # price value of the next
                                                    # financial day.

# At this time, X_train and y_train are lists, but we need them to be NumPy
# array to be accepted by our future RNN.
X_train, y_train = np.array(X_train), np.array(y_train)
#X_train.shape


# RESHAPING
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#print(X_train.shape)