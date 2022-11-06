# THE RECURRENT NEURAL NETWORK

# IMPORTING THE LIBRAIRIES
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from preprocessing_data import X_train

# INITILIASING THE RNN
# Call it regressor (classifier in ANN and CNN) because this time we're
# predicting a continuous output and we'll do regression.
# Remind : Regression is about predicting the continuous value, classification
# is about predicting a category, a class.

regressor = Sequential()

# ADDING THE FIRST LSTM LAYER AND SOME DROPOUT REGULARISATION
# Why add some dropout regularisation ? To prevent overfitting !

# About LSTM class :
#
#   units : we are going to stack many layers, we want high dimensionality.
#           We can increase the dimensionality by adding a large number of
#           neurons. Here choose 50. 50 neurons in this first LSTM layer.
#   return_sequencies : set true if we add another LSTM layer, the last one
#                       will have false (default value)
#   input_shape :   shape of the input, here in 3D, corresponding to the
#                   observations, the timesteps, and indicators.
regressor.add(
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(X_train.shape[1], 1)
    )
)

# We will ignore 20% of the neurons, a classical value to use.
regressor.add(Dropout(0.2))


# ADDING A SECOND LSTM LAYER AND SOME DROPOUT REGULARISATION
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(
    LSTM(
        units=50,
        return_sequences=True
    )
)
regressor.add(Dropout(0.2))


# ADDING A THIRD LSTM LAYER AND SOME DROPOUT REGULARISATION
regressor.add(
    LSTM(
        units=50,
        return_sequences=True
    )
)
regressor.add(Dropout(0.2))


# ADDING A FOURTH LSTM LAYER AND SOME DROPOUT REGULARISATION
regressor.add(
    LSTM(
        units=50
    )
)
regressor.add(Dropout(0.2))


# ADDING THE OUTPUT LAYER
# As we will predict one value in one dimension, units will be 1.
regressor.add(
    Dense(
        units=1
    )
)

