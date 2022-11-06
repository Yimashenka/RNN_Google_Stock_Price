# IMPORT THE DEPENDENCIES
from preprocessing_data import X_train, y_train
from rnn_building import regressor


# COMPILING THE RNN
# For the optimizer, we have a choice.

#   RMSprop : kind of advanced stochastic gradient descent optimizer, recom-
#   -mended for RNN
#   ADAM : works fine, we know it.

# For the loss, we are not doing classification anymore, so it's not gonna be
# cross entropy. This time we're dealing with regression problem because we
# have to predict a continuous value, and the loss for this kind of problem is
# the mean square error.

regressor.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# FITTING THE RNN TO THE TRAINING SET
regressor.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32
)