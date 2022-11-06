# IMPORTING THE DEPENDENCIES
from rnn_compiling_training import regressor
from preprocessing_data import pd, dataset_train, sc, np
import matplotlib.pyplot as plt

# GETTING THE REAL DATA (FROM 2017)
dataset_test = pd.read_csv("data/Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

# GETTING THE PREDICTED STOCK PRICE OF 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
  X_test.append(inputs[i-60:i, 0])

X_test= np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# VISUALISING THE RESULT
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
