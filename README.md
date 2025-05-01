This project implements a Long Short-Term Memory (LSTM) neural network to predict stock prices using historical data. The model is built using TensorFlow and processes American Airlines (AAL) stock data over a 5-year period.

The LSTM model consists of:

Input layer: Accepts sequences of 60 days of stock prices.
Two LSTM layers: 64 units (with return sequences) and 32 units.
Two Dense layers: 16 units (ReLU activation) and 1 unit (output).
Optimizer: Adam
Loss: Mean Squared Error (MSE)
Metric: Mean Absolute Error (MAE)

-The model uses a sequence length of 60 days to predict the next day's closing price.
-Data is normalized to improve training stability.
-The training dataset uses 85% of the data, with the remaining 15% for validation.
-Modify the sequence_length, batch_size, or train_ratio in prepare_dataset() to experiment with different configurations.
