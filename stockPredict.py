import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def processData():
    file_path = '/Users/joshuawang/Downloads/MLProj/MYNN/stockPredictionzz/aal_stock_5yr.csv'
    try:
        aal_data = pd.read_csv(file_path)
    except FileNotFoundError:
        data = pd.read_csv('/Users/joshuawang/Downloads/MLProj/MYNN/stockPredictionzz/all_stocks_5yr.csv', delimiter=',', on_bad_lines='skip')
        aal_data = data[data['Name'] == 'AAL']
        aal_data.to_csv('/Users/joshuawang/Downloads/MLProj/MYNN/stockPredictionzz/aal_stock_5yr.csv', index=False)
    
    return aal_data

def split_dataset_indices(total_length, train_ratio):
    """Calculate indices for train-validation split."""
    train_size = int(total_length * train_ratio)
    train_indices = np.arange(train_size)
    val_indices = np.arange(train_size, total_length)
    return train_indices, val_indices

def prepare_dataset(data, sequence_length=60, batch_size=32, train_ratio=0.85):
    """
    Convert stock data into a TensorFlow dataset for training.
    sequence_length: Number of past days to use for predicting the next day.
    batch_size: Number of sequences in each batch.
    """
    # Use 'Close' price for prediction (you can modify to include other features)
    close_prices = data['close'].values
    
    # Create sequences of sequence_length days
    X, y = [], []
    for i in range(len(close_prices) - sequence_length):
        X.append(close_prices[i:i + sequence_length])
        y.append(close_prices[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Normalize the data (important for neural networks)
    X_mean = X.mean()
    X_std = X.std()
    X = (X - X_mean) / X_std
    y = (y - X_mean) / X_std
    
    train_indices, val_indices = split_dataset_indices(len(X), train_ratio)
    # Convert to TensorFlow dataset (dataset that will be used to train the model)
    train_dataset = tf.data.Dataset.from_tensor_slices((X[train_indices], y[train_indices]))
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size)

    #dataset that will be used to validate the accuracy of the model (dataset used for testing)
    val_dataset = tf.data.Dataset.from_tensor_slices((X[val_indices], y[val_indices]))
    val_dataset = val_dataset.batch(batch_size)
    
    return train_dataset, val_dataset, X_mean, X_std

def plot_results(history, predictions, actual, X_mean, X_std):
    """Plot training/validation loss and predicted vs. actual prices."""
    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    # unnormalize predictions and actual values
    predictions = predictions * X_std + X_mean
    actual = actual * X_std + X_mean
    
    # Plot predicted vs. actual prices
    plt.subplot(1, 2, 2)
    plt.plot(actual, label='Actual Prices', color='blue')
    plt.plot(predictions, label='Predicted Prices', color='orange', linestyle='--')
    plt.title('Predicted vs. Actual Stock Prices')
    plt.xlabel('Time Step')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Load and process the data
    data = processData()
    
    # Prepare the TensorFlow dataset
    (train_dataset, val_dataset, X_mean, X_std) = prepare_dataset(data)
    
    # Print dataset info
    print("Dataset created with batches of sequences.")
    for X_batch, y_batch in train_dataset.take(1):
        print(f"Training batch - X shape: {X_batch.shape}, y shape: {y_batch.shape}")
    for X_batch, y_batch in val_dataset.take(1):
        print(f"Validate batch - X shape: {X_batch.shape}, y shape: {y_batch.shape}")

    model = models.Sequential([
        layers.Input(shape=(60, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

    #compiles the model -> optimizes, specifices how to calc loss
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    #generates an overview of the model, details of each layer, etc
    model.summary()
    

    #model.fit(train_dataset, epochs=5, verbose=1)
    #uses "new data" for testing -> old method would take samples of data used in the training, this method splits up training and testing data
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=5, verbose=1)

    #next step... make predictions using val_dataset
    #visualize using mathplotlib.pyplot
    
    val_X, val_y = [], []
    for X_batch, y_batch in val_dataset:
        val_X.append(X_batch.numpy())
        val_y.append(y_batch.numpy())
    val_X = np.concatenate(val_X, axis=0)
    val_y = np.concatenate(val_y, axis=0)

    predictions = model.predict(val_X)
        
    # Plot results
    plot_results(history, predictions.flatten(), val_y, X_mean, X_std)

if __name__ == "__main__":
    main()
