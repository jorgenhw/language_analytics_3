from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import os
import matplotlib.pyplot as plt

def create_model(max_sequence_len, total_words, 
                 embedding_dim=10,
                 hidden_layer_size=30):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 
                        embedding_dim, # embedding dimension (10)
                        input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(hidden_layer_size))
    model.add(Dropout(0.1))
    
    # Add Output Layer
    model.add(Dense(total_words, 
                    activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    
    return model

# Fitting model to data and saving history
def fit_model(model, predictors, label, epochs=10, batch_size=128, verbose=1):
    history = model.fit(predictors, 
                    label, 
                    epochs=epochs,
                    batch_size=batch_size, 
                    verbose=verbose)
    return history

# Make a folder for the model if it does not exist
def make_misc_folder():
    if not os.path.exists("misc"):
        os.makedirs("misc")
        print("Misc folder created.")
    else:
        print("Misc folder already exists.")

# Saving plot of training history
def save_training_plot(history):
    plt.plot(history.history['loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss over epochs")
    plt.savefig(os.path.join("misc", "training_plot.png"))

# Saving model
def save_model(model, model_name):
    model.save(os.path.join("misc", model_name))

# Loading model
def load_model(load_path):
    if os.path.exists(load_path): # 
        model = load_model(load_path)
        print("Model loaded successfully.")
        return model
    else:
        print("Model file does not exist.")
        return None