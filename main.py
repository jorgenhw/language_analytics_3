"""
This script loads data from the data folder, preprocesses it, tokenizes it, creates a model, trains it and generates text.
"""

import os
import src.preprocessing_data as pre
import src.creating_model as cm
import src.generate_text as gt
import argparse

# to import model 
from tensorflow import keras

def main(args):
    """
    ### Loading data ###
    """
    # Path to data folder
    print("Loading data...")
    data_dir = os.path.join("data")
    # Function to load all comments from a group of csv files in the data folder. Outputs a list of comments.
    comments = pre.load_data(data_dir)
    # Reducing the comments to 1000 for making the script run faster
    comments = comments[:20]
    print(f'Data loaded successfully and reduced to {len(comments)} comments.')

    """
    ### Preprocessing data ###
    ... encoding as utf8, removing symbols, numbers and lowercasing
    """
    # Preprocessing data (remove punctuation, numbers, empty comments)
    print("Preprocessing data...")
    comments = pre.preprocess_data(comments,encoding=args.encoding)
    

    """
    ### Tokenzing comments ###
    """
    print("Tokenizing comments...")
    # Function to tokenize comments using tensorflow tokenizer
    tokenizer, total_words = pre.tokenize_comments(comments)

    # Turns every text into a sequence of tokens based on the vocabulary from the tokenizer
    input_sequences = pre.get_sequence_of_tokens(tokenizer, comments)

    # Then we pad our input sequences to make them all the same length
    predictors, label, max_sequence_len = pre.generate_padded_sequences(input_sequences, total_words)

    """
    ### Creating and training model ###
    """
    # Creating model
    model = cm.create_model(max_sequence_len, total_words, args.embedding_dim, args.hidden_layer_size)

    # Fitting model and saving history
    history = cm.fit_model(model, predictors, label, args.epochs, args.batch_size, args.verbose)

    # Creating model folder if it does not exist
    cm.make_misc_folder()

    # saving plot of training history
    cm.save_training_plot(history)

    """
    ### Saving model ###
    """
    
    # Saving model
    print("Saving model...")
    # save model 
    cm.save_model(model, f'pre_trained_model_{max_sequence_len}.h5')
    print("Model saved successfully.")

    """
    ### Generating text ###
    """
    # load model 
    print("Loading model...")
    model = keras.models.load_model(os.path.join("misc", f'pre_trained_model_{max_sequence_len}.h5'))
    print("Model loaded successfully.")

    # Generate text
    generated_text = gt.generate_text(seed_text=args.seed_text, # seed text
                            next_words = args.next_words, # number of words to generate
                            model = model, # model to use for generation
                            tokenizer = tokenizer, # tokenizer used to train the model
                            max_sequence_len = max_sequence_len) # length of sequences used to train the model

    print(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script uses a RNN model to generate text. The model is trained on a comments dataset.')
    
    # Arguments for pre processing data
    parser.add_argument('--encoding', type=str, default="utf8", help='Encoding to use when loading data. Default is "utf8".')

    # Arguments for creating and training model
    parser.add_argument('--embedding_dim', type=int, default=10, help='Embedding dimension. Default is 10.')
    parser.add_argument('--hidden_layer_size', type=int, default=30, help='Hidden layer size. Default is 30.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs. Default is 10.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size. Default is 128.')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity mode. Default is 1.')

    # arguments for generating text
    parser.add_argument('--seed_text', type=str, default="I like the smell of fries because", help='Seed text to use for generating text. Default is "I love".')
    parser.add_argument('--next_words', type=int, default=5, help='Number of words to generate. Default is 10.')
    
    args = parser.parse_args()
    main(args)