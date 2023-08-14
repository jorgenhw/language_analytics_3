# System imports
import string # for removing punctuation
import pickle # for saving tokenizer
import os
import re

# Tensorflow imports
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku 

# Other imports
import pandas as pd # 
import numpy as np #
from tqdm import tqdm # for progress bar

"""
########### Loading data ###########
"""
# Function to load all comments from a group of csv files in the data folder. Outputs a list of comments.
def load_data(data_dir):
    comments = []
    for filename in tqdm(os.listdir(data_dir)):
        article_df = pd.read_csv(os.path.join(data_dir, filename), dtype=str, delimiter=',')
        comments.extend(list(article_df["commentBody"].values))
    return comments


"""
########### Preprocessing data ###########
"""
def preprocess_data(comments, encoding="utf8"):
    for i in tqdm(range(len(comments))):
        # checking if the comment is a string
        if isinstance(comments[i], str): # If the comment is not a string, the function will skip the preprocessing steps and move on to the next comment.
            # encoding as utf8 and removing non-ascii characters
            comments[i] = comments[i].encode(encoding).decode("ascii",'ignore') # encoding as utf8 and removing non-ascii characters
            # removing symbols
            comments[i] = comments[i].translate(str.maketrans('', '', string.punctuation))
            # removing numbers
            comments[i] = re.sub(r'\d+', '', comments[i])
            # lowercasing
            comments[i] = comments[i].lower()
            return comments

"""
########### Tokenzing comments ###########
"""
# Function to tokenize comments using tensorflow tokenizer
def tokenize_comments(comments):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(comments)
    total_words = len(tokenizer.word_index) + 1
    return tokenizer, total_words

# then use the ```get_sequence_of_tokens()``` function we defined above, which turns every text into a sequence of tokens based on the vocabulary from the tokenizer.
def get_sequence_of_tokens(tokenizer, comments):
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in tqdm(comments):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences

# Then we pad our input sequences to make them all the same length.
def generate_padded_sequences(input_sequences, total_words):
    # get the length of the longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])
    # make every sequence the length of the longest on
    input_sequences = np.array(pad_sequences(input_sequences, 
                                            maxlen=max_sequence_len, 
                                            padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, 
                            num_classes=total_words)
    return predictors, label, max_sequence_len
