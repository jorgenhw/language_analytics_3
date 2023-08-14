from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=1)
        output_word = next((word for word, index in tokenizer.word_index.items() if index == predicted), "")
        seed_text += " " + output_word
    return seed_text