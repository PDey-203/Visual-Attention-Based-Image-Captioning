import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.layers import Dense


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        context_vector = attention_weights * features

        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector


caption_model = tf.keras.models.load_model(
    r"C:\Users\PRITAM\OneDrive\Desktop\caption_Model.keras",
    custom_objects={"BahdanauAttention": BahdanauAttention},
)

feature_extractor = tf.keras.models.load_model(
    r"C:\Users\PRITAM\OneDrive\Desktop\feature_extractor.keras"
)

with open(r"C:\Users\PRITAM\OneDrive\Desktop\tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


max_length = 37


def extract_features(filename):
    img = load_img(filename, target_size=(480, 480))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    features = feature_extractor.predict(img, verbose=0)
    features = features.reshape(1, -1, features.shape[-1])
    return features


def generate_caption_beam(model, photo, tokenizer, max_len, beam_size):
    start_token = tokenizer.word_index["startseq"]
    end_token = tokenizer.word_index["endseq"]

    # Each item = (current_sequence, score)
    sequences = [([start_token], 0.0)]

    # Generate words step by step
    for step in range(max_len):

        all_candidates = []

        for seq, score in sequences:

            # If caption already ended, keep it
            if seq[-1] == end_token:
                all_candidates.append((seq, score))
                continue

            # Pad sequence so model can read it
            padded = pad_sequences([seq], maxlen=max_len, padding="post")

            # Predict next word probabilities
            predictions = model.predict([photo, padded], verbose=0)[0]

            # Select top k predictions
            top_k_words = np.argsort(predictions)[-beam_size:]

            # Create new candidate sequences
            for word_id in top_k_words:
                prob = predictions[word_id]
                new_seq = seq + [word_id]
                new_score = score + np.log(prob + 1e-10)

                all_candidates.append((new_seq, new_score))

        # Sort candidates by score (higher is better)
        ordered = sorted(
            all_candidates,
            key=lambda x: x[1] / len(x[0]),
            reverse=True,
        )

        # Keep only best beam_size sequences
        sequences = ordered[:beam_size]

    best_sequence = sequences[0][0]

    # Convert word indices to actual words
    caption_words = []
    for idx in best_sequence:
        if idx not in [start_token, end_token]:
            word = tokenizer.index_word.get(idx)
            if word:
                caption_words.append(word)

    return " ".join(caption_words)


image_path = (
    r"C:\Users\PRITAM\OneDrive\Desktop\julia-vivcharyk-zFNn_F7arz4-unsplash.jpg"
)

photo_features = extract_features(image_path)
predicted_caption = generate_caption_beam(
    model=caption_model,
    photo=photo_features,
    tokenizer=tokenizer,
    max_len=max_length,
    beam_size=5,
)
print(f"Generated Caption is: {predicted_caption}")
