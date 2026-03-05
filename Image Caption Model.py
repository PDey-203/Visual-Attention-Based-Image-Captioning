import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.regularizers import l2


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
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


def data_generator(image_list, caption_dict):
    for img_id in image_list:
        photo = features[img_id]  # (225,1280)
        captions = caption_dict[img_id]

        for caption in captions:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq = seq[:i]
                out_seq = seq[i]

                in_seq = pad_sequences([in_seq], maxlen=max_length, padding="post")[0]

                yield (photo, in_seq), out_seq


def build_tf_dataset(image_list, caption_dict, batch_size=64, is_training=True):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(image_list, caption_dict),
        output_signature=(
            (
                tf.TensorSpec(shape=(225, 1280), dtype=tf.float32),
                tf.TensorSpec(shape=(max_length,), dtype=tf.int32),
            ),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    if is_training:
        dataset = dataset.shuffle(1500)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def preprocess_caption(text):
    text = text.lower()

    cleaned = ""
    for char in text:
        if char.isalpha() or char == " ":
            cleaned += char

    cleaned = " ".join(cleaned.split())
    new_text = "startseq " + cleaned + " endseq"
    return new_text


def extract_features(filename):
    img = load_img(filename, target_size=(480, 480))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    features = feature_extractor.predict(img, verbose=0)
    features = features.reshape(-1, features.shape[-1])
    return features


df = pd.read_csv(r"C:\Users\PRITAM\OneDrive\Desktop\captions.csv")

df["cleaned_caption"] = df["caption"].apply(preprocess_caption)


tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(df["cleaned_caption"])
vocab_size = len(tokenizer.word_index) + 1

with open(r"C:\Users\PRITAM\OneDrive\Desktop\tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved successfully.")


sequences = tokenizer.texts_to_sequences(df["cleaned_caption"])
max_length = max(len(seq) for seq in sequences)
print("Maximum length of caption:", max_length)

unique_images = df["image"].unique().tolist()
train_images, test_images = train_test_split(
    unique_images, test_size=0.2, random_state=42
)


captions_dict = {}
for img, caption in zip(df["image"], df["cleaned_caption"]):
    captions_dict.setdefault(img, []).append(caption)

train_dict = {img: captions_dict[img] for img in train_images}
test_dict = {img: captions_dict[img] for img in test_images}


base_model = EfficientNetV2L(weights="imagenet", include_top=False)
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
feature_extractor.trainable = False
feature_extractor.summary()


features = {}
for img_name in unique_images:
    path = r"C:\\Users\\PRITAM\\OneDrive\\Desktop\\Images\\" + img_name
    features[img_name] = extract_features(path)

feature_extractor.save(r"C:\Users\PRITAM\OneDrive\Desktop\feature_extractor.keras")
print("Feature extractor saved successfully.")


image_input = Input(shape=(225, 1280))
text_input = Input(shape=(max_length,))

embedding = Embedding(vocab_size, 256, mask_zero=True)(text_input)
embedding = Dropout(0.4)(embedding)
lstm_output, state_h, state_c = LSTM(256, dropout=0.3, return_state=True)(embedding)

attention = BahdanauAttention(256)
context_vector = attention(image_input, state_h)

combined = Concatenate()([context_vector, state_h])
dense1 = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(combined)
dropout = Dropout(0.4)(dense1)

output = Dense(vocab_size, activation="softmax")(dropout)

caption_model = Model([image_input, text_input], output)
caption_model.summary()

caption_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"],
)

train_dataset = build_tf_dataset(train_images, train_dict, is_training=True)
val_dataset = build_tf_dataset(test_images, test_dict, is_training=False)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True
)

caption_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    callbacks=[early_stop, lr_scheduler],
)

caption_model.save(r"C:\Users\PRITAM\OneDrive\Desktop\caption_Model.keras")
print("Caption model saved successfully.")
