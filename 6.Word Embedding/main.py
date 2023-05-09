import tensorflow as tf
import numpy as np

# sample text data
texts = ["The quick brown fox jumps over the lazy dog.",
         "She sells seashells by the seashore."]

# tokenize text data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)

# perform padding to make the sequence lengths the same
max_sequence_length = max([len(seq) for seq in sequences])
data = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length)

# train Word2Vec model
embedding_dim = 100
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index) + 1, embedding_dim, input_length=max_sequence_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(data, np.array([1, 0]), epochs=50, verbose=0)

model.save("wordemd.h5")
# get word vectors
word_vectors = model.get_weights()[0]
vector = word_vectors[word_index['jumps']]
print(vector)