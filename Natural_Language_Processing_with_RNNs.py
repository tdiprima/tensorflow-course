vocab = {}  # Map word to integer representing it
word_encoding = 1


def bag_of_words(text):
    global word_encoding

    # Create a list of all the words in the text, we'll assume there is no grammar in our text for this example
    words = text.lower().split(" ")

    # Store all the encodings and their frequency
    bag = {}

    for word in words:
        if word in vocab:
            encoding = vocab[word]  # Get encoding from vocab
        else:
            vocab[word] = word_encoding
            encoding = word_encoding
            word_encoding += 1

        if encoding in bag:
            bag[encoding] += 1
        else:
            bag[encoding] = 1

    return bag


text = "this is a test to see if this test will work is is test a a"
bag = bag_of_words(text)

import pprint as pp

print("\nBag of words:\n")
pp.pprint(bag)

print("\nMap word to integer:\n")
pp.pprint(vocab)

positive_review = "I thought the movie was going to be bad but it was actually amazing"
negative_review = "I thought the movie was going to be amazing but it was actually bad"

pos_bag = bag_of_words(positive_review)
neg_bag = bag_of_words(negative_review)

print("\nPositive:\n", pos_bag)
print("\nNegative:\n", neg_bag)

# They're almost the same.

# WE'VE KEPT THE ORDERING IN THIS ONE
vocab = {}
word_encoding = 1


def one_hot_encoding(text):
    global word_encoding

    words = text.lower().split(" ")
    encoding = []  # Instead of putting it in a map, put it in a list this time.

    for word in words:
        if word in vocab:
            code = vocab[word]
            encoding.append(code)
        else:
            vocab[word] = word_encoding
            encoding.append(word_encoding)
            word_encoding += 1

    return encoding


text = "this is a test to see if this test will work is is test a a"
encoding = one_hot_encoding(text)

print("\nOne-Hot Encoding:\n", encoding)
print("\nVocab:\n", vocab)

positive_review = "I thought the movie was going to be bad but it was actually amazing"
negative_review = "I thought the movie was going to be amazing but it was actually bad"

pos_encode = one_hot_encoding(positive_review)
neg_encode = one_hot_encoding(negative_review)

print("Positive:", pos_encode)
print("Negative:", neg_encode)

# DATA IS FROM KERAS
from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

# Lets look at one review
# train_data[1]
len(train_data[1])

from keras.utils import pad_sequences

train_data = keras.utils.pad_sequences(train_data, maxlen=MAXLEN)
test_data = keras.utils.pad_sequences(test_data, maxlen=MAXLEN)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),  # word embedding layer
    tf.keras.layers.LSTM(32),  # LSTM layer
    tf.keras.layers.Dense(1, activation="sigmoid")  # dense node => predicted sentiment
])

model.summary()

# Loss function: binary cross-entropy
# Optimizer: RMSProp (Root Mean Squared Propagation); similar to the gradient descent algorithm with momentum.
# Metrics: accuracy

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])

history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

results = model.evaluate(test_data, test_labels)

results

word_index = imdb.get_word_index()


def encode_text(text):
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    print("1", tokens)  # len: 7

    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    print("2", tokens)

    return keras.utils.pad_sequences([tokens], MAXLEN)[0]


text = "that movie was just amazing, so amazing"
encoded = encode_text(text)

# encoded  # len: 250

reverse_word_index = {value: key for (key, value) in word_index.items()}


def decode_integers(integers):
    """
    While we're at it, let's make a decode function.
    """
    PAD = 0
    text = ""
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + " "

    return text[:-1]


decode_integers(encoded)

def predict(text):
    """
    Make a prediction
    """
    encoded_text = encode_text(text)
    pred = np.zeros((1, 250))
    pred[0] = encoded_text
    result = model.predict(pred)
    print(result[0])


positive_review = "That movie was! really loved it and would great watch it again because it was amazingly great"
predict(positive_review)

print()

negative_review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review)

from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# Length of text: the number of characters in it
print('\nLength of text: {} characters'.format(len(text)))

# Take a look at the first 250 characters in text
print(text[:250])

# TODO: Look for a better way.
vocab = sorted(set(text))

# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)


def text_to_int(text):
    """
    Text to integer
    """
    return np.array([char2idx[c] for c in text])


text_as_int = text_to_int(text)

# Let's look at how part of our text is encoded
print("\nText:", text[:13])
print("\nEncoded:", text_to_int(text[:13]))

def int_to_text(ints):
    """
    Integer to text
    """
    try:
        ints = ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])


int_to_text(text_as_int[:13])

seq_length = 100  # length of sequence for a training example
examples_per_epoch = len(text) // (seq_length + 1)  # perform integer division

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# BATCH
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):  # for the example: hello
    input_text = chunk[:-1]  # hell
    target_text = chunk[1:]  # ello
    return input_text, target_text  # hell, ello


# We use MAP to apply the above function to every entry
dataset = sequences.map(split_input_target)

for x, y in dataset.take(2):
    print("\n\nEXAMPLE\n")
    print("INPUT:")
    print(int_to_text(x))
    print("\nOUTPUT:")
    print(int_to_text(y))

# MAKE TRAINING BATCHES
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)  # number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """
    BUILD MODEL
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)

model.summary()

for input_example_batch, target_example_batch in data.take(1):
    # Ask our model for a prediction on our first batch of training data (64 entries)
    example_batch_predictions = model(input_example_batch)
    
    # Print the output shape
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

# We can see that the prediction is an array of 64 arrays, one for each entry in the batch

print("\nLen:", len(example_batch_predictions))

# tf.Tensor, shape=(64, 100, 65), dtype=float32)
# print("\nPreds:\n", example_batch_predictions)

# Examine one prediction
pred = example_batch_predictions[0]

print("\nLen:", len(pred))
# print("\nPreds:\n", pred)  # shape=(100, 65)

# Notice this is a 2d array of length 100, where each interior array is the prediction for the next character at each time step

# Finally, we'll look at a prediction at the first timestep
time_pred = pred[0]

print("\nLen:", len(time_pred))
# print(time_pred)  # tf.Tensor, shape=(65,), dtype=float32)

# And of course it's 65 values, representing the probability of each character occurring next

# If we want to determine the predicted character, we need to sample the output distribution
# (pick a value based on probabillity)
sampled_indices = tf.random.categorical(pred, num_samples=1)

# now we can reshape that array...
sampled_indices = np.reshape(sampled_indices, (1, -1))[0]

# ...and convert all the integers to numbers to see the actual characters
predicted_chars = int_to_text(sampled_indices)

# and this is what the model predicted for training sequence 1
predicted_chars

def loss(labels, logits):
    """
    Custom LOSS FUNCTION
    """
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# Adam optimizer, custom loss function
model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'

# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# CHECKPOINT CALLBACK
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# epochs=50
history = model.fit(data, epochs=50, callbacks=[checkpoint_callback])

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string):
    """
    Generate text using the learned model (Evaluation step)
    """
    # Number of characters to generate
    num_generate = 800

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension

        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

inp = input("Type a starting string:")

print(generate_text(model, inp))  # She is a bass player