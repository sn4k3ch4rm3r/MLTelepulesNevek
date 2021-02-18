# %%
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import os
import time
from tensorflow.python.framework.ops import IndexedSlices
# %%
df = pd.read_csv('telepulesek.csv')
names = df['Név'].values
np.random.shuffle(names)
for i in range(len(names)-1):
	names[i] += " " + names[i+1]

df = pd.read_csv('telepulesek.csv')
names_origin = df['Név'].values
names_origin
# names
# %%
chars = tf.strings.unicode_split(names, 'UTF-8')
chars
# %%
string = ""
for i in names:
	string += i
vocab = sorted(set(string))
print('{} unique characters'.format(len(vocab)))

# %%
ids_from_chars = preprocessing.StringLookup(
	vocabulary=list(vocab))

# %%
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
	 vocabulary=ids_from_chars.get_vocabulary(), invert=True)

# %%
def text_from_ids(ids):
	return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
# %%
ids = ids_from_chars(chars)
ids
# %%
def split_input_target(sequence):
	input_text = sequence[:-1]
	target_text = sequence[1:]
	return input_text, target_text

#%%
max_len = max(len(e) for e in ids)
res = tf.stack([tf.pad(e, [(0,max_len - len(e))], constant_values=0) for e in ids], axis=0)
res
#%%
ids_dataset = tf.data.Dataset.from_tensor_slices(res)
for seq in ids_dataset.take(10):
  print(text_from_ids(seq).numpy().decode('utf-8').replace('[UNK]', '0'))

#%%
data = res.numpy()
np.random.shuffle(data)
for seq in data[:5]:
	print(text_from_ids(seq).numpy().decode('utf-8').replace('[UNK]', '0'))
#%%
dataset = ids_dataset.map(split_input_target)
for input_example, target_example in  dataset.take(1):
    print("Input :", text_from_ids(input_example).numpy().decode('utf-8').replace('[UNK]', ''))
    print("Target:", text_from_ids(target_example).numpy().decode('utf-8').replace('[UNK]', ''))
# %%
BATCH_SIZE = 32
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

dataset
# %%
# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024
# %%
class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True, 
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else: 
      return x

# %%
model = MyModel(
    # Be sure the vocabulary size matches the `StringLookup` layers.
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)
# %%
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
# %%
model.summary()

# %%
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
# %%
print("Input:\n", text_from_ids(input_example_batch[0]).numpy().decode('utf-8'))
print()
print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy().decode('utf-8'))
# %%
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
# %%
example_batch_loss = loss(target_example_batch, example_batch_predictions)
mean_loss = example_batch_loss.numpy().mean()
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("Mean loss:        ", mean_loss)
# %%
model.compile(optimizer='adam', loss=loss)
# %%
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
# %%
EPOCHS = 100
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
# %%
class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature=temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "" or "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['','[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices = skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())]) 
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits] 
    predicted_logits, states =  self.model(inputs=input_ids, states=states, 
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "" or "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states
# %%
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
# %%
start = time.time()
states = None
next_char = tf.constant(['A'])
result = [next_char]

for n in range(2000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()

# print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)

print(f"\nRun time: {end - start}")

villages = result[0].numpy().decode('utf-8').split(' ')[:-1]
for i in villages:
	if not i in names_origin and len(i) < 25:
		print(i)
# %%
