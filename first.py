import tensorflow as tf

# Define the input layers
input_1 = tf.keras.layers.Input(shape=(100,))
input_2 = tf.keras.layers.Input(shape=(100,))

# Define the shared layers
shared_layer_1 = tf.keras.layers.Dense(128, activation='relu')
shared_layer_2 = tf.keras.layers.Dense(64, activation='relu')
shared_layer_3 = tf.keras.layers.Dense(32, activation='relu')

# Define the individual layers
individual_layer_1 = tf.keras.layers.Dense(16, activation='relu')
individual_layer_2 = tf.keras.layers.Dense(16, activation='relu')

# Define the shared embedding
embedding_1 = shared_layer_3(shared_layer_2(shared_layer_1(input_1)))
embedding_2 = shared_layer_3(shared_layer_2(shared_layer_1(input_2)))

# Concatenate the embeddings
concatenated_embedding = tf.keras.layers.Concatenate()([embedding_1, embedding_2])

# Add the individual layers
individual_output_1 = individual_layer_1(concatenated_embedding)
individual_output_2 = individual_layer_2(concatenated_embedding)

# Define the model
model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=[individual_output_1, individual_output_2])
