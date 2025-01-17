import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
import pandas as pd
import keras.saving

# Register 'sampling' for serialization
@keras.saving.register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Load data
data = pd.read_csv('data/processed_esol.csv')
smiles = data['smiles']

# Define VAE model
input_dim = 100
latent_dim = 10

inputs = Input(shape=(input_dim,), name="encoder_input")
h = Dense(128, activation='relu', name="encoder_hidden")(inputs)
z_mean = Dense(latent_dim, name="z_mean")(h)
z_log_var = Dense(latent_dim, name="z_log_var")(h)

z = Lambda(
    sampling,
    output_shape=(latent_dim,),
    name="sampling"
)([z_mean, z_log_var])

decoder_h = Dense(128, activation='relu', name="decoder_hidden")
decoder_output = Dense(input_dim, activation='sigmoid', name="decoder_output")
h_decoded = decoder_h(z)
outputs = decoder_output(h_decoded)

vae = Model(inputs, outputs, name="vae_model")
vae.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
X = np.random.rand(len(smiles), input_dim)
vae.fit(X, X, epochs=50, batch_size=16)

# Save the model
vae.save('models/molecule_generator.h5')
print("Model saved as 'models/molecule_generator.h5'")
