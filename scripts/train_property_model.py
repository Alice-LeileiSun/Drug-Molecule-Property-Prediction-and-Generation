import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import keras.saving

# Register 'sampling' for serialization
@keras.saving.register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Register 'mse' for serialization
@keras.saving.register_keras_serializable()
def mse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

# Load data
data = pd.read_csv('data/processed_esol.csv')

# Prepare features and target
X = data[['MolecularWeight']]
Y = data['measured log solubility in mols per litre']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use in inference
import joblib
joblib.dump(scaler, 'models/scaler.pkl')

# Define the model
input_dim = X_train.shape[1]
latent_dim = 2

inputs = Input(shape=(input_dim,), name="encoder_input")
h = Dense(64, activation="relu", name="encoder_dense_1")(inputs)
z_mean = Dense(latent_dim, name="z_mean")(h)
z_log_var = Dense(latent_dim, name="z_log_var")(h)

z = Lambda(
    sampling,
    output_shape=(latent_dim,),
    name="sampling"
)([z_mean, z_log_var])

decoder_h = Dense(64, activation="relu", name="decoder_dense_1")
decoder_output = Dense(1, name="decoder_output")
h_decoded = decoder_h(z)
outputs = decoder_output(h_decoded)

model = Model(inputs, outputs, name="property_model")
model.compile(optimizer="adam", loss=mse, metrics=["mae"])

# Train the model
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    callbacks=[early_stopping]
)

# Save the model
model.save('models/property_model.h5')
print("Model saved to models/property_model.h5")
