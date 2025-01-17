import os
from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from rdkit import Chem
from rdkit.Chem import AllChem
import keras.saving

# Register 'sampling' for serialization
@keras.saving.register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    epsilon = np.random.normal(size=(z_mean.shape[0], z_mean.shape[1]))
    return z_mean + np.exp(0.5 * z_log_var) * epsilon

# Register 'mse' for serialization
@keras.saving.register_keras_serializable()
def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'app/static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
property_model_path = os.path.abspath('models/property_model.h5')
generator_model_path = os.path.abspath('models/molecule_generator.h5')

property_model = load_model(property_model_path, custom_objects={"sampling": sampling, "mse": mse})
generator_model = load_model(generator_model_path, custom_objects={"sampling": sampling})

# Preprocess SMILES string
def preprocess_smiles(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fingerprint).reshape(1, -1)

# Routes
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return "No file uploaded.", 400
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            with open(file_path, 'r') as f:
                smiles = f.read().strip()
            smiles_vector = preprocess_smiles(smiles)
            prediction = property_model.predict(smiles_vector)
            latent_vector = np.random.normal(size=(1, generator_model.input_shape[1]))
            generated_molecule = generator_model.predict(latent_vector)

            return render_template(
                'result.html',
                prediction=f"Predicted property: {prediction[0][0]}",
                generated_molecule=f"Generated molecule vector: {generated_molecule.tolist()}",
                smiles=smiles
            )
        except Exception as e:
            return f"An error occurred: {e}", 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
