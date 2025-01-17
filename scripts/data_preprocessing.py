import pandas as pd
from rdkit import Chem   
from rdkit.Chem import Descriptors 

# load data
data = pd.read_csv('data/raw/esol.csv')

# check data
print(data.head())

#Convert SMILES to molecular objects
def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)

#Calculate molecular descriptors (e.g., molecular weight)
def compute_descriptors(smiles):
    mol = smiles_to_mol(smiles)
    if mol:
        return Descriptors.MolWt(mol)
    else:
        return None
    
#Add a molecular weight column
data['MolecularWeight']=data['smiles'].apply(compute_descriptors)

#Save processed data
data.to_csv('data/processed_esol.csv', index=False)
