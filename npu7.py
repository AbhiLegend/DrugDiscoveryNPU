from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image
import numpy as np
import torch
import openvino.runtime as ov
#import intel_npu_acceleration_library

# Define the function to convert SMILES to fingerprints
def smiles_to_fp(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp)

# Load the OpenVINO model (update the path as needed)
model_path = 'lipophilicity_openvino.xml'  # Update this path
core = ov.Core()
#compiled_model = intel_npu_acceleration_library.compile(model_path, "NPU")
compiled_model = core.compile_model(model_path, "NPU")
print("Compiling in NPU")

# Define the prediction function
def predict_lipophilicity(smiles):
    fp = smiles_to_fp(smiles)
    input_tensor = torch.tensor(fp, dtype=torch.float32).unsqueeze(0)
    input_numpy = input_tensor.numpy()

    # Create OpenVINO tensor from NumPy array
    ov_input_tensor = ov.Tensor(input_numpy)

    # Run model inference
    result = compiled_model([ov_input_tensor])[0]
    return result[0]

# Example SMILES strings
smiles_options = [
    "C[C@H](N)C(=O)O", "CCO", "CCN(CC)CC", "CC(=O)O", "C1=CC=C(C=C1)C(=O)O",
    "C1CCC(CC1)N", "CC(C(=O)O)N", "C1CCCCC1", "C1=CC=CC=C1", "C1=CN=C(N=C1)N",
    "C1CC1", "C1=CC=C(C=C1)O", "C1=CN=CN1", "C1=CC=C(C=C1)N", "C1=CC=CC=C1N",
    "C1CCC(CC1)O", "C1=CC=C(C=C1)Cl", "C1=CN=C(N=C1)N", "C1CCNCC1", "C1=CC=C(C=C1)F"
]

# Example SMILES selection
selected_smiles = smiles_options[0]

# Example prediction
predicted_lipophilicity = predict_lipophilicity(selected_smiles)
print(f"Predicted Lipophilicity: {predicted_lipophilicity}")

# Visualize the molecule
mol = Chem.MolFromSmiles(selected_smiles)
mol_image = Draw.MolToImage(mol)
mol_image.show()
