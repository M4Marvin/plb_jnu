from rich import print
import os
from pathlib import Path
from pydantic import BaseModel, DirectoryPath, FilePath
from preprocessing.config import PDBBindDataset
import requests

# Update paths to include full structure
data_root = Path("data/pdb_bind")
general_dataset_path = data_root / "general-set"
refined_dataset_path = data_root / "refined-set"

# Create PDBBind dataset object
dataset = PDBBindDataset.from_root(data_root)

# Now you can access any complex like:
complex_1a1b = dataset.get_complex("1a1b")

print(complex_1a1b)
print(dataset.complexes.__len__())

# i have indexed and stored all the complexes
