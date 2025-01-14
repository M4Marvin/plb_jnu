from rich import print
from pathlib import Path
from preprocessing.config import PDBBindDataset
from preprocessing.processors.protein import PocketProcessor

# Update paths to include full structure
data_root = Path("data/pdb_bind")
general_dataset_path = data_root / "general-set"
refined_dataset_path = data_root / "refined-set"

# Create PDBBind dataset object
dataset = PDBBindDataset.from_root(data_root)

# Now you can access any complex like:
complex_1a1b = dataset.get_complex("1a4h")

print(complex_1a1b)
print(dataset.complexes.__len__())

pp = PocketProcessor()
pp.process_complex(complex=complex_1a1b)
