from rich import print
from pathlib import Path
import numpy as np
from preprocessing.models import PDBBindDataset
from preprocessing.processors.protein import PocketProcessor
from preprocessing.processors.featurizer import Featurizer, FeaturizerConfig, make_grid
import pandas as pd
from biopandas.mol2 import PandasMol2


def test_featurization():
    # Setup paths
    data_root = Path("data/pdb_bind")
    general_dataset_path = data_root / "general-set"
    refined_dataset_path = data_root / "refined-set"

    # Load dataset
    dataset = PDBBindDataset.from_root(data_root)
    complex_1a4h = dataset.get_complex("1a4h")
    print("Processing complex:", complex_1a4h)

    # Initialize pocket processor
    pp = PocketProcessor()
    pp.process_complex(complex=complex_1a4h)

    # Load element properties
    elements = pd.read_csv("data/elements.csv")
    print("\nElement properties:")
    print(elements.head())

    # Create custom property calculator using element properties
    def get_vdw_radius(atom_row):
        atom_type = atom_row["atom_type"].split(".")[
            0
        ]  # Get base element from MOL2 type
        try:
            return float(
                elements[elements["symbol"] == atom_type]["vdw_radius"].iloc[0]
            )
        except (IndexError, KeyError):
            return 2.0  # Default value

    # Initialize featurizer with custom property
    config = FeaturizerConfig(
        save_molecule_codes=True, custom_properties=[get_vdw_radius]
    )
    featurizer = Featurizer(config)

    # Process ligand
    print("\nProcessing ligand...")
    ligand_mol = PandasMol2().read_mol2(complex_1a4h.ligand_mol2)
    ligand_coords, ligand_features = featurizer.get_features(ligand_mol, molcode=1.0)
    print(
        f"Ligand: {ligand_features.shape[0]} atoms, {ligand_features.shape[1]} features"
    )

    # Process protein pocket
    print("\nProcessing protein pocket...")
    pocket_mol = PandasMol2().read_mol2(complex_1a4h.charged_pocket_mol2)
    pocket_coords, pocket_features = featurizer.get_features(pocket_mol, molcode=-1.0)
    print(
        f"Pocket: {pocket_features.shape[0]} atoms, {pocket_features.shape[1]} features"
    )
    print(pocket_coords.shape)
    print(pocket_features.shape)

    # Create grids
    print("\nCreating feature grids...")
    ligand_grid = make_grid(ligand_coords, ligand_features)
    pocket_grid = make_grid(pocket_coords, pocket_features)

    print("\nGrid shapes:")
    print(f"Ligand grid: {ligand_grid.shape}")
    print(f"Pocket grid: {pocket_grid.shape}")

    # Print feature names
    print("\nFeature names:")
    print(featurizer.FEATURE_NAMES)

    return {
        "ligand_coords": ligand_coords,
        "ligand_features": ligand_features,
        "pocket_coords": pocket_coords,
        "pocket_features": pocket_features,
        "ligand_grid": ligand_grid,
        "pocket_grid": pocket_grid,
    }


if __name__ == "__main__":
    results = test_featurization()

    # Additional analysis of results
    print("\nFeature statistics:")
    for name, features in [
        ("Ligand", results["ligand_features"]),
        ("Pocket", results["pocket_features"]),
    ]:
        print(f"\n{name}:")
        print(f"Min values: {features.min(axis=0)}")
        print(f"Max values: {features.max(axis=0)}")
        print(f"Mean values: {features.mean(axis=0)}")
