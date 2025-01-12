from pydantic import BaseModel, DirectoryPath, FilePath, validator
from typing import Dict, List
from pathlib import Path
import pandas as pd


class PDBBindComplex(BaseModel):
    pdb_id: str
    pocket_pdb: FilePath
    protein_pdb: FilePath
    ligand_mol2: FilePath
    affinity: float
    set_type: str  # "general" or "refined"

    @classmethod
    def from_pdb_id(
        cls, pdb_id: str, affinity: float, base_path: Path, set_type: str
    ) -> "PDBBindComplex":
        """Create a complex from PDB ID and base directory"""
        return cls(
            pdb_id=pdb_id,
            pocket_pdb=base_path / f"{pdb_id}_pocket.pdb",
            protein_pdb=base_path / f"{pdb_id}_protein.pdb",
            ligand_mol2=base_path / f"{pdb_id}_ligand.mol2",
            affinity=affinity,
            set_type=set_type,
        )


class PDBBindDataset(BaseModel):
    root_path: DirectoryPath
    general_set: DirectoryPath
    refined_set: DirectoryPath
    binding_data: FilePath
    cleaned_data: FilePath
    complexes: Dict[str, PDBBindComplex]

    @classmethod
    def from_root(cls, root_path: Path) -> "PDBBindDataset":
        """Initialize dataset from root directory"""
        dataset = {
            "root_path": root_path,
            "general_set": root_path / "general-set",
            "refined_set": root_path / "refined-set",
            "binding_data": root_path / "PDBbind_2020_data.csv",
            "cleaned_data": root_path / "PDBbind_cleaned_dataset.csv",
        }

        # Read cleaned data to get PDB IDs
        cleaned_df = pd.read_csv(dataset["cleaned_data"])

        # Initialize complexes
        complexes = {}
        for _, row in cleaned_df.iterrows():
            pdb_id = row["PDB ID"]
            base_path = (
                dataset["general_set"]
                if row["set"] == "general"
                else dataset["refined_set"]
            )
            complex_path = base_path / pdb_id
            affinity = row["-log(Kd/Ki)"]
            complexes[pdb_id] = PDBBindComplex.from_pdb_id(
                pdb_id, affinity, complex_path, set_type=row["set"]
            )

        dataset["complexes"] = complexes
        return cls(**dataset)

    def get_complex(self, pdb_id: str) -> PDBBindComplex:
        """Get complex by PDB ID"""
        return self.complexes[pdb_id]
