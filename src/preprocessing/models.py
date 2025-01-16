from pydantic import BaseModel, DirectoryPath, FilePath
from typing import Dict, Optional
from pathlib import Path
import pandas as pd


class FilePattern(BaseModel):
    """
    A class to define and generate file paths based on a given base path and PDB ID.

    Attributes:
    -----------
    pocket_pdb : str
        The pattern for the pocket PDB file.
    protein_pdb : str
        The pattern for the protein PDB file.
    ligand_mol2 : str
        The pattern for the ligand MOL2 file.
    charged_pocket_mol2 : str
        The pattern for the charged pocket MOL2 file.
    """

    pocket_pdb: str = "{pdb_id}_pocket.pdb"
    protein_pdb: str = "{pdb_id}_protein.pdb"
    ligand_mol2: str = "{pdb_id}_ligand.mol2"
    charged_pocket_mol2: str = "{pdb_id}_charged_pocket.mol2"

    def get_path(self, base_path: Path, pdb_id: str, pattern: str) -> Path:
        """
        Generates the full file path based on the base path, PDB ID, and file pattern.

        Parameters:
        -----------
        base_path : Path
            The base directory path.
        pdb_id : str
            The PDB ID.
        pattern : str
            The file pattern attribute name.

        Returns:
        --------
        Path
            The full file path.
        """
        return base_path / getattr(self, pattern).format(pdb_id=pdb_id)


# Create a single instance to be used throughout
PATTERNS = FilePattern()


class PDBBindComplex(BaseModel):
    """
    A class to represent a PDBBind complex, containing paths to protein and ligand files,
    as well as processed files like the charged pocket mol2 file.

    Attributes:
    -----------
    pdb_id : str
        The PDB ID of the complex.
    protein_pdb : FilePath
        The path to the protein PDB file.
    ligand_mol2 : FilePath
        The path to the ligand MOL2 file.
    charged_pocket_mol2 : Optional[FilePath]
        The path to the charged pocket MOL2 file (default is None).
    affinity : float
        The affinity value for the complex.
    unrealistic_charge_present : bool
        Indicates if unrealistic charges are present in the charged pocket MOL2 file (default is False).
    set_type : str
        The type of set the complex belongs to ("general" or "refined").
    """

    pdb_id: str
    protein_pdb: FilePath
    ligand_mol2: FilePath
    charged_pocket_mol2: Optional[FilePath] = None
    affinity: float
    unrealistic_charge_present: bool = False
    set_type: str  # "general" or "refined"

    @classmethod
    def from_pdb_id(
        cls, pdb_id: str, affinity: float, base_path: Path, set_type: str
    ) -> "PDBBindComplex":
        """
        Creates a PDBBindComplex instance from a given PDB ID, affinity, base path, and set type.

        Parameters:
        -----------
        pdb_id : str
            The PDB ID of the complex.
        affinity : float
            The affinity value for the complex.
        base_path : Path
            The base directory path.
        set_type : str
            The type of set the complex belongs to ("general" or "refined").

        Returns:
        --------
        PDBBindComplex
            A new PDBBindComplex instance.
        """

        return cls(
            pdb_id=pdb_id,
            protein_pdb=PATTERNS.get_path(base_path, pdb_id, "protein_pdb"),
            ligand_mol2=PATTERNS.get_path(base_path, pdb_id, "ligand_mol2"),
            affinity=affinity,
            set_type=set_type,
        )

    def set_charged_pocket(self, base_path: Path):
        """
        Sets the path to the charged pocket MOL2 file after preprocessing.

        Parameters:
        -----------
        base_path : Path
            The base directory path.
        """
        self.charged_pocket_mol2 = PATTERNS.get_path(
            base_path, self.pdb_id, "charged_pocket_mol2"
        )


class PDBBindDataset(BaseModel):
    """
    A class to represent a PDBBind dataset, containing paths to the general and refined sets,
    binding data, cleaned data, and individual complexes.

    Attributes:
    -----------
    root_path : DirectoryPath
        The root directory path of the PDBBind dataset.
    general_set : DirectoryPath
        The directory path to the general set.
    refined_set : DirectoryPath
        The directory path to the refined set.
    binding_data : FilePath
        The path to the raw binding data CSV file.
    cleaned_data : FilePath
        The path to the cleaned binding data CSV file.
    complexes : Dict[str, PDBBindComplex]
        A dictionary of PDBBindComplex instances indexed by PDB ID.
    """

    root_path: DirectoryPath
    general_set: DirectoryPath
    refined_set: DirectoryPath
    binding_data: FilePath
    cleaned_data: FilePath
    complexes: Dict[str, PDBBindComplex]

    @classmethod
    def from_root(cls, root_path: Path) -> "PDBBindDataset":
        """
        Initializes a PDBBindDataset instance from a given root directory.

        Parameters:
        -----------
        root_path : Path
            The root directory path of the PDBBind dataset.

        Returns:
        --------
        PDBBindDataset
            A new PDBBindDataset instance.
        """
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
        """
        Retrieves a PDBBindComplex instance by PDB ID.

        Parameters:
        -----------
        pdb_id : str
            The PDB ID of the complex.

        Returns:
        --------
        PDBBindComplex
            The PDBBindComplex instance corresponding to the given PDB ID.
        """
        return self.complexes[pdb_id]
