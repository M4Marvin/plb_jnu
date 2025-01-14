from pathlib import Path

import numpy as np
import requests
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from pydantic import BaseModel, Field
from pymol import cmd
from scipy.spatial import cKDTree

from preprocessing.config import FilePattern, PDBBindComplex

PATTERNS = FilePattern()


class PocketProcessorConfig(BaseModel):
    distance_threshold: float = Field(
        default=8.0, description="Distance threshold for pocket extraction in Angstroms"
    )
    include_heteroatoms: bool = Field(
        default=False, description="Include heteroatoms in pocket extraction"
    )
    verbose: bool = Field(default=False, description="Enable verbose output")
    acc2_api_url: str = Field(
        default="https://acc2-api.biodata.ceitec.cz", description="URL for ACC2 API"
    )
    charge_method: str = Field(
        default="eqeq", description="Method for charge calculation"
    )


class PocketProcessor(BaseModel):
    config: PocketProcessorConfig = Field(default_factory=PocketProcessorConfig)

    def process_complex(self, complex: PDBBindComplex) -> Path:
        base_dir = complex.protein_pdb.parent

        pocket_pdb = self.extract_pocket(
            protein_pdb=complex.protein_pdb,
            ligand_mol2=complex.ligand_mol2,
            output_path=PATTERNS.get_path(base_dir, complex.pdb_id, "pocket_pdb"),
        )

        pocket_mol2 = self.add_h_and_convert_to_mol2(
            pdb_path=pocket_pdb, output_path=base_dir / "intermediate_pocket.mol2"
        )

        charged_mol2_path = PATTERNS.get_path(
            base_dir, complex.pdb_id, "charged_pocket_mol2"
        )
        charged_mol2 = self.add_mol2_charges(
            pocket_mol2=pocket_mol2, output_path=charged_mol2_path
        )

        complex.charged_pocket_mol2 = charged_mol2

        if (base_dir / "intermediate_pocket.mol2").exists():
            (base_dir / "intermediate_pocket.mol2").unlink()

        if (pocket_pdb).exists():
            (pocket_pdb).unlink()

        return charged_mol2

    def add_h_and_convert_to_mol2(self, pdb_path: Path, output_path: Path) -> Path:
        """Add hydrogens and convert PDB to MOL2 format using PyMOL"""
        cmd.delete("all")
        cmd.load(str(pdb_path))
        cmd.h_add("sol")
        cmd.save(str(output_path))
        return output_path

    def add_mol2_charges(self, pocket_mol2: Path, output_path: Path) -> Path:
        """Add charges to MOL2 file using ACC2 API"""
        r = requests.post(
            f"{self.config.acc2_api_url}/send_files",
            files={"file[]": open(pocket_mol2, "rb")},
        )

        r_id = list(r.json()["structure_ids"].values())[0]

        r_out = requests.get(
            # f"{self.config.acc2_api_url}/calculate_charges?structure_id={r_id}&method={self.config.charge_method}&generate_mol2=true"
            f"{self.config.acc2_api_url}/calculate_charges?structure_id={r_id}&generate_mol2=true"
        )

        output_path.write_bytes(r_out.content)
        return output_path

    def extract_pocket(
        self,
        protein_pdb: Path,
        ligand_mol2: Path,
        output_path: Path,
    ) -> Path:
        """Extract pocket around ligand"""
        # Read input files
        if self.config.verbose:
            print(f"Reading protein PDB: {protein_pdb}")
        protein = PandasPdb().read_pdb(str(protein_pdb))
        ligand = PandasMol2().read_mol2(str(ligand_mol2))

        # Define protein atoms dataframe
        protein_atom = protein.df["ATOM"]
        protein_hetatm = protein.df["HETATM"]
        ligand_nonh = ligand.df[ligand.df["atom_type"] != "H"]

        # Create k-d trees for protein atoms and heteroatoms
        protein_atom_tree = cKDTree(
            protein_atom[["x_coord", "y_coord", "z_coord"]].values
        )
        protein_hetatm_tree = cKDTree(
            protein_hetatm[["x_coord", "y_coord", "z_coord"]].values
        )

        # Find nearby atoms
        ligand_coords = ligand_nonh[["x", "y", "z"]].values
        nearby_atom_indices = protein_atom_tree.query_ball_point(
            ligand_coords, r=self.config.distance_threshold
        )
        nearby_hetatm_indices = protein_hetatm_tree.query_ball_point(
            ligand_coords, r=self.config.distance_threshold
        )

        # Get unique residues
        pocket_residues = (
            protein_atom.iloc[np.unique(np.concatenate(nearby_atom_indices))][
                ["chain_id", "residue_number", "insertion"]
            ]
            .apply(lambda x: "_".join(x.astype(str)), axis=1)
            .unique()
        )

        pocket_heteroatoms = protein_hetatm.iloc[
            np.unique(np.concatenate(nearby_hetatm_indices))
        ]["residue_number"].unique()

        # Filter atoms
        pocket_atoms = protein_atom[
            protein_atom.apply(
                lambda x: "_".join(
                    [str(x["chain_id"]), str(x["residue_number"]), str(x["insertion"])]
                ),
                axis=1,
            ).isin(pocket_residues)
        ]

        # Filter heteroatoms
        if self.config.include_heteroatoms:
            pocket_hetatms = protein_hetatm[
                protein_hetatm["residue_number"].isin(pocket_heteroatoms)
            ]
        else:
            pocket_hetatms = protein_hetatm[
                (protein_hetatm["residue_number"].isin(pocket_heteroatoms))
                & (protein_hetatm["residue_name"] == "HOH")
            ]

        # Create output PDB
        pred_pocket = PandasPdb()
        pred_pocket.df["ATOM"] = pocket_atoms
        pred_pocket.df["HETATM"] = pocket_hetatms

        if self.config.verbose:
            print(f"Saving pocket PDB to: {output_path}")
        pred_pocket.to_pdb(str(output_path))
        return output_path
