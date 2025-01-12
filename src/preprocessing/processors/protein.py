import requests
from pymol import cmd
import os
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from scipy.spatial import cKDTree
import numpy as np
from pathlib import Path

from preprocessing.config import PDBBindComplex


def add_mol2_charges(pocket_mol2, output_path):
    # upload the pocket mol2 file to the ACC2 API
    r = requests.post(
        "https://acc2-api.biodata.ceitec.cz/send_files",
        files={"file[]": open(pocket_mol2, "rb")},
    )

    # obtain ID number for uploaded file
    r_id = list(r.json()["structure_ids"].values())[0]

    # calculate charges using eqeq method
    r_out = requests.get(
        "https://acc2-api.biodata.ceitec.cz/calculate_charges?structure_id="
        + r_id
        + "&method=eqeq&generate_mol2=true"
    )

    # save output mol2 file
    open(f"{output_path}/charged_pocket.mol2", "wb").write(r_out.content)


def add_h_and_convert_to_mol2(pdb_path, output_path):
    cmd.delete("all")

    # load in the created pocket pdb file
    cmd.load("sample_data/pocket.pdb")

    # add hydrogens to the water molecules
    cmd.h_add("sol")

    # save the state as a mol2 file
    cmd.save("sample_data/pocket.mol2")


class PocketProcessor:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def process_complex(self, complex: PDBBindComplex) -> Path:
        """Process a single complex through the entire pocket pipeline"""
        # Create complex-specific output directory
        complex_dir = self.output_dir / complex.pdb_id
        complex_dir.mkdir(exist_ok=True)

        # Run pipeline stages
        pocket_pdb = self.extract_pocket(
            protein_pdb=complex.protein_pdb,
            ligand_mol2=complex.ligand_mol2,
            output_path=complex_dir,
        )

        pocket_mol2 = self.add_h_and_convert_to_mol2(
            pdb_path=pocket_pdb, output_path=complex_dir
        )

        charged_mol2 = self.add_mol2_charges(
            pocket_mol2=pocket_mol2, output_path=complex_dir
        )

        return charged_mol2

    def extract_pocket(
        self,
        protein_pdb: Path,
        ligand_mol2: Path,
        output_path: Path,
        distance_threshold: float = 8.0,
        include_heteroatoms: bool = False,
        verbose: bool = False,
    ) -> Path:
        """
        Extract the protein pocket residues and heteroatoms within a specified distance of the ligand.

        Parameters:
        - protein_pdb (str): Path to the protein PDB file.
        - ligand_mol2 (str): Path to the ligand MOL2 file.
        - distance_threshold (float): Distance threshold (in Angstroms) for determining the pocket (default: 8.0).
        - output_path (str): Path to save the pocket PDB file (default: None, saves as 'pocket.pdb' in the current directory).
        - include_heteroatoms (bool): Whether to include heteroatoms in the pocket (default: False, includes only water molecules).
        - verbose (bool): Whether to print verbose output for each step (default: False).

        Returns:
        - pred_pocket (PandasPdb): Biopandas PDB object representing the extracted pocket.
        """
        try:
            # Read in protein PDB file
            protein = PandasPdb().read_pdb(protein_pdb)
            if verbose:
                print("[green]Protein PDB file read successfully:[/green]", protein_pdb)
        except IOError:
            print("[red]Error: Could not read protein PDB file:[/red]", protein_pdb)
            return None

        try:
            # Read in ligand MOL2 file
            ligand = PandasMol2().read_mol2(ligand_mol2)
            if verbose:
                print("[green]Ligand MOL2 file read successfully:[/green]", ligand_mol2)
        except IOError:
            print("[red]Error: Could not read ligand MOL2 file:[/red]", ligand_mol2)
            return None

        # Define protein atoms dataframe
        protein_atom = protein.df["ATOM"]
        if verbose:
            print("[grey]Protein atoms dataframe created.[/grey]")
            print("Protein atoms dataframe shape:", protein_atom.shape)

        # Define protein heteroatoms dataframe
        protein_hetatm = protein.df["HETATM"]
        if verbose:
            print("[grey]Protein heteroatoms dataframe created.[/grey]")
            print("Protein heteroatoms dataframe shape:", protein_hetatm.shape)

        # Define ligand non-H atoms dataframe
        ligand_nonh = ligand.df[ligand.df["atom_type"] != "H"]
        if verbose:
            print("[grey]Ligand non-H atoms dataframe created.[/grey]")
            print("Ligand non-H atoms dataframe shape:", ligand_nonh.shape)

        # Create k-d trees for protein atoms and heteroatoms
        protein_atom_tree = cKDTree(
            protein_atom[["x_coord", "y_coord", "z_coord"]].values
        )
        protein_hetatm_tree = cKDTree(
            protein_hetatm[["x_coord", "y_coord", "z_coord"]].values
        )
        if verbose:
            print("[grey]K-d trees created for protein atoms and heteroatoms.[/grey]")

        # Find protein atoms within the distance threshold of ligand atoms
        ligand_coords = ligand_nonh[["x", "y", "z"]].values
        nearby_atom_indices = protein_atom_tree.query_ball_point(
            ligand_coords, r=distance_threshold
        )
        if verbose:
            print(
                "[grey]Nearby protein atoms within[/grey]",
                distance_threshold,
                "[grey]Angstroms of ligand atoms found.[/grey]",
            )
            print("Number of nearby protein atoms:", len(nearby_atom_indices))

        # Find protein heteroatoms within the distance threshold of ligand atoms
        nearby_hetatm_indices = protein_hetatm_tree.query_ball_point(
            ligand_coords, r=distance_threshold
        )
        if verbose:
            print(
                "[grey]Nearby protein heteroatoms within[/grey]",
                distance_threshold,
                "[grey]Angstroms of ligand atoms found.[/grey]",
            )
            print("Number of nearby protein heteroatoms:", len(nearby_hetatm_indices))

        # Get unique residues from nearby protein atoms
        pocket_residues = (
            protein_atom.iloc[np.unique(np.concatenate(nearby_atom_indices))][
                ["chain_id", "residue_number", "insertion"]
            ]
            .apply(lambda x: "_".join(x.astype(str)), axis=1)
            .unique()
        )
        if verbose:
            print("[grey]Unique pocket residues extracted.[/grey]")
            print("Number of unique pocket residues:", len(pocket_residues))

        # Get unique heteroatoms from nearby protein heteroatoms
        pocket_heteroatoms = protein_hetatm.iloc[
            np.unique(np.concatenate(nearby_hetatm_indices))
        ]["residue_number"].unique()
        if verbose:
            print("[grey]Unique pocket heteroatoms extracted.[/grey]")
            print("Number of unique pocket heteroatoms:", len(pocket_heteroatoms))

        # Filter protein atoms by pocket residues
        pocket_atoms = protein_atom[
            protein_atom.apply(
                lambda x: "_".join(
                    [str(x["chain_id"]), str(x["residue_number"]), str(x["insertion"])]
                ),
                axis=1,
            ).isin(pocket_residues)
        ]
        if verbose:
            print("[grey]Protein atoms filtered by pocket residues.[/grey]")
            print("Number of pocket atoms:", pocket_atoms.shape[0])

        # Filter protein heteroatoms by pocket heteroatoms
        if include_heteroatoms:
            pocket_hetatms = protein_hetatm[
                protein_hetatm["residue_number"].isin(pocket_heteroatoms)
            ]
            if verbose:
                print("[grey]All heteroatoms included in the pocket.[/grey]")
                print("Number of pocket heteroatoms:", pocket_hetatms.shape[0])
        else:
            pocket_hetatms = protein_hetatm[
                (protein_hetatm["residue_number"].isin(pocket_heteroatoms))
                & (protein_hetatm["residue_name"] == "HOH")
            ]
            if verbose:
                print("[grey]Only water molecules included in the pocket.[/grey]")
                print("Number of pocket water molecules:", pocket_hetatms.shape[0])

        # Initialize biopandas object to write out pocket PDB file
        pred_pocket = PandasPdb()
        if verbose:
            print("[grey]Biopandas PDB object initialized.[/grey]")

        # Define the atoms and heteroatoms of the object
        pred_pocket.df["ATOM"], pred_pocket.df["HETATM"] = pocket_atoms, pocket_hetatms
        if verbose:
            print(
                "[grey]Pocket atoms and heteroatoms assigned to the PDB object.[/grey]"
            )

        # Save the created object to a PDB file
        if output_path is None:
            output_path = "pocket.pdb"
        else:
            output_path = os.path.join(output_path, "pocket.pdb")

        try:
            pred_pocket.to_pdb(output_path)
            print("[green]Pocket PDB file saved as:[/green]", output_path)
        except IOError:
            print("[red]Error: Could not save pocket PDB file:[/red]", output_path)

        return output_path / "pocket.pdb"

    def add_h_and_convert_to_mol2(self, pdb_path: Path, output_path: Path) -> Path:
        """Original add_h_and_convert_to_mol2 logic here"""
        cmd.delete("all")
        cmd.load(str(pdb_path))
        cmd.h_add("sol")

        output_file = output_path / "pocket.mol2"
        cmd.save(str(output_file))
        return output_file

    def add_mol2_charges(self, pocket_mol2: Path, output_path: Path) -> Path:
        """Original add_mol2_charges logic here"""
        r = requests.post(
            "https://acc2-api.biodata.ceitec.cz/send_files",
            files={"file[]": open(pocket_mol2, "rb")},
        )

        r_id = list(r.json()["structure_ids"].values())[0]

        r_out = requests.get(
            f"https://acc2-api.biodata.ceitec.cz/calculate_charges?structure_id={r_id}&method=eqeq&generate_mol2=true"
        )

        output_file = output_path / "charged_pocket.mol2"
        output_file.write_bytes(r_out.content)
        return output_file
