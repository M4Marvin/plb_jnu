import os
import pickle

import numpy as np
import openbabel.pybel
import pandas as pd
import plotly.express as px
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from rich.jupyter import print
from scipy.spatial import cKDTree


def create_cleaned_dataset(
    PDBbind_dataset_path,
    general_set_PDBs_path,
    refined_set_PDBs_path,
    output_name,
    plot=False,
    verbose=True,
):
    """
    Produces a csv file containing PDB id, binding affinity, and set (general/refined)

    Inputs:
    1) PDBbind_dataset_path: path to PDBbind dataset; dataset is included in github repository as 'PDBbind_2020_data.csv'
    2) general_set_PDBs_path: path to PDBbind general set excluding refined set PDBs
    3) refined_set_PDBs_path: path to PDBbind refined set PDBs
    4) output_name: name for the output csv file. Must end in .csv
    5) plot = True will generate a plot of density as a function of binding affinity for general
    and refined sets

    Output:
    1) A cleaned csv containing PDB id, binding affinity, and set (general/refined):
    'output_name.csv'
    """
    # Load dataset
    data = pd.read_csv(PDBbind_dataset_path)
    if verbose:
        print(f"Loaded dataset from {PDBbind_dataset_path} with {len(data)} entries.")

    # Check for NaNs in affinity data
    if data["-log(Kd/Ki)"].isnull().any():
        if verbose:
            print("There are NaNs present in affinity data!")

    # Efficiently check for missing structural data
    general_pdb_set = set(os.listdir(general_set_PDBs_path))
    refined_pdb_set = set(os.listdir(refined_set_PDBs_path))
    if verbose:
        print(
            f"Found {len(general_pdb_set)} PDBs in general set, {len(refined_pdb_set)} in refined set."
        )

    data["set"] = data["PDB ID"].apply(
        lambda x: (
            "general"
            if x in general_pdb_set
            else ("refined" if x in refined_pdb_set else np.nan)
        )
    )
    initial_length = len(data)
    data.dropna(subset=["set"], inplace=True)
    if verbose:
        print(f"Removed {initial_length - len(data)} entries without structural data.")

    # Write out csv of cleaned dataset
    data[["PDB ID", "-log(Kd/Ki)", "set"]].to_csv(output_name, index=False)
    if verbose:
        print(f"Cleaned dataset written to {output_name} with {len(data)} entries.")

    # Plot if required
    if plot:
        fig = px.histogram(
            data,
            x="-log(Kd/Ki)",
            color="set",
            barmode="overlay",
            histnorm="density",
            title="Density of Binding Affinity by Set",
        )
        fig.update_layout(xaxis_title="-log(Kd/Ki)", yaxis_title="Density")
        fig.show()
        if verbose:
            print(
                "Interactive plot generated showing the density of binding affinity by set."
            )


def extract_pocket(
    protein_pdb,
    ligand_mol2,
    distance_threshold=10.0,
    output_path=None,
    include_heteroatoms=False,
    verbose=False,
):
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
    protein_atom_tree = cKDTree(protein_atom[["x_coord", "y_coord", "z_coord"]].values)
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
        print("[grey]Pocket atoms and heteroatoms assigned to the PDB object.[/grey]")

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

    return pred_pocket
