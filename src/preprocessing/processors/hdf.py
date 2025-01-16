import pickle
from pathlib import Path
import h5py
from src.preprocessing.config import PDBBindDataset
from preprocessing.processors.protein import PocketProcessor
import openbabel

import numpy as np

import pybel
from src.preprocessing.processors.featurizer import Featurizer

featurizer = Featurizer()


def parse_mol_vdw(mol, element_dict):
    vdw_radii = []
    for atom in mol.atoms:
        element = atom.type
        vdw_radii.append(element_dict.get(element, 1.0))  # Default to 1.0 if not found
    return vdw_radii


# Create a new HDF5 file to store all of the data
output_total_hdf = Path("/path/to/output/data.h5")
with h5py.File(output_total_hdf, "a") as f:
    pocket_generator = __get_pocket()  # Assuming __get_pocket() is defined elsewhere

    for complex in dataset.complexes.values():
        pdbid = complex.pdb_id

        # Avoid duplicates
        if pdbid in f.keys():
            continue

        # Read ligand file using pybel
        ligand_path = complex.ligand_mol2
        ligand = next(openbabel.pybel.readfile("mol2", ligand_path))

        # Extract features from pocket and check for unrealistic charges
        pocket_coords, pocket_features, pocket_vdw = next(pocket_generator)

        # Extract features from ligand and check for unrealistic charges
        ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
        ligand_vdw = parse_mol_vdw(mol=ligand, element_dict=element_dict)
        if high_charge(ligand):
            if pdbid not in bad_complexes:
                bad_complexes.append(pdbid)

        # Center the ligand and pocket coordinates
        centroid = ligand_coords.mean(axis=0)
        ligand_coords -= centroid
        pocket_coords -= centroid

        # Assemble the features into one large numpy array: rows are heavy atoms, columns are coordinates and features
        data = np.concatenate(
            (
                np.concatenate((ligand_coords, pocket_coords)),
                np.concatenate((ligand_features, pocket_features)),
            ),
            axis=1,
        )

        # Concatenate van der Waals radii into one numpy array
        vdw_radii = np.concatenate((ligand_vdw, pocket_vdw))

        # Create a new dataset for this complex in the HDF5 file
        dataset = f.create_dataset(
            pdbid, data=data, shape=data.shape, dtype="float32", compression="lzf"
        )

        # Add the affinity and van der Waals radii as attributes for this dataset
        dataset.attrs["affinity"] = affinities_ind.loc[pdbid]
        assert len(vdw_radii) == data.shape[0]
        dataset.attrs["van_der_waals"] = vdw_radii
