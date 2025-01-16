import numpy as np
from biopandas.mol2 import PandasMol2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from math import ceil
import warnings


@dataclass
class FeaturizerConfig:
    """Configuration for molecular featurization"""

    atom_codes: Optional[Dict[str, int]] = (
        None  # Changed to str keys for MOL2 atom types
    )
    atom_labels: Optional[List[str]] = None
    save_molecule_codes: bool = True
    custom_properties: Optional[List[callable]] = None


class Featurizer:
    """Calculates atomic features for MOL2 format molecules using BioPandas.
    Features encode atom type and properties for machine learning applications.
    """

    def __init__(self, config: Optional[FeaturizerConfig] = None):
        """Initialize featurizer with given configuration

        Parameters
        ----------
        config : FeaturizerConfig, optional
            Configuration for featurization. If None, uses default settings.
        """
        if config is None:
            config = FeaturizerConfig()

        self.FEATURE_NAMES = []

        # Setup atom codes
        if config.atom_codes is not None:
            self._validate_atom_codes(config.atom_codes)
            self.ATOM_CODES = config.atom_codes
            self.NUM_ATOM_CLASSES = len(set(config.atom_codes.values()))

            # Setup atom labels
            if config.atom_labels:
                if len(config.atom_labels) != self.NUM_ATOM_CLASSES:
                    raise ValueError(
                        f"Expected {self.NUM_ATOM_CLASSES} atom labels, got {len(config.atom_labels)}"
                    )
                self.FEATURE_NAMES.extend(config.atom_labels)
            else:
                self.FEATURE_NAMES.extend(
                    [f"atom{i}" for i in range(self.NUM_ATOM_CLASSES)]
                )
        else:
            self._setup_default_atom_codes()

        self.save_molecule_codes = config.save_molecule_codes
        if self.save_molecule_codes:
            self.FEATURE_NAMES.append("molcode")

        # Setup custom property calculators
        self.custom_properties = config.custom_properties or []
        for i, func in enumerate(self.custom_properties):
            name = getattr(func, "__name__", f"func{i}")
            self.FEATURE_NAMES.append(name)

    def _validate_atom_codes(self, atom_codes: Dict[str, int]) -> None:
        """Validate atom codes dictionary"""
        if not isinstance(atom_codes, dict):
            raise TypeError(f"Atom codes should be dict, got {type(atom_codes)}")

        codes = set(atom_codes.values())
        for i in range(len(codes)):
            if i not in codes:
                raise ValueError(f"Missing atom code {i}")

    def _setup_default_atom_codes(self) -> None:
        """Setup default atom encoding scheme for common MOL2 atom types"""
        self.ATOM_CODES = {
            "C.3": 0,  # sp3 carbon
            "C.2": 0,  # sp2 carbon
            "C.1": 0,  # sp carbon
            "C.ar": 0,  # aromatic carbon
            "N.3": 1,  # sp3 nitrogen
            "N.2": 1,  # sp2 nitrogen
            "N.1": 1,  # sp nitrogen
            "N.ar": 1,  # aromatic nitrogen
            "O.3": 2,  # sp3 oxygen
            "O.2": 2,  # sp2 oxygen
            "O.co2": 2,  # carboxylate oxygen
            "S.3": 3,  # sp3 sulfur
            "S.2": 3,  # sp2 sulfur
            "P.3": 4,  # sp3 phosphorus
            "F": 5,  # fluorine
            "Cl": 5,  # chlorine
            "Br": 5,  # bromine
            "I": 5,  # iodine
        }

        self.FEATURE_NAMES.extend(
            ["carbon", "nitrogen", "oxygen", "sulfur", "phosphorus", "halogen"]
        )

        self.NUM_ATOM_CLASSES = 6  # Number of unique atom types

    def encode_atom(self, atom_type: str) -> np.ndarray:
        """Encode atom type with a binary vector

        Parameters
        ----------
        atom_type : str
            MOL2 atom type (e.g., 'C.3', 'O.2', etc.)

        Returns
        -------
        np.ndarray
            One-hot encoding of atom type
        """
        encoding = np.zeros(self.NUM_ATOM_CLASSES)
        try:
            encoding[self.ATOM_CODES[atom_type]] = 1.0
        except KeyError:
            pass  # Return zero vector for unknown atoms
        return encoding

    def get_features(
        self, mol: PandasMol2, molcode: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract coordinates and features from MOL2 molecule

        Parameters
        ----------
        mol : PandasMol2
            Molecule to featurize
        molcode : float, optional
            Code to identify molecule type (e.g., 1.0 for ligand, -1.0 for protein)

        Returns
        -------
        coords : np.ndarray
            Atomic coordinates, shape (N, 3)
        features : np.ndarray
            Atomic features, shape (N, F)
        """
        if molcode is None and self.save_molecule_codes:
            raise ValueError("molcode required when save_molecule_codes is True")

        # Get heavy atoms only
        atoms_df = mol.df[~mol.df["atom_type"].str.startswith("H")]

        if len(atoms_df) == 0:
            raise ValueError("No heavy atoms found in molecule")

        # Extract coordinates
        coords = atoms_df[["x", "y", "z"]].values.astype(np.float32)

        # Build feature matrix
        features = []
        for _, atom in atoms_df.iterrows():
            atom_features = [
                *self.encode_atom(atom["atom_type"]),
                *[prop(atom) for prop in self.custom_properties],
            ]
            features.append(atom_features)

        features = np.array(features, dtype=np.float32)

        if self.save_molecule_codes:
            features = np.hstack([features, molcode * np.ones((len(features), 1))])

        if np.isnan(features).any():
            raise RuntimeError("NaN values found in features")

        return coords, features


def make_grid(
    coords: np.ndarray,
    features: np.ndarray,
    grid_resolution: float = 1.0,
    max_dist: float = 10.0,
) -> np.ndarray:
    """Convert atomic coordinates and features into a 3D grid representation

    Parameters
    ----------
    coords : np.ndarray
        Atomic coordinates, shape (N, 3)
    features : np.ndarray
        Atomic features, shape (N, F)
    grid_resolution : float
        Size of grid cells in Angstroms
    max_dist : float
        Maximum distance from center of grid

    Returns
    -------
    np.ndarray
        4D array of features distributed in 3D space
        Shape: (1, M, M, M, F) where M = 2 * max_dist / grid_resolution + 1
    """
    # Validate inputs
    if not isinstance(coords, np.ndarray) or coords.shape[1] != 3:
        raise ValueError("coords must be array of shape (N, 3)")
    if not isinstance(features, np.ndarray) or len(coords) != len(features):
        raise ValueError("features must be array of shape (N, F)")

    num_features = features.shape[1]
    box_size = ceil(2 * max_dist / grid_resolution + 1)

    # Estimate memory usage
    grid_size = box_size**3 * num_features * 4  # 4 bytes per float32
    if grid_size > 1e9:  # 1 GB
        warnings.warn(f"Grid will use approximately {grid_size / 1e9:.1f}GB of memory")

    # Move atoms to nearest grid points
    grid_coords = (coords + max_dist) / grid_resolution
    grid_coords = grid_coords.round().astype(int)

    # Remove atoms outside box
    in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)

    # Create grid and add features
    grid = np.zeros((1, box_size, box_size, box_size, num_features), dtype=np.float32)
    for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
        grid[0, x, y, z] += f

    return grid
