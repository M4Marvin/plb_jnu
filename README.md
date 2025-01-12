# Deep Learning Framework for Protein-Ligand Binding Affinity Prediction

## Overview

This research project implements a novel deep learning architecture combining 3D Convolutional Neural Networks (CNNs) and Graph Convolutional Networks (GCN) to predict protein-ligand binding affinities. The model processes structural and chemical information through parallel networks before combining outputs for final prediction.

## Architecture

- Dual 3D-CNN branches processing voxelized molecular representations (48×48×48×19)
- Graph Convolutional Network (GCN) for molecular topology
- Multi-Layer Perceptron (MLP) for final affinity prediction

## Dataset

- Based on the PDBbind dataset
- Preprocessed molecular structures with computed charges (MOL2 format)
- Voxelized representations for CNN input

## Requirements

```python
torch>=1.9.0
numpy
pandas
h5py
biopandas
```

## Project Structure

``` bash
├── LICENSE
├── README.md
├── data
│   ├── pdb_bind
│   │   ├── refined-set              # Refined Set Data
│   │   └── v2020-other-PL          # General Set
│   └── sample_data
├── notebooks
└── src
    ├── models
    ├── preprocessing
    └── training
```

## Setup

- to be added.

## Usage

- to ber added.

<!-- ## Citation
[Add your paper details once published] -->

## License

MIT

## Contact

[Your Name]
[Your Institution]
[Your Email]
