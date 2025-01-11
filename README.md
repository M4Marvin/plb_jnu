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
├── data/
│   ├── processed/          # Processed mol2 files and HDF5 datasets
│   └── raw/               # Original PDBbind data
├── src/
│   ├── models/            # Neural network architectures
│   ├── preprocessing/     # Data processing scripts
│   └── training/         # Training and evaluation scripts
├── notebooks/            # Analysis and visualization notebooks
└── results/             # Experimental results and model checkpoints
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
