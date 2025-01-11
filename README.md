# Machine Learning for Page fault prediction

This repository contains the work surrounding the semester project around ML for page prefetching.

## Project Structure

```
.
├── data/               # Data collection and storage
│   ├── analysis/      # Analysis notebooks and scripts
│   │   ├── helpers/       # Helper scripts for data collection
│   └── data/  # Where the acual data lies
├── models/            # Model implementation of NLinear
├── prediction/        # Prediction-related code and experiments
│   ├── gpu_scripts/  # Scripts and dockerfiles to run on RCP
│   └── predictions.ipynb  # Notebook parsing our results
├── tests/            # Tests to test metric parrallelism
└── utils/            # Utility functions and helpers
```

## Trace collection.
Trace collecion is thoroughly explained [here](data/data/fltrace.md)

## Requirements

See `requirements.txt` for a complete list of python dependencies.

## Models

The implementation of Nlinear and the scripts used for GPU running are found under models/ directory.

## Analysis Tools

All analysis work is under data/analysis/ folder

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run analysis scripts from the analysis directory:
```bash
cd data/analysis
jupyter notebook
```