# DMPS: Dominant-Mechanism Progressive Synthesis for Sparse Circularly Polarized Concentric Ring Arrays

## Overview

This repository provides the Python implementation of dominant-mechanism progressive synthesis (DMPS) for wide-angle beam synthesis of sparse circularly polarized (CP) concentric ring arrays.

DMPS follows a progressive design flow. An isolated element pattern (IEP) model supports efficient array geometry synthesis; ringwise physical rotations provide passive polarization preconditioning; and full-array vector active element patterns (Vec-AEPs) enable scan-dependent axial-ratio (AR) and peak sidelobe-level (PSLL) optimization under bounded amplitude and phase control.

## Code Structure

The `code/` directory contains 26 Python scripts. The three main entry points are:

- `run_geometric_backbone.py`: IEP-based sparse ring geometry synthesis.
- `run_passive_preconditioning.py`: ringwise passive rotation and pattern preconditioning.
- `run_vecaep_residual_closure.py`: bounded amplitude-phase scan-codebook synthesis using full-array Vec-AEPs.

Supporting scripts are organized into five directories:

- `element_spectrum/`: element-pattern preprocessing and azimuthal spectrum modeling.
- `geometric_backbone/`: geometry optimization and dominant-mode analysis.
- `passive_preconditioning/`: passive rotation strategies and pattern evaluation.
- `hfss_modeling/`: HFSS array modeling and far-field export.
- `vecaep_residual_closure/`: Vec-AEP processing and constrained convex optimization.

## Requirements and Usage

The code was developed using Python 3.10.19. Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

The HFSS scripts target Ansys Electronics Desktop 2024 R2. HFSS and MOSEK require separately configured licenses.

To inspect the input options of each main driver, run:

```bash
python code/run_geometric_backbone.py --help
python code/run_passive_preconditioning.py --help
python code/run_vecaep_residual_closure.py --help
```

Full synthesis requires the corresponding preprocessed IEP data, intermediate geometry/rotation results, and full-array HFSS/Vec-AEP exports. These inputs must be supplied or generated separately.

## Example

The `Example/` directory includes a 2.4 GHz left-hand circularly polarized (LHCP) cut-corner patch element:

- `cutcorner_lhcp_2p4ghz_unit.aedt`: HFSS element model.
- `cutcorner_lhcp_2p4ghz_unit_farfield.csv`: corresponding complex far-field export.

These files illustrate the element model and the far-field data format; they are not a complete full-array synthesis dataset.

## Citation

If you use this repository in your research, please cite the corresponding DMPS paper. The full bibliographic information will be added after publication.
