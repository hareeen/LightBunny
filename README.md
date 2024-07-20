# LightBunny: Deep Learning based Lightweight EEG Source Imaging

LightBunny is an EEG source imaging model aiming at providing an estimation of the location, size, and temporal activity of the brain activities from scalp EEG/MEG recordings.

\[KOSOMBE 2024 Spring\] Suyoung Hwang, Jinseok Eo, Wooyong Lee, Hae-Jeong Park: "Electrophysical Source Imaging based on the Spatiotemporal Neural Network"

This work is supported by the National Research Foundation (NRF) funded by the Korean government. (No. 2022MSE5E8018285)

<details>

<summary>Overview of Network Architecture and Results</summary>

### Network Architecture

![image](https://github.com/user-attachments/assets/67c8db8f-cdfb-49f7-8040-625999afbbdc)

Parameter count: 5.06M

### Results

Spatial PCC: 0.85 ± 0.05<br/>
Temporal PCC: 0.98 ± 0.03

![image](https://github.com/user-attachments/assets/a9050ef6-a0c2-4814-8064-1e476bddd121)

**(A, B)** Ground Truth (A) and Estimated (B) Source of a Generated Sample Test Data<br/>
**(C)** Ground Truth and Estimated Potential of a Source Region of a Generated Sample Test Data<br/>
**(D)** Estimated Source of the Auditory-Evoked Potential Elicited in the Auditory Oddball Task

</details>

## Requirements

All python requirements can be installed with [Poetry](https://python-poetry.org/).

To set up the environment:
```sh
poetry install
```

To activate the environment:
```sh
poetry shell
```

Also requires MATLAB with appropriate toolbox installed.

## Train Data Generation

### TVB Simulation and Postprocessing

```sh
mkdir source
cd forward
./simulate.sh 0 994
```

`forward/simulate.sh` runs the simulation for regions within provided region range; divide the regions into groups of 16 and perform them in parallel
Since raw NMM simulation results are very large; the script post-processes the results at the end of each group's simulation.
During post-processing, the script automatically runs MATLAB in batch mode.

### Prepare Training/Testing Dataset

```sh
matlab -batch generate_sythetic_source
```

After running the MATLAB script, follow instructions in `prep_metadata.ipynb`.

To switch between training/testing dataset, modify train flag in `forward/generate_sythetic_source.m` and corresponding filenames in `prep_metadata.ipynb`

## Forward Calculation

Follow instructions in `prep_leadfield.ipynb`.

By default, the script calculates the forward solution to 64-channel EEF on a standard 10-20 system based on the fsaverage template, but this can be modified to other templates or other systems.

## Training & Evaluation

Follow instructions in `main.ipynb`.

Evaluation with real data (.edf, etc.) requires realigning that data to the channel orientation of the forward solution used to train the network. The scripts in `prep_edf.ipynb` can help with this task.
