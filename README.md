# KUL image reconstruction teaching notebooks

Collection of ipython notebooks used for teaching. 

## Running the notebooks

### Option 1: online using binder (recommended)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KUL-recon-lab/teaching_notebooks/HEAD)

You can run all notebooks online without installing python on your local machine by clicking on the
link to Binder above.


### Option 1: local installation

1. Download and install conda [e.g. here](https://github.com/conda-forge/miniforge) and git [e.g. here](https://git-scm.com/downloads)
2. Clone this repository 
```
git clone https://github.com/KUL-recon-lab/teaching_notebooks.git
```
or download a zipped version by clicking "code -> download Zip"
3. Create a conda environment "teaching_notebooks" with all packages needed
```
cd teaching_notebooks
conda env create -f environment.yaml
```
4. Acticate the conda environment
```
conda activate teaching_notebooks
```
5. Start your notebook
```
cd MC_MLEM_noise_transfer
jupyter notebook MC_MLEM_noise_transfer.ipynb
```
