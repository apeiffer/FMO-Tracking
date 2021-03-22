# FMO-Tracking
This Computer Vision Research Project is based on the RVOS code, which is a temporal and spatial machine learning method used to segment videos. The original code can be found on the following repository: https://github.com/imatge-upc/rvos/tree/12016db67a18e4c4dea3f42f5b3c0b2623761c5b.

## Installation Notes
* The `rvos` subfolder is just the existing RVOS code we have cloned - it is there so we can synchronize the files that are needed in the `RVOS_FMO.ipynb` Colab notebook. This is necessary because Google Colab is mounted on each of our own Google Drives, so the files used by the notebook have to be on each of our accounts if we want to run the notebook.
* The `databases` subfolder is created by following the instructions in `rvos/README.md` with the DAVIS dataset.
* As such, the RVOS subfolder must be placed at `My Drive/FMO-Tracking` in each of our Google Drive workspaces.
