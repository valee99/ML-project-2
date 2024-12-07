# Plankton Segmentation with Yolo11

## Project Setup

Install required packages with :
`pip install -r requirements.txt`

Set up a data folder as following : 
- `mkdir -p data`
- `mkdir -p data/data_raw`
- `mkdir -p data/data_labeled`
- `mkdir -p data/data_split`

In `data/data_raw` create the folder that will store the original .json and .tif as :
- `mkdir -p data/data_raw/geojson_file`
- `mkdir -p data/data_raw/images`

Copy your files to those folders:
- `cp -r path/to/geojson/files data/data_raw/geojson_file/`
- `cp -r path/to/tif/images data/data_raw/images/`


## Prepare Data

For each script, the arguments given are those used to obtain the best performances.

Load the raw data and format it for the task and divide it into two datasets, one for "small" objects and one for "big" objects:

- `python src/load_from_raw.py --path_geojson "./data/data_raw/geojson_file" --path_images "./data/data_raw/images" --path_output "./data/data_labeled" --min_contrast 0 --max_contrast 255 --min_surface 200 --task "seg"`
- `python src/load_from_raw.py --path_geojson "./data/data_raw/geojson_file" --path_images "./data/data_raw/images" --path_output "./data/data_labeled" --min_contrast 0 --max_contrast 255 --min_surface 0 --max_surface 200 --task "seg"`

Then create the dataset:

- `python src/create_dataset.py --path_labeled "./data/data_labeled/ctrst-0-255_srfc-200_prcs-0_seg" --path_split "./data/data_split" --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1 --all_slices`
- `python src/create_dataset.py --path_labeled "./data/data_labeled/ctrst-0-255_srfc-0-200_prcs-0_seg" --path_split "./data/data_split" --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1 --all_slices`

The argument `--all_slices` is used to save all slices from the images in the test set in order to make predictions on all the slice when visualising the results at the end. 

Create the patches needed for the small-objects model:

- `python src/create_patch.py --path_dataset "data/data_split/ctrst-0-255_srfc-0-200_prcs-0_seg" --task "seg" --n_rows_patch 8 --n_cols_patch 8`

Add data augmentations to the training set (needed for training only):

- `python src/augment_dataset.py --path_data_train "data/data_split/ctrst-0-255_srfc-0-200_prcs-0_seg/train" --task "seg"`
- `python src/augment_dataset.py --path_data_train "data/data_split/ctrst-0-255_srfc-200_prcs-0_seg/train" --task "seg"`

## Train Models

## Visualise Results

Go to `visualise.ipynb`and adapt the path to your data and models if needed.
The weights to the models are currently saved there:

- small-objects model : `runs/segmentation/yolo11n_ctrst-0-255_srfc-0-200_prcs-0_seg_patch_epochs-100_imgsz-190_batch-8/weights/best.pt`
- big-objects model : `runs/segmentation/yolo11n_ctrst-0-255_srfc-200_prcs-0_seg_epochs-100_imgsz-1520_batch-8/weights/best.pt`