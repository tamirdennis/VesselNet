# Towards non-invasive blood count using a deep learning pipeline from bulbar conjunctiva videos:  
**Hemoglobin and Red Blood Cell Estimation from Bulbar Conjunctiva Videos**


This repository contains the codebase for the paper "Towards non-invasive blood count using a deep learning pipeline from bulbar conjunctiva videos" (2026, NPJ Digital Medicine).
The codebase includes the training and testing scripts for the VesselNet model,
which estimates the hemoglobin and RBC blood markers from a prepared dataset of bulbar conjunctiva vessels made by our "Videos to Vessels" pipeline. 

---

## Table of Contents
1. [Installation](#installation)  
2. [Repository Structure](#repository-structure)  
3. [Usage](#usage)  
   - [Testing a Pretrained Model](#testing-a-pretrained-model)  
     - [Example: Test on HGB](#example-test-on-hgb)  
     - [Example: Test on RBC](#example-test-on-rbc)  
   - [Training a Model](#training-a-model)  
4. [Figures & Results](#figures--results)  
5. [Citation](#citation)  
6. [License](#license)

---

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/tamirdennis/VesselNet.git
   cd VesselNet
   ```

2. **Create a virtual environment** and install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
   Make sure to install the correct PyTorch build for your hardware (CUDA vs. CPU).

---

## Repository Structure
We share our full Vessels dataset, patient information, and trained models in the following [DOI link](https://doi.org/10.6084/m9.figshare.28280930). 
To reproduce the results, download and place it in the repository's root directory.
In order to run the given scripts, the repository should have the following structure:
```
VesselNet/
│
├── source/
│   ├── test_vessels.py                 # Testing script
│   ├── train_vessels.py                # Training script
│   ├── utils/
│   │   ├── datasets_utils.py           # Dataset and transform utilities
│   │   ├── testing_utils.py            # Testing, plotting, metrics
│   │   ├── training_utils.py           # Training loops, data splits
│   │   ├── vessels_models/             # Model definitions
│   │   └── sampling_utils.py           # Custom samplers and binning
│   └── ...
├── Models/                             # Pretrained or saved model checkpoints
│   ├── RBC/                            # RBC model checkpoints
│   ├── HGB/                            # Hemoglobin model checkpoints
├── Results/                            # Output directory for results, figures
├── Vessels_Dataset/                    # Data folder for raw/preprocessed samples
│   ├── samples_per_patients.json       # Full vessels samples per patient dataset
│   ├── samples_per_patients_thin.json  # Filtered thin vessels samples per patient dataset
├── NHANES/ (optional)                  # Optional folder for population-level analyses
│   ├── DEMO_L.xpt
│   ├── CBC_L.xpt
│   ├── DEMO_P.xpt
│   ├── CBC_P.xpt
├── README.md                           # This readme
├── requirements.txt                    # Python dependencies
└── Patients_Info.xlsx                  # Excel file with patient information.
```

### NHANES Files (Optional)
To enable population-level statistical analysis, download the following files from the official NHANES website:

- DEMO_L.xpt  
- CBC_L.xpt  
- DEMO_P.xpt  
- CBC_P.xpt  

Download at: https://wwwn.cdc.gov/nchs/nhanes/

Place all four files inside the `NHANES/` folder.  
If `--nhanes_folder_path` is left as `None` (default), the analysis runs normally **without** population statistics.

---

## Usage

### Testing a Trained Model

You can test a trained model on vessels data with **`test_vessels.py`**.

#### Example: Test on HGB
Assuming the repository structure explained above and that the Models/HGB folders is the save_path of train_vessels.py HGB training, you can run the following command to test the pretrained model on the HGB blood marker:

```bash
python3 ./source/test_vessels.py \
    --stem_kernel 2 \
    --stem_stride 2 \
    --block_kernel 7 \
    --use_thickness \
    --patients_info_xlsx_path ./Patients_Info.xlsx \
    --test_graphs_dir ./Results/HGB_thin_vessels \
    --load_existing_samples ./Vessels_Dataset/samples_per_patients_thin.json \
    --load_existing_data_split ./Models/HGB \
    --models_path ./Models/HGB \
    --bag_size 16 \
    --gt_key HGB \
    --low_threshold 11 \
    --females_threshold 12 \
    --males_threshold 13.5 \
    --nhanes_folder_path ./NHANES
```

Note:  
- The `--nhanes_folder_path` argument is **optional**.  
- If you wish to test the model **without** NHANES statistical analysis, simply remove this argument (default = None).

#### Example: Test on RBC
Similarly, you can test the pretrained model on the RBC blood marker:
```bash
python3 ./source/test_vessels.py \
    --stem_kernel 2 \
    --stem_stride 2 \
    --block_kernel 7 \
    --use_thickness \
    --patients_info_xlsx_path ./Patients_Info.xlsx \
    --test_graphs_dir ./Results/RBC_thin_vessels \
    --load_existing_samples ./Vessels_Dataset/samples_per_patients_thin.json \
    --load_existing_data_split ./Models/RBC \
    --models_path ./Models/RBC \
    --bag_size 16 \
    --gt_key RBC \
    --low_threshold 3.8 \
    --females_threshold 3.8 \
    --males_threshold 4.4
```

These commands generate regression plots, Bland-Altman plots, ROC curves, and metrics in the specified `--test_graphs_dir`. Results are saved in CSV and JSON formats for analysis.

---

### Training a Model

To train multiple cross-validation folds, use **`train_vessels.py`**. Example command (note that this will override the file in output_filtered_samples_path and the trained models in save_path):

```bash
/home/tamirdenis/miniconda3/envs/ocd_transcrowd/bin/python3 /home/tamirdenis/projects/VesselsNet/source/train_vessels.py \
    --gt_key HGB \
    --train_val_test_split 0.8 0.1 0.1 \
    --full_vessels_dataset_path ./Vessels_Dataset/samples_per_patients.json \
    --output_filtered_samples_path ./Vessels_Dataset/samples_per_patients_thin.json \
    --load_existing_data_split ./Models/HGB \
    --use_thickness \
    --min_vessel_thickness 0 \
    --max_vessel_thickness 8 \
    --min_num_vessels 8 \
    --stem_kernel 2 \
    --stem_stride 2 \
    --block_kernel 7 \
    --number_of_experiments 20 \
    --batch_size 32 \
    --bag_size 16 \
    --save_path ./Models/HGB \
    --patients_info_xlsx_path ./Patients_Info.xlsx \
    --sampler_bins 0 10.5 14 400 \
    --lr 0.0001
```

**Key details**:
- `--full_vessels_dataset_path` points to a JSON file of your full vessel samples (unless you have `--load_existing_samples`).
- `--save_path` is where checkpoints, and normalizing files will be stored per fold run.
- `--number_of_experiments 20` indicates cross-validation folds or multiple training runs.
- After training, look for `model_best.pth` in every fold run in your save directory.

---

## Figures & Results

<p align="center">
  <img src="figures/Abstract.png" width="350" />
</p>
<p align="center">Abstract Figure.</p>

<p align="center">
  <img src="figures/Regression Results.png" width="350" />
</p>
<p align="center">Main Results.</p>

---

## Citation

If you use this codebase in your research, please cite:

```
@article{denis2026towards,
  title={Towards Non-Invasive Blood Count Using a Deep Learning Pipeline from Bulbar Conjunctiva Videos},
  author={Denis, Tamir and Sher, Ifat and Praisman, Emily and Haiadry, Marian and Zag, Amir and Benjamini, Ohad and Avigdor, Abraham and Asraf, Keren and Doolman, Ram and Wolf, Lior and Suchowski, Haim and Rotenstreich, Ygal},
  journal={npj Digital Medicine},
  year={2026},
  volume={...},
  number={...},
  pages={...},
  doi={...}
}
```

---

## License

This repository is open-sourced under the [MIT License](./LICENSE). Please see the `LICENSE` file for more details. Contributions, feedback, and pull requests are welcome—feel free to open an issue or contact us!
