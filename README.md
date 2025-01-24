# Non-Invasive Blood Count Using Deep Learning:  
**Hemoglobin and Red Blood Cell Estimation from Bulbar Conjunctiva Videos**

Blood tests are crucial for diagnosing and monitoring various health conditions but remain invasive, time-consuming, and costly. They require venous blood draws, skilled personnel, and sterile conditions, contributing to patient discomfort, delays in results, and increased healthcare costs. We present a non-invasive method for estimating red blood cell count and hemoglobin levels, combining bulbar conjunctiva vascular flow data with advanced machine learning techniques. In our study, we collected high-magnification videos of bulbar conjunctiva capillaries from 224 patients and paired them with laboratory blood counts of samples taken on the same day.

Our approach transforms unstable, high-dimensional videos into a low-dimensional spatiotemporal vessel dataset, utilizing pre-trained deep models for stabilization and segmentation, thereby enabling new avenues for medical research, diagnostics, and telemedicine approaches. Leveraging this new vessel representation, we developed a deep-learning model to estimate hemoglobin levels or red blood cell counts via regression, using multiple vessel instances from each patient.  

This approach, which holds the potential for non-invasive blood testing, presents a significant advance toward point-of-care diagnostics and could transform healthcare by enabling rapid, cost-effective diagnostics without the need for invasive blood draws, improving accessibility and reducing delays in diagnosis and treatment.

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

3. **Verify** that PyTorch sees your GPU (if applicable):
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   A `True` output means CUDA support is active.

---

## Repository Structure

```
VesselNet/
│
├── source/
│   ├── test_vessels.py          # Testing script
│   ├── train_vessels.py         # Training script
│   ├── utils/
│   │   ├── datasets_utils.py    # Dataset and transform utilities
│   │   ├── testing_utils.py     # Testing, plotting, metrics
│   │   ├── training_utils.py    # Training loops, data splits
│   │   ├── vessels_models/      # Model definitions
│   │   └── sampling_utils.py    # Custom samplers and binning
│   └── ...
├── Models/                      # Pretrained or saved model checkpoints
├── Results/                     # Output directory for results, figures
├── data/                        # Data folder for raw/preprocessed samples
├── README.md                    # This readme
└── requirements.txt             # Python dependencies
```

---

## Usage

### Testing a Pretrained Model

After training (or downloading) a **pretrained model**, you can test it on vessel data with **`test_vessels.py`**.

#### Example: Test on HGB

```bash
python3 ./source/test_vessels.py \
    --stem_kernel 2 \
    --stem_stride 2 \
    --block_kernel 7 \
    --use_thickness \
    --random_crop 80 80 \
    --dropout_p 0 \
    --patients_info_xlsx_path ./VesselNet/Patients_Info.xlsx \
    --test_graphs_dir ./Results/HGB_thin_vessels \
    --load_existing_samples ./Models/HGB \
    --load_existing_data_split ./Models/HGB \
    --save_path ./VesselNet/Models/HGB \
    --bag_size 16 \
    --gt_key HGB
```

#### Example: Test on RBC

```bash
python3 ./source/test_vessels.py \
    --stem_kernel 2 \
    --stem_stride 2 \
    --block_kernel 7 \
    --use_thickness \
    --random_crop 80 80 \
    --dropout_p 0 \
    --patients_info_xlsx_path ./VesselNet/Patients_Info.xlsx \
    --test_graphs_dir ./Results/RBC_thin_vessels \
    --load_existing_samples ./Models/RBC \
    --load_existing_data_split ./Models/RBC \
    --save_path ./VesselNet/Models/RBC \
    --bag_size 16 \
    --gt_key RBC
```

These commands generate regression plots, Bland-Altman plots, ROC curves, and metrics in the specified `--test_graphs_dir`. Results are saved in CSV and JSON formats for analysis.

---

### Training a Model

To train a new model (or multiple cross-validation folds), use **`train_vessels.py`**. Example command:

```bash
/home/tamirdenis/miniconda3/envs/ocd_transcrowd/bin/python3 /home/tamirdenis/projects/VesselsNet/source/train_vessels.py \
    --gt_key RBC \
    --train_val_test_split 0.8 0.1 0.1 \
    --vessels_dataset_path ./VesselNet/samples_per_patients.json \
    --load_existing_data_split /mnt/data1/Veye_learning/save_file/V30_60_vessels_RBC_bins_sample_cnn_mil_mlp_reg_v760_20 \
    --use_thickness \
    --random_crop 80 80 \
    --min_vessel_thickness 0 \
    --max_vessel_thickness 8 \
    --dropout_p 0.0 \
    --min_num_vessels 8 \
    --stem_kernel 2 \
    --stem_stride 2 \
    --block_kernel 7 \
    --number_of_experiments 20 \
    --batch_size 32 \
    --bag_size 16 \
    --save_path /mnt/data1/Veye_learning/save_file/V30_70_vessels_RBC_bins_sample_cnn_mil_mlp_reg_thin_paper_final \
    --patients_info_xlsx_path ./VesselNet/Patients_Info.xlsx \
    --sampler_bins 0 3.8 5.2 400 \
    --lr 0.0001 \
    --weight_decay 0.0 \
    --epoch_size 50 \
    --vessels_length 80 \
    --epochs 30
```

**Key details**:
- `--vessels_dataset_path` points to a JSON file of your vessel samples (unless you have `--load_existing_samples`).
- `--train_val_test_split 0.8 0.1 0.1` splits the data if no existing split is provided.
- `--save_path` is where the logs, checkpoints, and results will be stored.
- `--number_of_experiments 20` indicates cross-validation folds or multiple training runs.
- After training, look for `model_best.pth` in your save directory.

---

## Figures & Results

<p align="center">
  <img src="docs/figures/figure_method_placeholder.png" width="350" />
</p>
<p align="center"><b>Figure 1:</b> Placeholder for the pipeline from bulbar conjunctiva video to a predicted blood marker.</p>

<p align="center">
  <img src="docs/figures/figure_results_placeholder.png" width="350" />
</p>
<p align="center"><b>Figure 2:</b> Placeholder for main results and metrics (ROC, regression plots, etc.).</p>

---

## Citation

If you use this codebase in your research, please cite:

```
@article{Denis2023noninvasive,
  title={Non-Invasive Blood Count Using Deep Learning: Hemoglobin and Red Blood Cell Estimation from Bulbar Conjunctiva Videos},
  author={
    Denis, Tamir and Sher, Ifat and Prasiman, Emily and
    Haiadry, Marian and Zag, Amir and Benjamini, Ohad and
    Avigdor, Abraham and Asraf, Keren and Doolman, Ram and
    Wolf, Lior and Suchowski, Haim and Rotenstreich, Ygal
  },
  journal={Nature Biomedical Engineering (Under Review)},
  year={2025}
}
```

---

## License

This repository is open-sourced under the [MIT License](./LICENSE). Please see the `LICENSE` file for more details. Contributions, feedback, and pull requests are welcome—feel free to open an issue or contact us!