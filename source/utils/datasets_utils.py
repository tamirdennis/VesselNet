import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Normalize, RandomCrop
from torch.multiprocessing import set_start_method

print("cuda:")
print(torch.cuda.is_available())

try:
    set_start_method('spawn')
except RuntimeError:
    pass



def filter_vessels_by_args(vessels_info_d, args):
    """
    Filters the vessels dictionary in-place based on thickness thresholds.

    Args:
        vessels_info_d (dict):
            {patient_id: [list_of_vessel_dicts]}, each containing 'thickness' and 'vessel'.
        args (Namespace):
            Must contain min_vessel_thickness and max_vessel_thickness attributes.

    Returns:
        None. The original dictionary is modified.
    """
    for patient, vessels in vessels_info_d.items():
        passed_vessels = []
        for vessel_info in vessels:
            if args.min_vessel_thickness < vessel_info['thickness'] < args.max_vessel_thickness:
                passed_vessels.append(vessel_info)
        vessels_info_d[patient] = passed_vessels


class VesselsBagDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for storing vessel images and (optionally) thickness in "bags".

    Each item is either:
      - (vessel_images_bag, gt) in training mode, or
      - (vessel_images_bag, gt, patient_id) in evaluation mode,
      with an optional thickness tensor if use_thickness=True.
    """

    def __init__(self,
                 inds_to_patients,
                 vessels_samples_per_patient,
                 bag_size,
                 transforms_list=None,
                 train=False,
                 bags_per_patient=None,
                 random_crop=None,
                 vessel_images_mean_std=None,
                 vessel_thickness_mean_std=None,
                 use_thickness=False):
        """
        Initialize the VesselsBagDataset.

        Args:
            inds_to_patients (dict):
                A dictionary mapping an index to { 'patient': patient_id, 'gt': ground_truth }.
            vessels_samples_per_patient (dict):
                {patient_id: [list_of_vessel_samples]}.
            bag_size (int):
                Number of vessel samples in one "bag".
            transforms_list (list, optional):
                List of torchvision transforms to apply to the vessel images.
            train (bool, optional):
                If True, random bag sampling is performed each time __getitem__ is called.
                If False, a fixed set of random bags is pre-generated for each patient.
            bags_per_patient (int, optional):
                How many bags to generate per patient if train=False.
            random_crop (tuple or list, optional):
                If specified, uses RandomCrop of this size on each sample.
            vessel_images_mean_std (tuple, optional):
                (mean, std) for normalizing the vessel images.
            vessel_thickness_mean_std (tuple, optional):
                (mean, std) for normalizing thickness values.
            use_thickness (bool, optional):
                If True, includes thickness as part of the model input.
        """
        super(VesselsBagDataset, self).__init__()

        self.inds_to_patients = inds_to_patients
        self.vessels_samples_per_patient = vessels_samples_per_patient
        self.transforms_list = transforms_list if transforms_list is not None else []
        if random_crop is not None:
            self.transforms_list.append(RandomCrop(random_crop))
        if vessel_images_mean_std is not None:
            self.transforms_list.append(Normalize(vessel_images_mean_std[0], vessel_images_mean_std[1]))
        self.vessel_thickness_mean_std = vessel_thickness_mean_std
        self.transforms_list = transforms.Compose(self.transforms_list)
        self.train = train
        self.bag_size = bag_size

        num_vessels = 0
        for vessels_list in vessels_samples_per_patient.values():
            num_vessels += len(vessels_list)
        self.num_vessels = num_vessels // bag_size if (bags_per_patient is None or self.train) \
            else bags_per_patient * len(vessels_samples_per_patient)
        self.fixed_vessels_bags_inds_per_patient = {}
        self.curr_vessels_bags_per_patient = {}
        self.bags_per_patient = bags_per_patient
        self.use_thickness = use_thickness
        if not train:
            assert type(self.bags_per_patient) is int and self.bags_per_patient > 0
            self.fixed_vessels_bags_inds_per_patient = self._create_fixed_vessels_bags_per_patient()
            self.curr_vessels_bags_per_patient = {patient: 0 for patient in
                                                  self.fixed_vessels_bags_inds_per_patient.keys()}

    def _create_fixed_vessels_bags_per_patient(self):
        """
        Pre-generates fixed random "bags" of vessel samples for each patient.

        Returns:
            dict: {patient_id: [list_of_vessel_indices_arrays]}
                  Each list has length = bags_per_patient, and each array has size = bag_size.
        """
        returned = {}
        for patient, vessels_list in self.vessels_samples_per_patient.items():
            curr_patient_vessels_bags = []
            curr_vessels_inds = list(range(len(vessels_list)))
            for _ in range(self.bags_per_patient):
                vessels_bag_inds = np.random.choice(curr_vessels_inds, size=self.bag_size, replace=True)
                curr_patient_vessels_bags.append(vessels_bag_inds)
            returned[patient] = curr_patient_vessels_bags
        return returned

    def __getitem__(self, idx: int):
        """
        Fetch one bag of vessel samples.

        If train=True, a new random bag is sampled on each call.
        If train=False, we cycle through a predetermined list of bags for each patient.

        Args:
            idx (int): Index used to determine which patient and bag to load.

        Returns:
            tuple:
                - If train=True and use_thickness=False: (vessel_images_bag, gt)
                - If train=True and use_thickness=True: (vessel_images_bag, vessels_thickness_bag, gt)
                - If train=False and use_thickness=False: (vessel_images_bag, gt, patient_id)
                - If train=False and use_thickness=True: (vessel_images_bag, vessels_thickness_bag, gt, patient_id)
        """
        curr_patient = self.inds_to_patients[idx]['patient']
        curr_vessels = self.vessels_samples_per_patient[curr_patient]
        curr_vessels_inds = list(range(len(curr_vessels)))
        if self.train:
            vessels_bag_inds = np.random.choice(curr_vessels_inds, size=self.bag_size, replace=True)
        else:
            curr_patient_vessels_bags = self.fixed_vessels_bags_inds_per_patient[curr_patient]
            curr_patient_curr_bag = self.curr_vessels_bags_per_patient[curr_patient]
            vessels_bag_inds = curr_patient_vessels_bags[curr_patient_curr_bag]
            self.curr_vessels_bags_per_patient[curr_patient] = \
                (self.curr_vessels_bags_per_patient[curr_patient] + 1) % self.bags_per_patient

        vessels_bag = np.array([
            curr_vessels[ind]['vessel'] if type(curr_vessels[ind]) == dict else curr_vessels[ind]
            for ind in vessels_bag_inds
        ])
        if self.use_thickness:
            vessels_thickness_bag = np.array([curr_vessels[ind]['thickness'] for ind in vessels_bag_inds])
            vessels_thickness_bag = torch.Tensor(vessels_thickness_bag)
            if self.vessel_thickness_mean_std is not None:
                vessels_thickness_bag = (
                    vessels_thickness_bag - self.vessel_thickness_mean_std[0]
                ) / self.vessel_thickness_mean_std[1]
        else:
            vessels_thickness_bag = None

        vessels_bag = torch.Tensor(vessels_bag)
        vessels_bag = self.transforms_list(vessels_bag)
        curr_gt = self.inds_to_patients[idx]['gt']

        if self.train:
            if vessels_thickness_bag is not None:
                return vessels_bag, vessels_thickness_bag, curr_gt
            return vessels_bag, curr_gt
        else:
            if vessels_thickness_bag is not None:
                return vessels_bag, vessels_thickness_bag, curr_gt, curr_patient
            return vessels_bag, curr_gt, curr_patient

    def __len__(self):
        """
        Returns the total number of "bags" in this dataset, depending on train/bags_per_patient logic.

        Returns:
            int: Number of available "bags".
        """
        return self.num_vessels


def load_vessels_samples_from_json(save_path):
    """
    Loads vessels samples from JSON file in a given directory.

    Args:
        save_path (str): Path to the directory containing 'samples_per_patients.json'.

    Returns:
            - vessels_samples_per_patient (dict): {patient_id: [vessel_samples]}
    """
    samples_per_patients_path = Path(save_path) / 'samples_per_patients.json'
    with open(samples_per_patients_path.as_posix(), 'r') as f:
        samples_per_patients_json = json.load(f)
    vessels_samples_per_patient = {}
    for p, vessels in samples_per_patients_json.items():
        if len(vessels) == 0:
            continue
        vessels_samples_per_patient[p] = [
            {'vessel': np.array(v['vessel']), 'thickness': v['thickness']} if type(v) == dict
            else np.array(v)
            for v in vessels
        ]
    return vessels_samples_per_patient


def load_vessels_data_split(save_path, all_patients):
    """
    Loads train, val, and test patient lists from text files in a given directory.

    Args:
        save_path (str): Directory containing 'train_patients.txt', 'val_patients.txt', and 'test_patients.txt'.
        all_patients (list): All patient IDs available.

    Returns:
        tuple: (train_patients, val_patients, test_patients)
    """
    train_patients_p = Path(save_path) / 'train_patients.txt'
    val_patients_p = Path(save_path) / 'val_patients.txt'
    test_patients_p = Path(save_path) / 'test_patients.txt'

    with open(test_patients_p.as_posix(), 'r') as f:
        test_patients = [p.strip() for p in f.readlines()]
        test_patients = [p for p in test_patients if p in all_patients]

    with open(val_patients_p.as_posix(), 'r') as f:
        val_patients = [p.strip() for p in f.readlines()]
        val_patients = [p for p in val_patients if p in all_patients]

    with open(train_patients_p.as_posix(), 'r') as f:
        train_patients = [p.strip() for p in f.readlines()]
        train_patients = [p for p in train_patients if p in all_patients]

    return train_patients, val_patients, test_patients


def load_relevant_gts(patients_list, patients_info_xlsx_path, gt_key):
    """
    Return a dictionary mapping each patient (from patients_list) to its measured lab value
    (either 'HGB' or 'RBC') taken from patients_info_xlsx_path.

    The XLSX file must have:
      - An 'ID' column for patient ID,
      - 'Lab Hb [gr/dL]' for HGB (when gt_key='HGB'),
      - 'Lab RBC [M/microL]' for RBC (when gt_key='RBC').

    Args:
        patients_list (list of str):
            Patient IDs (e.g., ['patient1', 'patient2', ...]).
        patients_info_xlsx_path (str):
            Path to the Excel file containing 'ID' and the lab columns needed.
        gt_key (str):
            Either 'HGB' or 'RBC'. Specifies which lab value to extract.

    Returns:
        dict: { lowercased_patient_id: float_lab_value }

    Raises:
        ValueError: If required columns do not exist or if gt_key is invalid.
    """
    # Load the spreadsheet
    df = pd.read_excel(patients_info_xlsx_path, engine='openpyxl')

    # Map user-friendly key to the exact column name in the spreadsheet
    col_map = {
        'HGB': 'Lab Hb [gr/dL]',
        'RBC': 'Lab RBC [M/microL]',
    }

    # Basic validation of columns and gt_key
    if 'ID' not in df.columns:
        raise ValueError("The Excel file must contain an 'ID' column.")
    if gt_key not in col_map:
        raise ValueError(f"Invalid gt_key: {gt_key}. Must be one of {list(col_map.keys())}.")
    if col_map[gt_key] not in df.columns:
        raise ValueError(
            f"The Excel file must contain a '{col_map[gt_key]}' column "
            f"for gt_key='{gt_key}'."
        )

    # Convert the 'ID' column to lowercase for consistent lookup
    df['ID'] = df['ID'].astype(str).str.lower()

    # Create a set of lowercased patient IDs from the input list
    lower_patients_list = [p.lower() for p in patients_list]
    relevant_col = col_map[gt_key]

    gt_per_patient = {}
    # Populate the dictionary for patients that appear in the DataFrame
    for _, row in df.iterrows():
        patient_id = row['ID']
        # Check if this patient's ID is in our target list
        if patient_id in lower_patients_list:
            lab_value = row[relevant_col]
            # Skip if the spreadsheet cell is empty (NaN)
            if pd.isnull(lab_value):
                continue
            gt_per_patient[patient_id] = float(lab_value)

    # Check which patients were not found in the Excel file and print a warning
    missing_patients = set(lower_patients_list) - set(gt_per_patient.keys())
    for missing_id in missing_patients:
        print(f"Warning: patient '{missing_id}' was not found in '{patients_info_xlsx_path}'.")

    return gt_per_patient