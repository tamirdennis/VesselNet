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

# print gpu available:
print("cuda:")
print(torch.cuda.is_available())

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def get_relevant_gts(relevant_paths, gt_key, relevant_xls_paths, as_dict=False):
    relevant_xls_names_cleaned = [Path(s).stem.replace('_', '').replace('-', '').lower() for s in relevant_xls_paths]
    relevant_dirs_cleaned = [relevant_parent.replace('_', '').replace('-', '').lower() for relevant_parent in
                             relevant_paths]
    relevant_dir_cleaned_unique = list(np.unique(relevant_dirs_cleaned))
    dir_cleaned_to_gt = {}
    for relevant_dir_cleaned in relevant_dir_cleaned_unique:
        if relevant_dir_cleaned in relevant_xls_names_cleaned:
            relevant_xls_p = relevant_xls_paths[relevant_xls_names_cleaned.index(relevant_dir_cleaned)]
            relevant_xl_np = np.array(pd.read_excel(relevant_xls_p, header=None))
            first_col_list = list(relevant_xl_np[:, 0])
            if gt_key in first_col_list:
                gt_key_index = first_col_list.index(gt_key)
            else:
                print(gt_key)
                print("not inside:")
                print(relevant_xls_p)
                print(relevant_xl_np)
                continue
            gt = relevant_xl_np[gt_key_index, 1]
            dir_cleaned_to_gt[relevant_dir_cleaned] = gt
        else:
            print(f'Could not find {relevant_dir_cleaned} relevant xls')
    if as_dict:
        relevant_gts = {relevant_paths[k]: dir_cleaned_to_gt.get(relevant_dir_cleaned, None) for k, relevant_dir_cleaned
                        in
                        enumerate(relevant_dirs_cleaned)}
    else:
        relevant_gts = [dir_cleaned_to_gt.get(relevant_dir_cleaned, None) for relevant_dir_cleaned in
                        relevant_dirs_cleaned]
    return relevant_gts


def filter_vessel_hist_by_args(vessels_info_d, args):
    for patient, vessels in vessels_info_d.items():
        passed_vessels = []
        for vessel_info in vessels:
            if args.min_vessel_thickness < vessel_info['thickness'] < args.max_vessel_thickness:
                passed_vessels.append(vessel_info)

        vessels_info_d[patient] = passed_vessels


class VesselsBagDataset(torch.utils.data.Dataset):

    def __init__(self, inds_to_patients, vessels_samples_per_patient, bag_size,
                 transforms_list=None, train=False, bags_per_patient=None, random_crop=None,
                 vessel_images_mean_std=None, vessel_thickness_mean_std=None, use_thickness=False):
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
        self.num_vessels = num_vessels // bag_size if (
                bags_per_patient is None or self.train) else bags_per_patient * len(vessels_samples_per_patient)
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
            self.curr_vessels_bags_per_patient[curr_patient] = (self.curr_vessels_bags_per_patient[
                                                                    curr_patient] + 1) % self.bags_per_patient
        vessels_bag = np.array(
            [curr_vessels[ind]['vessel'] if type(curr_vessels[ind]) == dict else curr_vessels[ind] for ind in
             vessels_bag_inds])
        if self.use_thickness:
            vessels_thickness_bag = np.array([curr_vessels[ind]['thickness'] for ind in vessels_bag_inds])
            vessels_thickness_bag = torch.Tensor(vessels_thickness_bag)
            if self.vessel_thickness_mean_std is not None:
                vessels_thickness_bag = (vessels_thickness_bag - self.vessel_thickness_mean_std[0]) / \
                                        self.vessel_thickness_mean_std[1]
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
        return self.num_vessels


def load_vessels_samples_and_keys_from_jsons(save_path):
    gts_keys_per_patient_path = Path(save_path) / 'gts_keys_per_patient.json'
    with open(gts_keys_per_patient_path.as_posix(), 'r') as f:
        gts_keys_per_patient = json.load(f)
    samples_per_patients_path = Path(save_path) / 'samples_per_patients.json'
    with open(samples_per_patients_path.as_posix(), 'r') as f:
        samples_per_patients_json = json.load(f)
    vessels_samples_per_patient = {}
    for p, vessels in samples_per_patients_json.items():
        if len(vessels) == 0:
            continue
        vessels_samples_per_patient[p] = [
            {'vessel': np.array(v['vessel']), 'thickness': v['thickness']} if type(v) == dict
            else np.array(v) for v in vessels]

    return vessels_samples_per_patient, gts_keys_per_patient


def load_vessels_data_split(save_path, all_patients):
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


def load_relevant_gts(args, gts_keys_per_patient, vessels_samples_per_patient):
    relevant_paths_keys = []
    for p, patient_keys in gts_keys_per_patient.items():
        relevant_paths_keys += patient_keys
    relevant_xls_paths = list(Path(args.gt_dir).rglob('*.xlsx'))
    relevant_xls_paths = [xlsx_p.as_posix() for xlsx_p in relevant_xls_paths]
    vids_gts = get_relevant_gts(relevant_paths_keys, args.gt_key, relevant_xls_paths, as_dict=True)
    gt_per_patient = {}
    for p, vessels in vessels_samples_per_patient.items():
        if len(vessels) == 0:
            print(f'patient {p} has no relevant vessels')
            continue
        gts_l = [vids_gts[p_key] for p_key in gts_keys_per_patient[p]]
        gts_l = [gt for gt in gts_l if gt is not None]
        if len(gts_l) == 0:
            print(f'patient {p} has no records for {args.gt_key}')
            continue
        curr_gt = np.mean(np.unique(gts_l))
        gt_per_patient[p] = curr_gt
    return gt_per_patient
