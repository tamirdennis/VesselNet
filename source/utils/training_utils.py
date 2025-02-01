import random
import numpy as np
import torch
from pathlib import Path
import json

from source.utils.datasets_utils import (
    filter_vessels_by_args, VesselsBagDataset,
    load_vessels_samples_from_json, load_vessels_data_split,
    load_relevant_gts
)
from source.utils.sampling_utils import ValuesBinsSampler
from source.utils.testing_utils import vessel_model_validation
from source.utils.vessels_models.models import get_vessels_model


def train_vessels(args):
    """
    Main training procedure for VesselNet model.

    Steps:
      1. Load or filter vessels from JSON/dataset.
      2. Load ground-truth (gt) values from the given XLSX file for each patient.
      3. Remove patients with fewer vessels than min_num_vessels.
      4. If no existing data split is given, create cross-validation folds.
      5. For each experiment/fold:
         a) Prepare train/val/test splits.
         b) Compute normalization factors for vessel images/thickness from training set.
         c) Build DataLoaders for train/val sets.
         d) Train for the specified number of epochs, saving the best model on validation loss.
         e) Clear memory and move on to the next fold if necessary.

    Args:
        args (Namespace): Command-line arguments with attributes such as:
          - load_existing_samples, full_vessels_dataset_path
          - patients_info_xlsx_path, gt_key
          - min_num_vessels, train_val_test_split, number_of_experiments
          - sampler_bins, load_existing_data_split
          - batch_size, bag_size, weight_decay, lr, epochs, epoch_size, etc.
    """
    Path(args.save_path).mkdir(exist_ok=True)

    # 1. Load or filter vessel samples:
    if args.load_existing_samples is not None:
        # load existing filtered samples
        print('Loading existing filtered vessels samples')
        vessels_samples_per_patient = load_vessels_samples_from_json(args.load_existing_samples)
    else:
        vessels_samples_per_patient = load_vessels_samples_from_json(args.full_vessels_dataset_path)
        filter_vessels_by_args(vessels_samples_per_patient, args)
        print('Finished filtering vessels, now saving them for later use')
        save_vessels_samples_as_json(args.output_filtered_samples_path, vessels_samples_per_patient)

    print('Finished loading vessels')

    # 2. Load ground-truth for each patient:
    gt_per_patient = load_relevant_gts(list(vessels_samples_per_patient.keys()),
                                       args.patients_info_xlsx_path,
                                       args.gt_key)

    # 3. Filter out patients with too few vessels:
    num_vessels_per_patient = {p: len(vessels) for p, vessels in vessels_samples_per_patient.items()}
    patients_without_enough_vessels = [
        p for p, num_vessels in num_vessels_per_patient.items()
        if (num_vessels < args.min_num_vessels and p in gt_per_patient)
    ]
    gt_per_patient = {
        p: mean_gt for p, mean_gt in gt_per_patient.items() if p not in patients_without_enough_vessels
    }
    vessels_samples_per_patient = {
        p: vessels for p, vessels in vessels_samples_per_patient.items() if p not in patients_without_enough_vessels
    }

    # 4. Prepare bins for stratified sampling if specified:
    if args.sampler_bins is None:
        bins = None
    else:
        bins = [(args.sampler_bins[i], args.sampler_bins[i + 1]) for i in range(len(args.sampler_bins) - 1)]

    # 5. Create or load data splits for cross-validation:
    if args.load_existing_data_split is None:
        train_val_test_patients = create_stratified_cross_validation_folds(
            gt_per_patient,
            args.number_of_experiments,
            bins,
            args.train_val_test_split[1]
        )
    else:
        existing_data_split_root = Path(args.load_existing_data_split)

    save_path_root = Path(args.save_path)
    save_path_root.mkdir(exist_ok=True)

    for i in range(args.number_of_experiments):
        if args.number_of_experiments != 1:
            args.save_path = save_path_root / f'run_{i}'
        else:
            args.save_path = save_path_root
        args.save_path.mkdir(exist_ok=True)
        args.save_path = args.save_path.as_posix()

        if args.load_existing_data_split is None:
            train_patients, val_patients, test_patients = train_val_test_patients[i]
            save_vessels_data_split(train_patients, val_patients, test_patients, args.save_path)
        else:
            args.load_existing_data_split = (existing_data_split_root / f'run_{i}').as_posix()
            all_patients = list(gt_per_patient.keys())
            train_patients, val_patients, test_patients = load_vessels_data_split(args.load_existing_data_split,
                                                                                  all_patients)

        print(f'Experiment {i}:\n'
              f'train patients: {train_patients}\n'
              f'val patients: {val_patients}\n'
              f'test patients: {test_patients}\n')

        # Log bin distribution for train/val/test
        if len(num_vessels_per_patient) != 0 and bins is not None:
            for bin in bins:
                bin_train_patients = [p for p in train_patients if bin[0] <= gt_per_patient[p] < bin[1]]
                bin_val_patients = [p for p in val_patients if bin[0] <= gt_per_patient[p] < bin[1]]
                bin_test_patients = [p for p in test_patients if bin[0] <= gt_per_patient[p] < bin[1]]

                train_numbers_list = [num_vessels_per_patient[p] for p in bin_train_patients]
                print(f'number of train patients in bin {bin} with no vessels: '
                      f'{len([n for n in train_numbers_list if n == 0])}')
                bin_train_num_vessels = sum(train_numbers_list)
                bin_val_num_vessels = sum([num_vessels_per_patient[p] for p in bin_val_patients])
                bin_test_num_vessels = sum([num_vessels_per_patient[p] for p in bin_test_patients])
                print(f'bin {bin}:\n'
                      f'train: {bin_train_num_vessels} vessels\n'
                      f'validation: {bin_val_num_vessels} vessels\n'
                      f'test: {bin_test_num_vessels} vessels\n')

        # Compute and save normalization factors from training data
        vessel_images_mean_std, vessel_thickness_mean_std = get_and_save_vessel_norm_factors(
            train_patients,
            vessels_samples_per_patient,
            gt_per_patient,
            args
        )

        # Build DataLoaders for train and val
        train_loader, val_dataset, val_sampler = get_vessels_datasets_loaders(
            train_patients,
            val_patients,
            gt_per_patient,
            vessels_samples_per_patient,
            bins,
            args,
            vessel_images_mean_std,
            vessel_thickness_mean_std
        )

        # Initialize model, loss, optimizer, scheduler
        vessels_model = get_vessels_model(args)
        criterion = torch.nn.L1Loss(size_average=False).cuda()
        optimizer = torch.optim.Adam(
            vessels_model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
        best_metric = float('inf')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Save args
        args_p = Path(args.save_path) / 'args.json'
        with open(args_p.as_posix(), 'w') as f:
            json.dump(vars(args), f)

        # 6. Training loop
        for epoch in range(args.epochs):
            print(f'Epoch {epoch}')
            vessels_model.train()
            for b_i in range(args.epoch_size):
                if args.use_thickness:
                    x, thickness, gt = next(iter(train_loader))
                    thickness = thickness.cuda()
                else:
                    x, gt = next(iter(train_loader))
                    thickness = None

                x = x.cuda()
                gt = gt.cuda()
                out = vessels_model(x) if thickness is None else vessels_model(x, thickness)
                gt = gt.type(torch.FloatTensor).cuda().unsqueeze(1)
                loss = criterion(out, gt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Validation every 2 epochs
            if epoch % 2 == 0 and epoch > 0:
                best_metric, results_per_patient, val_loss = vessel_model_validation(
                    val_dataset,
                    val_sampler,
                    vessels_model,
                    criterion,
                    args,
                    best_metric,
                    optimizer,
                    epoch,
                    save_best=True
                )

        # Clean up
        del train_loader, val_dataset, val_sampler, vessels_model, criterion, optimizer, scheduler
        torch.cuda.empty_cache()


def get_and_save_vessel_norm_factors(train_patients,
                                     vessels_samples_per_patient,
                                     gt_per_patient,
                                     args):
    """
    Compute and save normalization factors (mean and std) for vessel images and thickness
    based on a small sample from the training set.

    Args:
        train_patients (list):
            IDs of the patients in the training set.
        vessels_samples_per_patient (dict):
            {patient_id: [list of vessel samples]}, each item possibly including 'vessel' and 'thickness'.
        gt_per_patient (dict):
            {patient_id: ground_truth_value}, though this is not used for the norm factors here.
        args (Namespace):
            Contains bag_size, random_crop, save_path, etc.

    Returns:
        tuple of tuples:
            ( (vessel_images_mean, vessel_images_std), (vessel_thickness_mean, vessel_thickness_std) )
    """
    train_dict_for_sampler = {
        i: {'patient': patient, 'gt': gt}
        for i, (patient, gt) in enumerate(gt_per_patient.items())
        if patient in train_patients
    }

    # Sample a few bags from the training set
    bags_per_patient = 2
    train_sampler = list(train_dict_for_sampler.keys()) * bags_per_patient
    train_bag_dataset = VesselsBagDataset(
        train_dict_for_sampler,
        vessels_samples_per_patient,
        args.bag_size,
        train=False,
        random_crop=args.random_crop,
        bags_per_patient=bags_per_patient,
        use_thickness=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_bag_dataset,
        batch_size=len(train_bag_dataset),
        drop_last=False,
        sampler=train_sampler
    )
    train_loader = iter(train_loader)
    x, thicknesses, _, _ = next(train_loader)

    # Compute means and stds
    x = x.cuda()
    vessel_images_mean = x.mean(dim=(0, 1, 3, 4))
    vessel_images_std = x.std(dim=(0, 1, 3, 4))
    vessel_thickness_mean = thicknesses.mean()
    vessel_thickness_std = thicknesses.std()

    vessel_images_mean_std = {'mean': vessel_images_mean.tolist(), 'std': vessel_images_std.tolist()}
    vessel_thickness_mean_std = {'mean': vessel_thickness_mean.tolist(), 'std': vessel_thickness_std.tolist()}

    # Save to JSON
    vessel_images_mean_std_p = Path(args.save_path) / 'vessel_images_mean_std.json'
    with open(vessel_images_mean_std_p.as_posix(), 'w') as f:
        json.dump(vessel_images_mean_std, f)

    vessel_thickness_mean_std_p = Path(args.save_path) / 'vessels_thickness_mean_std.json'
    with open(vessel_thickness_mean_std_p.as_posix(), 'w') as f:
        json.dump(vessel_thickness_mean_std, f)

    del x, thicknesses, _, train_loader, train_bag_dataset
    torch.cuda.empty_cache()

    return (vessel_images_mean, vessel_images_std), (vessel_thickness_mean, vessel_thickness_std)


def create_stratified_cross_validation_folds(gt_per_patient,
                                             num_of_folds,
                                             bins=None,
                                             val_ratio=0.05):
    """
    Creates stratified cross-validation folds based on ground-truth values and optional bin ranges.

    Args:
        gt_per_patient (dict):
            {patient_id: lab_value}, numeric lab_value used for binning.
        num_of_folds (int):
            Number of cross-validation folds.
        bins (list of tuples, optional):
            List of (bin_low, bin_high) tuples. If None, uses a single bin covering all data.
        val_ratio (float, optional):
            Fraction of training data to allocate to validation within each fold.

    Returns:
        list of tuples:
            Each tuple is (train_patients, val_patients, test_patients).
    """
    folds = [[] for _ in range(num_of_folds)]
    if bins is None:
        bins = [(-float('inf'), float('inf'))]

    # Identify the bin with the fewest patients
    minority_bin_index = np.argmin([
        len([p for p in gt_per_patient if bin[0] <= gt_per_patient[p] < bin[1]])
        for bin in bins
    ])
    num_minorities_per_fold = [0 for _ in range(num_of_folds)]

    # Distribute patients across folds bin by bin
    for bin_ind, bin in enumerate(bins):
        bin_patients = [p for p in gt_per_patient if bin[0] <= gt_per_patient[p] < bin[1]]
        random.shuffle(bin_patients)

        fold_size = len(bin_patients) // num_of_folds
        remainder = len(bin_patients) % num_of_folds

        folds = sorted(folds, key=lambda x: len(x))
        start_idx = 0
        for i in range(num_of_folds):
            end_idx = start_idx + fold_size + (1 if i < remainder else 0)
            bin_fold = bin_patients[start_idx:end_idx]
            folds[i].extend(bin_fold)
            if bin_ind == minority_bin_index:
                num_minorities_per_fold[i] += len(bin_fold)
            start_idx = end_idx

    # Shuffle each fold
    for fold in folds:
        random.shuffle(fold)

    # For each fold, set aside test_patients and subdivide the rest into train/val
    cv_folds = []
    for i in range(num_of_folds):
        test_patients = folds[i]
        train_val_patients = [p for j, fold in enumerate(folds) if j != i for p in fold]
        train_patients = []
        val_patients = []

        for bin in bins:
            bin_train_val_patients = [p for p in train_val_patients if bin[0] <= gt_per_patient[p] < bin[1]]
            random.shuffle(bin_train_val_patients)
            val_last_index = int(len(bin_train_val_patients) * val_ratio)
            val_last_index = max(val_last_index, 1)
            bin_train_patients = bin_train_val_patients[val_last_index:]
            bin_val_patients = bin_train_val_patients[:val_last_index]
            train_patients += bin_train_patients
            val_patients += bin_val_patients

        cv_folds.append((train_patients, val_patients, test_patients))

    return cv_folds


def save_vessels_data_split(train_patients, val_patients, test_patients, save_path):
    """
    Save train, val, and test patient IDs to separate text files under save_path.

    Args:
        train_patients (list):
            IDs for training.
        val_patients (list):
            IDs for validation.
        test_patients (list):
            IDs for testing.
        save_path (str):
            Directory where the .txt files are written.

    Returns:
        None. The text files are created or overwritten.
    """
    train_patients_p = Path(save_path) / 'train_patients.txt'
    val_patients_p = Path(save_path) / 'val_patients.txt'
    test_patients_p = Path(save_path) / 'test_patients.txt'

    with open(test_patients_p.as_posix(), 'w') as f:
        for p in test_patients:
            f.write(f'{p}\n')

    with open(val_patients_p.as_posix(), 'w') as f:
        for p in val_patients:
            f.write(f'{p}\n')

    with open(train_patients_p.as_posix(), 'w') as f:
        for p in train_patients:
            f.write(f'{p}\n')


def save_vessels_samples_as_json(vessels_output_path, vessels_samples_per_patient):
    """
    Saves the vessels_samples_per_patient dictionary to a JSON file.

    Args:
        vessels_output_path (str):
            Path where the JSON file is stored.
        vessels_samples_per_patient (dict):
            {patient_id: [list of vessel samples]}, each sample is either a dict of
            {'vessel': <numpy array>, 'thickness': <float>} or a numpy array.

    Returns:
        None. The JSON file 'samples_per_patients.json' is created or overwritten.
    """
    samples_per_patients_json = {}
    for p, vessels in vessels_samples_per_patient.items():
        for_json_l = [
            {'vessel': v['vessel'].tolist(), 'thickness': v['thickness']} if isinstance(v, dict) else v.tolist()
            for v in vessels
        ]
        samples_per_patients_json[p] = for_json_l

    with open(Path(vessels_output_path).as_posix(), 'w') as f:
        json.dump(samples_per_patients_json, f)


def get_vessels_datasets_loaders(train_patients,
                                 val_patients,
                                 mean_gt_per_patient,
                                 vessels_samples_per_patient,
                                 bins,
                                 args,
                                 vessel_images_mean_std,
                                 vessel_thickness_mean_std=None):
    """
    Builds the DataLoader objects for training and validation sets.

    Args:
        train_patients (list):
            List of patient IDs in the training set.
        val_patients (list):
            List of patient IDs in the validation set.
        mean_gt_per_patient (dict):
            {patient_id: ground_truth_value} for each patient in the dataset.
        vessels_samples_per_patient (dict):
            {patient_id: [vessel_samples]} for each patient.
        bins (list of tuples or None):
            Bin definitions for stratified sampling, e.g. [(0, 11), (11, 13.5), ...].
        args (Namespace):
            - Contains bag_size, batch_size, random_crop, etc.
        vessel_images_mean_std (tuple):
            (images_mean, images_std), used for normalizing images.
        vessel_thickness_mean_std (tuple, optional):
            (thickness_mean, thickness_std).

    Returns:
        tuple:
            (train_loader, val_bag_dataset, val_sampler)
    """
    val_dict_for_sampler = {
        i: {'patient': patient, 'gt': gt}
        for i, (patient, gt) in enumerate(mean_gt_per_patient.items())
        if patient in val_patients
    }
    vals_bags_per_patient = 5
    val_sampler = list(
        ValuesBinsSampler(
            max((len(val_dict_for_sampler) * vals_bags_per_patient // args.batch_size) * args.batch_size,
                args.batch_size),
            pre_data=val_dict_for_sampler,
            bins_size=10,
            bins=bins
        )
    )

    val_bag_dataset = VesselsBagDataset(
        val_dict_for_sampler,
        vessels_samples_per_patient,
        args.bag_size,
        train=False,
        bags_per_patient=vals_bags_per_patient,
        random_crop=args.random_crop,
        vessel_images_mean_std=vessel_images_mean_std,
        vessel_thickness_mean_std=vessel_thickness_mean_std,
        use_thickness=args.use_thickness
    )

    train_dict_for_sampler = {
        i: {'patient': patient, 'gt': mean_gt_per_patient[patient]}
        for i, patient in enumerate(train_patients)
    }
    train_sampler = ValuesBinsSampler(
        args.epoch_size * 1000,
        pre_data=train_dict_for_sampler,
        bins_size=10,
        bins=bins
    )
    train_bag_dataset = VesselsBagDataset(
        train_dict_for_sampler,
        vessels_samples_per_patient,
        args.bag_size,
        train=True,
        random_crop=args.random_crop,
        vessel_images_mean_std=vessel_images_mean_std,
        vessel_thickness_mean_std=vessel_thickness_mean_std,
        use_thickness=args.use_thickness
    )

    train_loader = torch.utils.data.DataLoader(
        train_bag_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        sampler=train_sampler
    )

    return train_loader, val_bag_dataset, val_sampler
