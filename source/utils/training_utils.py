import random

import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler

from source.utils.datasets_utils import \
    filter_vessel_hist_by_args, VesselsBagDataset, \
    load_vessels_samples_and_keys_from_jsons, load_vessels_data_split, load_relevant_gts
from source.utils.sampling_utils import ValuesBinsSampler
from source.utils.testing_utils import vessel_model_validation
from source.utils.vessels_models.models import get_vessels_model
from pathlib import Path
import json
import networkx as nx


def train_vessels(args):
    Path(args.save_path).mkdir(exist_ok=True)
    if args.load_existing_samples is not None:
        vessels_samples_per_patient, gts_keys_per_patient = \
            load_vessels_samples_and_keys_from_jsons(args.load_existing_samples)
    else:
        vessels_samples_per_patient, gts_keys_per_patient = \
            load_vessels_samples_and_keys_from_jsons(args.vessels_dataset_path)
        filter_vessel_hist_by_args(vessels_samples_per_patient, args)
        print('Finished filtering vessels, now saving them for later use')
        save_vessels_samples_and_keys_as_jsons(args.save_path, vessels_samples_per_patient, gts_keys_per_patient)
    print('Finished loading vessels')
    gt_per_patient = load_relevant_gts(args, gts_keys_per_patient, vessels_samples_per_patient)
    num_vessels_per_patient = {p: len(vessels) for p, vessels in vessels_samples_per_patient.items()}
    patients_without_enough_vessels = [p for p, num_vessels in num_vessels_per_patient.items() if
                                       (num_vessels < args.min_num_vessels and p in gt_per_patient)]
    gt_per_patient = {p: mean_gt for p, mean_gt in gt_per_patient.items() if
                      p not in patients_without_enough_vessels}
    vessels_samples_per_patient = {p: vessels for p, vessels in vessels_samples_per_patient.items() if
                                   p not in patients_without_enough_vessels}

    if args.sampler_bins is None:
        bins = None
    else:
        bins = [(args.sampler_bins[i], args.sampler_bins[i + 1]) for i in range(len(args.sampler_bins) - 1)]

    if args.load_existing_data_split is None:
        train_val_test_patients = create_stratified_cross_validation_folds(gt_per_patient,
                                                                           args.number_of_experiments,
                                                                           bins, args.train_val_test_split[1])
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
        print(f'Experiment {i}:\n')
        print(f'train patients: {train_patients}\n'
              f'val patients: {val_patients}\n'
              f'test patients: {test_patients}\n')

        if len(num_vessels_per_patient) != 0:
            # print amount of blue vessels and color vessels in each bin, in train validation and test:
            for bin in bins:
                bin_train_patients = [p for p in train_patients if bin[0] <= gt_per_patient[p] < bin[1]]
                bin_val_patients = [p for p in val_patients if bin[0] <= gt_per_patient[p] < bin[1]]
                bin_test_patients = [p for p in test_patients if bin[0] <= gt_per_patient[p] < bin[1]]
                train_numbers_list = [num_vessels_per_patient[p] for p in bin_train_patients]
                print(
                    f'number of train patients in bin {bin} with no vessels: {len([n for n in train_numbers_list if n == 0])}')
                bin_train_num_vessels = sum(train_numbers_list)
                bin_val_num_vessels = sum([num_vessels_per_patient[p] for p in bin_val_patients])
                bin_test_num_vessels = sum([num_vessels_per_patient[p] for p in bin_test_patients])
                print(f'bin {bin}:\n'
                      f'train: {bin_train_num_vessels} vessels\n'
                      f'validation: {bin_val_num_vessels} vessels\n'
                      f'test: {bin_test_num_vessels} vessels\n'
                      f'')
        vessel_images_mean_std, vessel_thickness_mean_std = get_and_save_vessel_norm_factors(train_patients,
                                                                                             vessels_samples_per_patient,
                                                                                             gt_per_patient,
                                                                                             args
                                                                                             )
        train_loader, val_dataset, val_sampler = get_vessels_datasets_loaders(train_patients, val_patients,
                                                                              gt_per_patient,
                                                                              vessels_samples_per_patient, bins,
                                                                              args,
                                                                              vessel_images_mean_std,
                                                                              vessel_thickness_mean_std)

        vessels_model = get_vessels_model(args)
        criterion = torch.nn.L1Loss(size_average=False).cuda()
        optimizer = torch.optim.Adam(vessels_model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                     weight_decay=args.weight_decay)
        best_metric = float('inf')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # save args in save_path:
        args_p = Path(args.save_path) / 'args.json'
        with open(args_p.as_posix(), 'w') as f:
            json.dump(vars(args), f)

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
            if epoch % 2 == 0 and epoch > 0:
                best_metric, results_per_patient, val_loss = vessel_model_validation(val_dataset, val_sampler,
                                                                                     vessels_model,
                                                                                     criterion, args, best_metric,
                                                                                     optimizer,
                                                                                     epoch, save_best=True)
        # clear cuda memory:
        del train_loader, val_dataset, val_sampler, vessels_model, criterion, optimizer, scheduler
        torch.cuda.empty_cache()


def create_graph_from_pixels(pixel_locations):
    """
    Create a graph from a list of pixel locations.

    Args:
    - pixel_locations (list of tuples): Each tuple represents the (row, col) location of a pixel

    Returns:
    - G (networkx.Graph): A graph where nodes are pixel locations and edges connect adjacent pixels
    """
    G = nx.Graph()

    pixel_set = set(pixel_locations)  # Convert to set for efficient lookups

    # Add nodes
    G.add_nodes_from(pixel_locations)
    # Define adjacency (8-connectivity: N, NE, E, SE, S, SW, W, NW)
    adjacency_offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                         (1, 0), (1, -1), (0, -1), (-1, -1)]

    # Add edges
    for location in pixel_locations:
        row, col = location
        for offset in adjacency_offsets:
            neighbor = (row + offset[0], col + offset[1])
            if neighbor in pixel_set:
                G.add_edge(location, neighbor)

    return G


def get_and_save_vessel_norm_factors(train_patients, vessels_samples_per_patient, gt_per_patient,
                                     args):
    train_dict_for_sampler = {i: {'patient': patient, 'gt': gt} for i, (patient, gt) in
                              enumerate(gt_per_patient.items()) if patient in train_patients}
    bags_per_patient = 2
    train_sampler = list(train_dict_for_sampler.keys()) * bags_per_patient
    train_bag_dataset = VesselsBagDataset(train_dict_for_sampler, vessels_samples_per_patient, args.bag_size,
                                          train=False, random_crop=args.random_crop,
                                          bags_per_patient=bags_per_patient, use_thickness=True)
    train_loader = torch.utils.data.DataLoader(train_bag_dataset,
                                               batch_size=len(train_bag_dataset), drop_last=False,
                                               sampler=train_sampler)
    train_loader = iter(train_loader)
    x, thicknesses, _, _ = next(train_loader)
    x = x.cuda()
    vessel_images_mean = x.mean(dim=(0, 1, 3, 4))
    vessel_images_std = x.std(dim=(0, 1, 3, 4))
    vessel_thickness_mean = thicknesses.mean()
    vessel_thickness_std = thicknesses.std()
    # save mean and std in args.save_path:
    vessel_images_mean_std = {'mean': vessel_images_mean.tolist(), 'std': vessel_images_std.tolist()}
    vessel_images_mean_std_p = Path(args.save_path) / 'vessel_images_mean_std.json'
    with open(vessel_images_mean_std_p.as_posix(), 'w') as f:
        json.dump(vessel_images_mean_std, f)

    vessel_thickness_mean_std = {'mean': vessel_thickness_mean.tolist(), 'std': vessel_thickness_std.tolist()}
    vessel_thickness_mean_std_p = Path(args.save_path) / 'vessels_thickness_mean_std.json'
    with open(vessel_thickness_mean_std_p.as_posix(), 'w') as f:
        json.dump(vessel_thickness_mean_std, f)

    del x, thicknesses, _, train_loader, train_bag_dataset
    torch.cuda.empty_cache()
    return (vessel_images_mean, vessel_images_std), (vessel_thickness_mean, vessel_thickness_std)


def create_stratified_cross_validation_folds(gt_per_patient, num_of_folds, bins=None,
                                             val_ratio=0.05):
    # Initialize a list to hold the folds
    folds = [[] for _ in range(num_of_folds)]
    if bins is None:
        bins = [(-float('inf'), float('inf'))]
    # Process each bin separately
    minority_bin_index = np.argmin([len([p for p in gt_per_patient if bin[0] <= gt_per_patient[p] < bin[1]])
                                    for bin in bins])
    num_minorities_per_fold = [0 for _ in range(num_of_folds)]
    for bin_ind, bin in enumerate(bins):
        # Filter patients in the current bin
        bin_patients = [p for p in gt_per_patient if bin[0] <= gt_per_patient[p] < bin[1]]

        random.shuffle(bin_patients)
        # Calculate the number of patients from this bin in each fold
        fold_size = len(bin_patients) // num_of_folds
        remainder = len(bin_patients) % num_of_folds

        # order folds by size of lists:
        folds = sorted(folds, key=lambda x: len(x))
        # Create folds for this bin
        start_idx = 0
        for i in range(num_of_folds):
            end_idx = start_idx + fold_size + (1 if i < remainder else 0)
            bin_fold = bin_patients[start_idx:end_idx]
            folds[i].extend(bin_fold)
            if bin_ind == minority_bin_index:
                num_minorities_per_fold[i] += len(bin_fold)
            start_idx = end_idx

    # Randomize each fold
    for fold in folds:
        random.shuffle(fold)

    # Create training and validation sets for each fold
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
    # saving test patients:
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


def save_vessels_samples_and_keys_as_jsons(save_path, vessels_samples_per_patient, gts_keys_per_patient):
    gts_keys_per_patient_path = Path(save_path) / 'gts_keys_per_patient.json'
    with open(gts_keys_per_patient_path.as_posix(), 'w') as f:
        json.dump(gts_keys_per_patient, f)
    samples_per_patients_path = Path(save_path) / 'samples_per_patients.json'
    samples_per_patients_json = {}
    for p, vessels in vessels_samples_per_patient.items():
        for_json_l = [{'vessel': v['vessel'].tolist(), 'thickness': v['thickness']} if type(v) == dict else v.tolist()
                      for
                      v in vessels]
        samples_per_patients_json[p] = for_json_l
    with open(samples_per_patients_path.as_posix(), 'w') as f:
        json.dump(samples_per_patients_json, f)


def get_vessels_datasets_loaders(train_patients, val_patients, mean_gt_per_patient,
                                 vessels_samples_per_patient, bins,
                                 args, vessel_images_mean_std, vessel_thickness_mean_std=None):
    val_dict_for_sampler = {i: {'patient': patient, 'gt': gt} for i, (patient, gt) in
                            enumerate(mean_gt_per_patient.items()) if patient in val_patients}
    vals_bags_per_patient = 5
    val_sampler = list(
        ValuesBinsSampler(max((len(val_dict_for_sampler) * vals_bags_per_patient // args.batch_size) * args.batch_size,
                              args.batch_size),
                          pre_data=val_dict_for_sampler, bins_size=10, bins=bins))

    val_bag_dataset = VesselsBagDataset(val_dict_for_sampler, vessels_samples_per_patient, args.bag_size,
                                        train=False,
                                        bags_per_patient=vals_bags_per_patient,
                                        random_crop=args.random_crop, vessel_images_mean_std=vessel_images_mean_std,
                                        vessel_thickness_mean_std=vessel_thickness_mean_std,
                                        use_thickness=args.use_thickness)
    train_dict_for_sampler = {i: {'patient': patient, 'gt': mean_gt_per_patient[patient]} for i, patient in
                              enumerate(train_patients)}

    train_sampler = ValuesBinsSampler(args.epoch_size * 1000, pre_data=train_dict_for_sampler,
                                      bins_size=10, bins=bins)
    train_bag_dataset = VesselsBagDataset(train_dict_for_sampler, vessels_samples_per_patient, args.bag_size,
                                          train=True, random_crop=args.random_crop,
                                          vessel_images_mean_std=vessel_images_mean_std,
                                          vessel_thickness_mean_std=vessel_thickness_mean_std,
                                          use_thickness=args.use_thickness)
    train_loader = torch.utils.data.DataLoader(train_bag_dataset,
                                               batch_size=args.batch_size, drop_last=False,
                                               sampler=train_sampler)
    return train_loader, val_bag_dataset, val_sampler
