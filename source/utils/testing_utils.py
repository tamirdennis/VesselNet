import os
import shutil

import torch
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
import json
from pathlib import Path
import numpy as np
from sklearn import metrics
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from source.utils.datasets_utils import load_vessels_samples_from_json, load_vessels_data_split, \
    load_relevant_gts, \
    VesselsBagDataset
from source.utils.vessels_models.models import get_vessels_model
from datetime import datetime
import random
from sklearn.metrics import roc_auc_score



def get_test_dataset(args,
                     vessels_samples_per_patient,
                     gt_per_patient,
                     test_patients,
                     vessel_images_mean_std,
                     vessel_thickness_mean_std):
    """
    Create a test dataset and sampler based on given arguments.

    Args:
        args (Namespace): Parsed arguments containing bags_per_patient, bag_size, etc.
        vessels_samples_per_patient (dict):
            A dictionary of {patient_id: [list_of_vessel_data]}.
        gt_per_patient (dict):
            A dictionary of {patient_id: ground_truth_value}.
        test_patients (list):
            A list of patient IDs to include in the test set.
        vessel_images_mean_std (tuple):
            A tuple (mean, std) for normalizing vessel images.
        vessel_thickness_mean_std (tuple):
            A tuple (mean, std) for normalizing vessel thickness values.

    Returns:
        test_dataset (VesselsBagDataset):
            Dataset object containing the test samples.
        test_sampler (list):
            A list of indices (in repeated groups) that ensures each patient
            is sampled bags_per_patient times.
    """
    test_dict_for_sampler = {
        i: {'patient': patient, 'gt': gt}
        for i, (patient, gt) in enumerate(gt_per_patient.items())
        if patient in test_patients
    }
    test_sampler_ls = [[i for _ in range(args.bags_per_patient)] for i in test_dict_for_sampler.keys()]
    test_sampler = [item for sublist in test_sampler_ls for item in sublist]

    test_dataset = VesselsBagDataset(
        test_dict_for_sampler,
        vessels_samples_per_patient,
        args.bag_size,
        transforms_list=[],
        train=False,
        bags_per_patient=args.bags_per_patient,
        random_crop=args.random_crop,
        vessel_images_mean_std=vessel_images_mean_std,
        vessel_thickness_mean_std=vessel_thickness_mean_std,
        use_thickness=args.use_thickness
    )
    return test_dataset, test_sampler


def get_id_to_gender_from_xlsx(file_path):
    """
    Extract a dictionary mapping patient 'ID' to 'Gender' from an Excel file.

    Args:
        file_path (str): Path to the Excel file (must have 'ID' and 'Gender' columns).

    Returns:
        dict:
            Dictionary with keys as stringified patient IDs (lowercased) and values as
            the corresponding gender ('M', 'F', or None if missing).
    """
    df = pd.read_excel(file_path, engine='openpyxl')

    if 'ID' in df.columns and 'Gender' in df.columns:
        id_gender_dict = {}
        for _, row in df.iterrows():
            gender = row['Gender'] if pd.notnull(row['Gender']) else None
            id_gender_dict[str(row['ID']).lower()] = gender
    else:
        raise ValueError("The Excel file must contain 'ID' and 'Gender' columns.")

    return id_gender_dict


def test_vessels(args):
    """
    The main testing function that evaluates the VesselNet model(s) on test patients.

    Steps:
      1. Loads patient gender info and sets up directories.
      2. Finds all run folders in args.save_path to test.
      3. Loads vessels samples, ground truths, and normalizing factors.
      4. Optionally loads an existing model checkpoint.
      5. For each split (train/val/test), collects predictions and organizes results.
      6. Generates multiple plots (regression, Bland-Altman, ROC) for overall and
         by-gender comparisons.
      7. Computes metrics (Spearman correlation, AUC) and saves them.

    Args:
        args (Namespace): Parsed arguments from test_vessels.py including:
            - patients_info_xlsx_path
            - use_thickness
            - low_threshold, males_threshold, females_threshold
            - load_existing_test_results
            - test_graphs_dir, save_path
            - load_existing_samples, load_existing_data_split
            - etc.

    Returns:
        None. Results are saved to disk.
    """
    Path(args.test_graphs_dir).mkdir(exist_ok=True)
    patient_to_gender = get_id_to_gender_from_xlsx(args.patients_info_xlsx_path)

    # prepare dataset for testing:
    runs_folders = [f for f in Path(args.save_path).iterdir() if f.is_dir()]
    data_splits_folders = [Path(args.load_existing_data_split) / f.name for f in runs_folders]

    vessels_samples_per_patient = load_vessels_samples_from_json(args.load_existing_samples)
    gt_per_patient = load_relevant_gts(list(vessels_samples_per_patient.keys()), args.patients_info_xlsx_path, args.gt_key)

    all_gt = list(gt_per_patient.values())
    all_gt = np.array(all_gt)
    range_xy = [np.min(all_gt) - 1, np.max(all_gt) + 1]

    model = get_vessels_model(args) if not args.load_existing_test_results else None

    all_patients_results_dict = {}
    all_patients_gts = []
    all_patients_preds = []
    all_male_patients_gts = []
    all_male_patients_preds = []
    all_female_patients_gts = []
    all_female_patients_preds = []

    for run_folder, data_split_folder in zip(runs_folders, data_splits_folders):
        print(f'run folder: {run_folder}')
        print(f'data split folder: {data_split_folder}')
        save_path = (run_folder / 'model_best.pth').as_posix() if len(runs_folders) > 1 else args.save_path
        all_patients = list(vessels_samples_per_patient.keys())
        train_patients, val_patients, test_patients = load_vessels_data_split(data_split_folder.as_posix(),
                                                                              all_patients)
        vessel_images_mean_std, vessel_thickness_mean_std = load_vessels_norm_factors(run_folder.as_posix())

        test_dataset, test_sampler = get_test_dataset(
            args,
            vessels_samples_per_patient,
            gt_per_patient,
            test_patients,
            vessel_images_mean_std,
            vessel_thickness_mean_std
        )

        # loading the trained model:
        if os.path.isfile(save_path):
            if not args.load_existing_test_results:
                print("=> loading checkpoint '{}'".format(save_path))
                checkpoint = torch.load(save_path)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                args.start_epoch = checkpoint['epoch']
                best_pred = checkpoint['best_prec1']
                print(f"=> loaded checkpoint '{save_path}' (epoch {checkpoint['epoch']})")
                print(f'best val metric: {best_pred}')
        else:
            raise ValueError(f"no checkpoint found at {save_path}")

        criterion = torch.nn.L1Loss(size_average=False).cuda()
        graphs_output_dir = Path(args.test_graphs_dir) / run_folder.name
        graphs_output_dir.mkdir(exist_ok=True)
        raw_results_per_patient_p = graphs_output_dir / 'raw_results_per_patient.json'

        if args.load_existing_test_results:
            with open(raw_results_per_patient_p, 'r') as json_file:
                results_per_patient = json.load(json_file)
        else:
            _, results_per_patient, _ = vessel_model_validation(
                test_dataset, test_sampler, model, criterion, args,
                float('inf'), None, args.start_epoch, save_best=False
            )

        with open(raw_results_per_patient_p, 'w') as json_file:
            json.dump(results_per_patient, json_file)

        patients_lists_df = []

        gts = []
        preds = []
        males_gts = []
        males_preds = []
        females_gts = []
        females_preds = []
        for patient, v in results_per_patient.items():
            preds_mean = np.mean(v['preds'])
            patients_lists_df.append([patient, preds_mean, v['gt'], np.std(v['preds'])])
            preds.append(preds_mean)
            gts.append(v['gt'])
            if patient not in patient_to_gender:
                print(f'patient {patient} does not have Gender info')
                continue
            if patient_to_gender[patient] == 'M':
                males_gts.append(v['gt'])
                males_preds.append(preds_mean)
            elif patient_to_gender[patient] == 'F':
                females_gts.append(v['gt'])
                females_preds.append(preds_mean)
        all_patients_results_dict[data_split_folder.name] = patients_lists_df
        all_patients_gts += gts
        all_patients_preds += preds
        all_male_patients_gts += males_gts
        all_male_patients_preds += males_preds
        all_female_patients_gts += females_gts
        all_female_patients_preds += females_preds

    all_patients_list_df = []
    for split_name, patients_list in all_patients_results_dict.items():
        all_patients_list_df.extend(
            [
                [split_name, patient, preds_mean, gt, std, patient_to_gender[patient]]
                for patient, preds_mean, gt, std in patients_list
            ]
        )
    all_patients_df = pd.DataFrame(
        all_patients_list_df,
        columns=['split', 'patient', 'preds_mean', 'gt', 'preds_std', 'gender']
    )
    all_patients_results_output_p = Path(args.test_graphs_dir) / 'all_patients_results.csv'
    if all_patients_results_output_p.exists():
        current_date = datetime.now()
        all_patients_results_output_p = Path(args.test_graphs_dir) / \
            f'all_patients_results_{current_date.strftime("%Y-%m-%d")}.csv'
    all_patients_df.to_csv(all_patients_results_output_p, index=False)
    plot = px.scatter(
        data_frame=all_patients_df,
        x='gt', y='preds_mean',
        title='all patients predictions',
        hover_data=['split', 'patient', 'preds_mean', 'gt', 'preds_std'],
        range_x=range_xy, range_y=range_xy, size_max=12,
    )
    plot.add_trace(
        go.Scatter(
            x=range_xy,
            y=range_xy,
            mode="lines",
            line=go.scatter.Line(color="gray", dash="dash"),
            showlegend=False,
        )
    )
    plot.update_traces(marker={'size': 12})
    plot.write_html(Path(args.test_graphs_dir) / 'all_patients_results_graph.html', auto_open=True)

    print('number of patients: ', len(all_patients_gts))
    print('number of male patients: ', len(all_male_patients_gts))
    print('number of female patients: ', len(all_female_patients_gts))

    all_patients_anemia_threshold = args.low_threshold
    all_patients_regression_plot_name = 'all_patients_regression_plot.png'
    all_patients_bland_altman_plot_name = 'all_patients_bland_altman_plot.png'
    all_patients_roc_curve_name = 'all_patients_roc_curve.png'
    all_patients_roc_title = 'ROC Curve - All Patients - Severe Anemia'
    all_auc, all_auc_ci, all_spearmanr, all_spearmanr_ci = results_and_plots_for_patients(
        all_patients_gts,
        all_patients_preds,
        all_patients_anemia_threshold,
        all_patients_bland_altman_plot_name,
        all_patients_regression_plot_name,
        all_patients_roc_curve_name,
        all_patients_roc_title,
        args.test_graphs_dir
    )

    # same plots for all males:
    male_patients_regression_plot_name = 'male_patients_regression_plot.png'
    male_patients_bland_altman_plot_name = 'male_patients_bland_altman_plot.png'
    male_patients_roc_curve_name = 'male_patients_roc_curve.png'
    male_patients_roc_title = 'ROC Curve - Males - Mild Anemia'
    male_auc, male_auc_ci, male_spearmanr, male_spearmanr_ci = results_and_plots_for_patients(
        all_male_patients_gts,
        all_male_patients_preds,
        args.males_threshold,
        male_patients_bland_altman_plot_name,
        male_patients_regression_plot_name,
        male_patients_roc_curve_name,
        male_patients_roc_title,
        args.test_graphs_dir
    )

    # same plots for all females:
    female_patients_regression_plot_name = 'female_patients_regression_plot.png'
    female_patients_bland_altman_plot_name = 'female_patients_bland_altman_plot.png'
    female_patients_roc_curve_name = 'female_patients_roc_curve.png'
    female_patients_roc_title = 'ROC Curve - Females - Severe Anemia'
    female_auc, female_auc_ci, female_spearmanr, female_spearmanr_ci = results_and_plots_for_patients(
        all_female_patients_gts,
        all_female_patients_preds,
        args.females_threshold,
        female_patients_bland_altman_plot_name,
        female_patients_regression_plot_name,
        female_patients_roc_curve_name,
        female_patients_roc_title,
        args.test_graphs_dir
    )

    all_metrics_dict = {
        'all_spearmanr': all_spearmanr,
        'spearmanr_CI_low': all_spearmanr_ci[0],
        'spearmanr_CI_high': all_spearmanr_ci[1],
        'all_auc': all_auc,
        'all_auc_ci': all_auc_ci,
        'male_spearmanr': male_spearmanr,
        'male_spearmanr_ci_low': male_spearmanr_ci[0],
        'male_spearmanr_ci_high': male_spearmanr_ci[1],
        'male_auc': male_auc,
        'male_auc_ci_low': male_auc_ci[0],
        'male_auc_ci_high': male_auc_ci[1],
        'female_spearmanr': female_spearmanr,
        'female_spearmanr_ci_low': female_spearmanr_ci[0],
        'female_spearmanr_ci_high': female_spearmanr_ci[1],
        'female_auc': female_auc,
        'female_auc_ci_low': female_auc_ci[0],
        'female_auc_ci_high': female_auc_ci[1],
    }
    with open(Path(args.test_graphs_dir) / 'all_metrics.json', 'w') as json_file:
        json.dump(all_metrics_dict, json_file)


def spearmanr_value(gts, preds):
    """
    Computes the Spearman correlation coefficient between ground truths and predictions.

    Args:
        gts (list or np.array): Ground truth values.
        preds (list or np.array): Predicted values.

    Returns:
        float: Spearman correlation coefficient (rho).
    """
    return spearmanr(gts, preds)[0]


def results_and_plots_for_patients(patients_gts,
                                   patients_preds,
                                   low_threshold,
                                   patients_bland_altman_plot_name,
                                   patients_regression_plot_name,
                                   all_patients_roc_curve_name,
                                   roc_title,
                                   graph_output_dir):
    """
    Calculates metrics and generates regression, Bland-Altman, and ROC plots for a group of patients.

    Args:
        patients_gts (list): Ground truth values for patients.
        patients_preds (list): Model predictions corresponding to patients_gts.
        low_threshold (float): The threshold below which the patient is considered "positive" (anemic).
        patients_bland_altman_plot_name (str): File name for the Bland-Altman plot image.
        patients_regression_plot_name (str): File name for the regression plot image.
        all_patients_roc_curve_name (str): File name for the ROC curve image.
        roc_title (str): Title used in the ROC curve plot.
        graph_output_dir (str): Directory where plots will be saved.

    Returns:
        tuple:
            (auc_value, (auc_ci_low, auc_ci_high), spearman_corr, (spearman_ci_low, spearman_ci_high))
    """
    gts_preds = [(patients_gts, patients_preds)]
    plot_multiple_regression_results(
        gts_preds,
        xlabel="Lab Hb (gr/L)",
        ylabel="VesselNet Hb (gr/L)",
        output_path=Path(graph_output_dir) / patients_regression_plot_name,
        x_range=[4, 20],
        y_range=[4, 20]
    )

    bland_altman_plot(
        patients_gts,
        patients_preds,
        Path(graph_output_dir) / patients_bland_altman_plot_name
    )

    all_auc, all_auc_ci = plot_roc_curves(
        gts_preds_list=gts_preds,
        roc_output_path=Path(graph_output_dir) / all_patients_roc_curve_name,
        title=roc_title,
        threshold=low_threshold
    )
    rho_spearman = spearmanr_value(patients_gts, patients_preds)
    rho_spearman_ci = metric_ci(
        patients_gts,
        patients_preds,
        spearmanr_value,
        threshold_strat=low_threshold
    )

    return all_auc, all_auc_ci, rho_spearman, rho_spearman_ci


def metric_ci(gt_list,
              preds_list,
              metric_func,
              threshold_strat,
              n_boot=1000,
              ci=95,
              *args,
              **kwargs):
    """
    Calculates the confidence interval for a custom metric using stratified bootstrapping.

    Args:
        gt_list (list or np.array): Ground truth values.
        preds_list (list or np.array): Prediction values.
        metric_func (function): A function that computes the metric, signature like metric_func(gt, preds).
        threshold_strat (float or None): If not None, used to split ground truth into binary classes for stratification.
        n_boot (int, optional): Number of bootstrap samples. Defaults to 1000.
        ci (int, optional): Confidence interval percentage (e.g., 95 for 95%). Defaults to 95.
        *args: Additional positional arguments to pass to the metric function.
        **kwargs: Additional keyword arguments to pass to the metric function.

    Returns:
        tuple:
            (lower_CI, upper_CI) representing the lower and upper bounds of the computed metric.
    """
    gt_array = np.array(gt_list)
    class_gt = gt_array if threshold_strat is None else gt_array <= threshold_strat
    preds_array = np.array(preds_list)

    unique_classes = np.unique(class_gt)
    if len(unique_classes) != 2:
        raise ValueError("Stratified sampling requires binary ground truth labels.")

    pos_indices = np.where(class_gt == unique_classes[1])[0]
    neg_indices = np.where(class_gt == unique_classes[0])[0]

    bootstrapped_metrics = []
    for _ in range(n_boot):
        sampled_pos_indices = np.random.choice(pos_indices, size=len(pos_indices), replace=True)
        sampled_neg_indices = np.random.choice(neg_indices, size=len(neg_indices), replace=True)
        sampled_indices = np.concatenate([sampled_pos_indices, sampled_neg_indices])
        np.random.shuffle(sampled_indices)

        sample_gt = gt_array[sampled_indices]
        sample_preds = preds_array[sampled_indices]

        boot_metric = metric_func(sample_gt, sample_preds, *args, **kwargs)
        bootstrapped_metrics.append(boot_metric)

    lower = np.percentile(bootstrapped_metrics, (100 - ci) / 2)
    upper = np.percentile(bootstrapped_metrics, 100 - (100 - ci) / 2)

    return lower, upper


def calc_auc(gt_list, preds_list, threshold=11):
    """
    Calculates the AUC (Area Under the Curve) for ROC given ground truths and predictions.

    Args:
        gt_list (list or np.array): Ground truth values.
        preds_list (list or np.array): Prediction values.
        threshold (float, optional): Threshold used to label ground truth as positive or negative.

    Returns:
        float: Calculated AUC value.
    """
    fpr, tpr, _ = metrics.roc_curve(np.array(gt_list) <= threshold, -np.array(preds_list))
    return metrics.auc(fpr, tpr)


def plot_roc_curves(gts_preds_list,
                    hue_labels=None,
                    roc_output_path=None,
                    threshold=11.0,
                    title='ROC Curves'):
    """
    Plots ROC curve(s) and calculates AUC (and confidence intervals) for multiple (gts, preds) pairs.

    Args:
        gts_preds_list (list): A list of tuples, where each tuple is (gts, preds).
        hue_labels (list, optional): Labels corresponding to each (gts, preds) pair for plotting.
        roc_output_path (Path, optional): If provided, saves the ROC plot to this path.
        threshold (float, optional): Threshold for classifying patients as anemic or not.
        title (str, optional): Title of the ROC plot.

    Returns:
        list or tuple:
            If multiple (gts, preds) sets are provided, returns a list of (AUC, (CI_low, CI_high)).
            If only one set is provided, returns a single (AUC, (CI_low, CI_high)) tuple.
    """
    auc_ci_list = []
    plt.figure(figsize=(6, 6))
    if hue_labels is None:
        hue_labels = [f"" for _ in range(len(gts_preds_list))]

    for i, ((gts, preds), label) in enumerate(zip(gts_preds_list, hue_labels)):
        gts = np.array(gts)
        preds = np.array(preds)
        y_true = gts <= threshold
        y_score = - preds
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        auc_ci = metric_ci(gts, preds, calc_auc, threshold_strat=threshold, threshold=threshold)
        auc_ci_list.append((roc_auc, auc_ci))
        plt.plot(fpr, tpr, drawstyle='steps-post',
                 label=f"{label} (AUC = {100 * roc_auc:.2f}%)")

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Random Classifier")
    if len(gts_preds_list) > 1:
        plt.title(title, fontsize=16)
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel("False Positive Rate (FPR)", fontsize=16)
    plt.ylabel("True Positive Rate (TPR)", fontsize=16)
    plt.legend(loc="lower right", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.tight_layout()

    if roc_output_path:
        plt.savefig(roc_output_path.as_posix())
    plt.clf()
    plt.close()

    if len(gts_preds_list) == 1:
        return auc_ci_list[0]
    return auc_ci_list


def plot_multiple_regression_results(gts_preds_list,
                                     hue_labels=None,
                                     xlabel="X",
                                     ylabel="Y",
                                     output_path=None,
                                     x_range=None,
                                     y_range=None):
    """
    Plot one or multiple regression (scatter/seaborn lmplot) results of ground truth vs predictions.

    Args:
        gts_preds_list (list of tuples):
            Each tuple is (gts, preds).
        hue_labels (list of str, optional):
            Labels for each pair (used if multiple regression lines).
        xlabel (str, optional): Label for the X-axis.
        ylabel (str, optional): Label for the Y-axis.
        output_path (Path or str, optional): If provided, saves the plot to this path.
        x_range (list or tuple, optional): Range for X-axis, e.g., [4, 20].
        y_range (list or tuple, optional): Range for Y-axis, e.g., [4, 20].

    Returns:
        None. The plot is saved or displayed inline.
    """
    if x_range is None:
        x_range = [4, 20]
    if y_range is None:
        y_range = [4, 20]

    x_col = 'gt'
    y_col = 'preds_mean'
    data_list = [pd.DataFrame({x_col: gts, y_col: preds}) for gts, preds in gts_preds_list]

    if hue_labels is None:
        hue_labels = [f"Model {i + 1}" for i in range(len(gts_preds_list))]

    combined_data = pd.concat(
        [df.assign(Model=label) for df, label in zip(data_list, hue_labels)],
        ignore_index=True
    )

    is_one_model = (len(data_list) == 1)
    plt.figure(figsize=(6, 6))

    if is_one_model:
        sns.scatterplot(data=data_list[0], x=x_col, y=y_col, alpha=1.0)
        plt.xlim(x_range)
        plt.ylim(y_range)
    else:
        my_plot = sns.lmplot(
            data=combined_data,
            x=x_col,
            y=y_col,
            hue="Model",
            height=6,
            aspect=1.5,
            ci=95,
            scatter_kws={"alpha": 0.5},
            scatter=False
        )
        my_plot._legend.get_title().set_fontsize(16)
        for t in my_plot._legend.texts:
            t.set_fontsize(16)
        plt.xlim(x_range)
        plt.ylim(y_range)

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.plot([x_range[0], x_range[1]], [y_range[0], y_range[1]], 'k--', lw=2)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def bland_altman_plot(gts, preds, file_path):
    """
    Generate a Bland-Altman plot comparing ground truths and predictions.

    Args:
        gts (list or np.array): Ground truth values.
        preds (list or np.array): Model predictions.
        file_path (str or Path): Path where the plot will be saved.

    Returns:
        None. Saves the Bland-Altman plot to disk.
    """
    plt.figure(figsize=(6, 6))
    gts = np.asarray(gts)
    preds = np.asarray(preds)
    mean = np.mean([gts, preds], axis=0)
    diff = gts - preds
    md = np.mean(diff)
    sd = np.std(diff, axis=0)

    plt.scatter(mean, diff)
    plt.axhline(md, color='black', linestyle='-')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
    plt.xlabel("Means", fontsize=16)
    plt.ylabel("Difference", fontsize=18)
    plt.ylim(md - 3.5 * sd, md + 3.5 * sd)
    plt.subplots_adjust(right=0.85)
    plt.gca().tick_params(axis='x', labelsize=14)
    plt.gca().tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()
    plt.close()


def vessel_model_validation(val_dataset,
                           val_sampler,
                           vessels_model,
                           criterion,
                           args,
                           best_metric,
                           optimizer,
                           epoch,
                           save_best=True):
    """
    Validates the vessels model on a given validation dataset and sampler.

    Args:
        val_dataset (Dataset): Validation dataset object.
        val_sampler (list): Indices specifying the sampling order for validation.
        vessels_model (nn.Module): The model to be validated.
        criterion (nn.Module): Loss function (e.g., L1Loss).
        args (Namespace): Parsed arguments (includes use_thickness, etc.).
        best_metric (float): The best metric value so far (for saving "best" checkpoint).
        optimizer (torch.optim.Optimizer or None): Optimizer for the model, if any.
        epoch (int): Current epoch number (used for logging).
        save_best (bool, optional): If True, saves the model checkpoint if the current metric is better.

    Returns:
        tuple:
          (best_metric, results_per_patient, val_loss)
    """
    print('start validation')
    val_loss = 0
    num_samples = 0
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, drop_last=True,
                                             sampler=val_sampler)
    vessels_model.eval()
    results_per_patient = {}

    with torch.no_grad():
        for sample_data in val_loader:
            if args.use_thickness:
                x, thickness, gt, patient = sample_data
            else:
                x, gt, patient = sample_data
                thickness = None
            x = x.cuda()
            gt = gt.cuda()
            out = vessels_model(x) if thickness is None else vessels_model(x, thickness)
            patient = patient[0]
            if patient not in results_per_patient:
                results_per_patient[patient] = {'gt': gt.item(), 'preds': []}
            results_per_patient[patient]['preds'].append(out.item())
            gt = gt.type(torch.FloatTensor).cuda().unsqueeze(1)

            loss = criterion(out, gt)
            val_loss += loss.item()
            num_samples += len(gt)
            for b_i, gg in enumerate(gt):
                if b_i % 10 == 0:
                    print(f'gt: {gg.item()}, pred: {out[b_i].item()}')

    val_loss /= num_samples
    print(f'epoch {epoch}, val loss: {val_loss}')
    print(f'chosen val metric: validation loss')
    is_best = val_loss <= best_metric
    best_metric = min(val_loss, best_metric)

    print(' * best val metric {val_m:.3f} '.format(val_m=best_metric))
    if save_best:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': vessels_model.state_dict(),
            'best_prec1': best_metric,
            'optimizer': optimizer.state_dict() if optimizer else None,
        }, is_best, args.save_path)

    return best_metric, results_per_patient, val_loss


def save_checkpoint(state, is_best, task_id, filename='checkpoint.pth'):
    """
    Saves the current model state to disk, and optionally updates model_best.pth.

    Args:
        state (dict): Dictionary containing model state, epoch, best metric, etc.
        is_best (bool): True if this state is the best so far, False otherwise.
        task_id (str): The directory where checkpoint will be stored.
        filename (str, optional): The checkpoint file name.

    Returns:
        None. The checkpoint is saved to disk.
    """
    checkpoint_p = Path(task_id) / filename
    torch.save(state, checkpoint_p.as_posix())
    if is_best:
        best_checkpoint_p = Path(task_id) / 'model_best.pth'
        shutil.copyfile(checkpoint_p.as_posix(), best_checkpoint_p.as_posix())


def load_vessels_norm_factors(save_path):
    """
    Loads normalization factors (mean/std) for vessel images and thickness from JSON files.

    Args:
        save_path (str): Path to the folder containing
            'vessel_images_mean_std.json' and 'vessels_thickness_mean_std.json'.

    Returns:
        tuple:
            ( (images_mean, images_std), (thickness_mean, thickness_std) )
    """
    vessel_images_mean_std_p = Path(save_path) / 'vessel_images_mean_std.json'
    with open(vessel_images_mean_std_p.as_posix(), 'r') as f:
        vessel_images_mean_std = json.load(f)
    vessels_thickness_mean_std_p = Path(save_path) / 'vessels_thickness_mean_std.json'
    with open(vessels_thickness_mean_std_p.as_posix(), 'r') as f:
        vessel_thickness_mean_std = json.load(f)
    vessel_images_mean_std = (vessel_images_mean_std['mean'], vessel_images_mean_std['std'])
    vessel_thickness_mean_std = (vessel_thickness_mean_std['mean'], vessel_thickness_mean_std['std'])
    return vessel_images_mean_std, vessel_thickness_mean_std
