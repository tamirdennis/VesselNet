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

from source.utils.datasets_utils import load_vessels_samples_and_keys_from_jsons, load_vessels_data_split, \
    load_relevant_gts, \
    VesselsBagDataset
from source.utils.vessels_models.models import get_vessels_model
from datetime import datetime
import random
from sklearn.metrics import roc_auc_score


def get_test_dataset(args, vessels_samples_per_patient, mean_gt_per_patient, test_patients, vessel_images_mean_std,
                     vessel_thickness_mean_std):
    test_dict_for_sampler = {i: {'patient': patient, 'gt': gt} for i, (patient, gt) in
                             enumerate(mean_gt_per_patient.items()) if patient in test_patients}
    test_sampler_ls = [[i for _ in range(args.bags_per_patient)] for i in test_dict_for_sampler.keys()]
    test_sampler = [item for sublist in test_sampler_ls for item in sublist]

    test_dataset = VesselsBagDataset(test_dict_for_sampler, vessels_samples_per_patient, args.bag_size,
                                     transforms_list=[], train=False, bags_per_patient=args.bags_per_patient,
                                     random_crop=args.random_crop, vessel_images_mean_std=vessel_images_mean_std,
                                     vessel_thickness_mean_std=vessel_thickness_mean_std,
                                     use_thickness=args.use_thickness)
    return test_dataset, test_sampler


def get_id_to_gender_from_xlsx(file_path):
    """
    Extracts a dictionary of IDs and their corresponding gender from an Excel file.
    If the gender is empty for some IDs, their value in the dictionary will be None.

    Args:
    - file_path (str): Path to the Excel file.

    Returns:
    - id_gender_dict (dict): Dictionary with IDs as keys and gender (or None) as values.
    """
    # Read the Excel file
    df = pd.read_excel(file_path, engine='openpyxl')

    # Check if 'id' and 'Gender' columns exist
    if 'id' in df.columns and 'Gender' in df.columns:
        # Initialize the dictionary
        id_gender_dict = {}
        # Iterate over each row to handle empty Gender values
        for _, row in df.iterrows():
            gender = row['Gender'] if pd.notnull(row['Gender']) else None
            id_gender_dict[str(row['id']).lower()] = gender
    else:
        raise ValueError("The Excel file must contain 'id' and 'Gender' columns.")

    return id_gender_dict


def test_vessels(args):

    Path(args.test_graphs_dir).mkdir(exist_ok=True)
    patient_to_gender = get_id_to_gender_from_xlsx(args.patients_info_xlsx_path)
    # prepare dataset for testing:
    runs_folders = [f for f in Path(args.save_path).iterdir() if f.is_dir()]
    data_splits_folders = [Path(args.load_existing_data_split) / f.name for f in runs_folders]

    vessels_samples_per_patient, gts_keys_per_patient = \
        load_vessels_samples_and_keys_from_jsons(args.load_existing_samples)
    gt_per_patient = load_relevant_gts(args, gts_keys_per_patient, vessels_samples_per_patient)

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

        test_dataset, test_sampler = get_test_dataset(args, vessels_samples_per_patient, gt_per_patient,
                                                      test_patients, vessel_images_mean_std, vessel_thickness_mean_std)

        # loading the trained model:

        if os.path.isfile(save_path):
            if not args.load_existing_test_results:
                print("=> loading checkpoint '{}'".format(save_path))
                checkpoint = torch.load(save_path)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                args.start_epoch = checkpoint['epoch']
                args.best_pred = checkpoint['best_prec1']
                print(f"=> loaded checkpoint '{save_path}' (epoch {checkpoint['epoch']})")
                print(f'best val metric: {args.best_pred}')
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
            _, results_per_patient, _ = vessel_model_validation(test_dataset, test_sampler, model,
                                                                criterion, args,
                                                                float('inf'), None,
                                                                args.start_epoch, save_best=False)

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
            [[split_name, patient, preds_mean, gt, std, patient_to_gender[patient]] for patient, preds_mean, gt, std
             in patients_list])
    all_patients_df = pd.DataFrame(all_patients_list_df,
                                   columns=['split', 'patient', 'preds_mean', 'gt', 'preds_std', 'gender'])
    all_patients_results_output_p = Path(args.test_graphs_dir) / 'all_patients_results.csv'
    if all_patients_results_output_p.exists():
        current_date = datetime.now()
        all_patients_results_output_p = Path(
            args.test_graphs_dir) / f'all_patients_results_{current_date.strftime("%Y-%m-%d")}.csv'
    all_patients_df.to_csv(all_patients_results_output_p, index=False)
    plot = px.scatter(data_frame=all_patients_df, x='gt', y='preds_mean', title='all patients predictions',
                      hover_data=['split', 'patient', 'preds_mean', 'gt', 'preds_std'],
                      range_x=range_xy, range_y=range_xy, size_max=12, )
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
    all_auc, all_auc_ci, all_spearmanr, all_spearmanr_ci = \
        results_and_plots_for_patients(
            all_patients_gts, all_patients_preds, all_patients_anemia_threshold, all_patients_bland_altman_plot_name,
            all_patients_regression_plot_name, all_patients_roc_curve_name,
            all_patients_roc_title,
            args.test_graphs_dir)

    # same plots for all males:
    male_patients_regression_plot_name = 'male_patients_regression_plot.png'
    male_patients_bland_altman_plot_name = 'male_patients_bland_altman_plot.png'
    male_patients_roc_curve_name = 'male_patients_roc_curve.png'
    male_patients_roc_title = 'ROC Curve - Males - Mild Anemia'
    # call classification_results_and_plots_for_patients for males:
    male_auc, male_auc_ci, male_spearmanr, male_spearmanr_ci = \
        results_and_plots_for_patients(
            all_male_patients_gts, all_male_patients_preds, args.males_threshold,
            male_patients_bland_altman_plot_name,
            male_patients_regression_plot_name, male_patients_roc_curve_name,
            male_patients_roc_title,
            args.test_graphs_dir)

    # same plots for all females:
    female_patients_regression_plot_name = 'female_patients_regression_plot.png'
    female_patients_bland_altman_plot_name = 'female_patients_bland_altman_plot.png'
    female_patients_roc_curve_name = 'female_patients_roc_curve.png'
    female_patients_roc_title = 'ROC Curve - Females - Severe Anemia'

    # call classification_results_and_plots_for_patients for females:
    female_auc, female_auc_ci, female_spearmanr, female_spearmanr_ci = \
        results_and_plots_for_patients(
            all_female_patients_gts, all_female_patients_preds, args.females_threshold,
            female_patients_bland_altman_plot_name,
            female_patients_regression_plot_name, female_patients_roc_curve_name,
            female_patients_roc_title,
            args.test_graphs_dir)

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
    # save metrics dict to json:
    with open(Path(args.test_graphs_dir) / 'all_metrics.json', 'w') as json_file:
        json.dump(all_metrics_dict, json_file)


def spearmanr_value(gts, preds):
    return spearmanr(gts, preds)[0]


def results_and_plots_for_patients(patients_gts, patients_preds, low_threshold,
                                   patients_bland_altman_plot_name, patients_regression_plot_name,
                                   all_patients_roc_curve_name, roc_title,
                                   graph_output_dir):
    gts_preds = [(patients_gts, patients_preds)]
    plot_multiple_regression_results(gts_preds,
                                     xlabel="Lab Hb (gr/L)",
                                     ylabel="VesselNet Hb (gr/L)",
                                     output_path=Path(graph_output_dir) / patients_regression_plot_name,
                                     x_range=[4, 20], y_range=[4, 20]
                                     )

    bland_altman_plot(patients_gts, patients_preds,
                      Path(graph_output_dir) / patients_bland_altman_plot_name
                      )

    all_auc, all_auc_ci = plot_roc_curves(gts_preds,
                                          roc_output_path=Path(graph_output_dir) / all_patients_roc_curve_name,
                                          title=roc_title, threshold=low_threshold)
    rho_spearman = spearmanr_value(patients_gts, patients_preds)
    rho_spearman_ci = metric_ci(patients_gts, patients_preds, spearmanr_value,
                                threshold_strat=low_threshold)

    return all_auc, all_auc_ci, rho_spearman, rho_spearman_ci


def metric_ci(gt_list, preds_list, metric_func, threshold_strat, n_boot=1000, ci=95, *args, **kwargs):
    """
    Calculates the confidence interval for a custom metric using bootstrapping.

    Parameters:
        gt_list (list or array): Ground truth values.
        preds_list (list or array): Prediction values.
        metric_func (function): A function that computes the metric. It should accept two inputs: (gt, preds).
        n_boot (int): Number of bootstrap samples.
        ci (float): Confidence interval percentage (e.g., 95 for 95%).
        *args: Additional positional arguments to pass to the metric function.
        **kwargs: Additional keyword arguments to pass to the metric function.

    Returns:
        tuple: (metric, lower_CI, upper_CI) where metric is the original metric value and lower_CI/upper_CI are the confidence interval bounds.
    """
    gt_array = np.array(gt_list)
    class_gt = gt_array if threshold_strat is None else gt_array <= threshold_strat
    preds_array = np.array(preds_list)

    # Ensure binary ground truth for stratification
    unique_classes = np.unique(class_gt)
    if len(unique_classes) != 2:
        raise ValueError("Stratified sampling requires binary ground truth labels.")

    # Separate positive and negative indices
    pos_indices = np.where(class_gt == unique_classes[1])[0]
    neg_indices = np.where(class_gt == unique_classes[0])[0]

    # Stratified bootstrapping for CI
    bootstrapped_metrics = []
    for _ in range(n_boot):
        # Sample positive and negative indices with replacement
        sampled_pos_indices = np.random.choice(pos_indices, size=len(pos_indices), replace=True)
        sampled_neg_indices = np.random.choice(neg_indices, size=len(neg_indices), replace=True)

        # Combine sampled indices
        sampled_indices = np.concatenate([sampled_pos_indices, sampled_neg_indices])
        np.random.shuffle(sampled_indices)  # Shuffle to avoid order effects

        # Sample the GT and predictions
        sample_gt = gt_array[sampled_indices]
        sample_preds = preds_array[sampled_indices]

        # Compute the metric for the sample
        boot_metric = metric_func(sample_gt, sample_preds, *args, **kwargs)
        bootstrapped_metrics.append(boot_metric)

    # Compute the confidence interval
    lower = np.percentile(bootstrapped_metrics, (100 - ci) / 2)
    upper = np.percentile(bootstrapped_metrics, 100 - (100 - ci) / 2)

    return lower, upper


def calc_auc(gt_list, preds_list, threshold=11):
    fpr, tpr, _ = metrics.roc_curve(np.array(gt_list) <= threshold, -np.array(preds_list))
    return metrics.auc(fpr, tpr)


def plot_roc_curves(gts_preds_list, hue_labels=None, roc_output_path=None,
                    threshold=11.0, title='ROC Curves'):

    auc_ci_list = []
    plt.figure(figsize=(6, 6))
    if hue_labels is None:
        hue_labels = [f"" for i in range(len(gts_preds_list))]
    for i, ((gts, preds), label) in enumerate(zip(gts_preds_list, hue_labels)):
        gts = np.array(gts)
        preds = np.array(preds)
        y_true = gts <= threshold
        y_score = - preds
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        auc_ci = metric_ci(gts, preds, calc_auc, threshold_strat=threshold, threshold=threshold)
        auc_ci_list.append((roc_auc, auc_ci))
        plt.plot(fpr, tpr, drawstyle='steps-post', label=f"{label} (AUC = {100 * roc_auc:.2f}%)",
                 color=None)

    # Plot the diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Random Classifier")

    # Add labels, title, and legend
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
    # Clear the current figure and close the plot to free memory
    plt.clf()
    plt.close()
    if len(gts_preds_list) == 1:
        return auc_ci_list[0]
    return auc_ci_list

def plot_multiple_regression_results(
        gts_preds_list,
        hue_labels=None,
        xlabel="X",
        ylabel="Y",
        output_path=None,
        x_range=None,
        y_range=None
):
    """
    Plot multiple regression lines with an optional confidence interval (CI),
    based on a list of ground truth (GT) values and predictions.
    It can handle one or multiple models:

    - If only one model (one pair of GT/pred), it uses a scatter plot.
    - If multiple models, it uses a seaborn lmplot with a confidence interval.

    Args:
        gts_preds_list (list of tuples): Each tuple should be (gts, preds),
            where 'gts' is a list (or array) of ground-truth values and
            'preds' is a list (or array) of predicted values.
        hue_labels (list of str, optional): Labels for each (gts, preds) pair.
        xlabel (str, optional): Label for the X-axis. Default is "X".
        ylabel (str, optional): Label for the Y-axis. Default is "Y".
        output_path (str, optional): If provided, path to save the figure.
        x_range (list or tuple, optional): Range [min, max] for the X-axis.
            Default is [4, 20].
        y_range (list or tuple, optional): Range [min, max] for the Y-axis.
            Default is [4, 20].

    Returns:
        None
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
    plt.figure(figsize=(6, 6))
    gts = np.asarray(gts)
    preds = np.asarray(preds)
    mean = np.mean([gts, preds], axis=0)
    diff = gts - preds  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(mean, diff)
    plt.axhline(md, color='black', linestyle='-')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
    plt.xlabel("Means", fontsize=16)
    plt.ylabel("Difference", fontsize=18)
    plt.ylim(md - 3.5 * sd, md + 3.5 * sd)

    plt.subplots_adjust(right=0.85)
    plt.gca().tick_params(axis='x', labelsize=14)  # For x-axis tick labels
    plt.gca().tick_params(axis='y', labelsize=14)  # For y-axis tick labels
    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()
    plt.close()


def vessel_model_validation(val_dataset, val_sampler, vessels_model, criterion, args, best_metric, optimizer, epoch,
                            save_best=True):
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
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_path)

    return best_metric, results_per_patient, val_loss


def save_checkpoint(state, is_best, task_id, filename='checkpoint.pth'):
    checkpoint_p = Path(task_id) / filename
    torch.save(state, checkpoint_p.as_posix())
    if is_best:
        best_checkpoint_p = Path(task_id) / 'model_best.pth'
        shutil.copyfile(checkpoint_p.as_posix(), best_checkpoint_p.as_posix())


def load_vessels_norm_factors(save_path):
    vessel_images_mean_std_p = Path(save_path) / 'vessel_images_mean_std.json'
    with open(vessel_images_mean_std_p.as_posix(), 'r') as f:
        vessel_images_mean_std = json.load(f)
    vessels_thickness_mean_std_p = Path(save_path) / 'vessels_thickness_mean_std.json'
    with open(vessels_thickness_mean_std_p.as_posix(), 'r') as f:
        vessel_thickness_mean_std = json.load(f)
    vessel_images_mean_std = vessel_images_mean_std['mean'], vessel_images_mean_std['std']
    vessel_thickness_mean_std = vessel_thickness_mean_std['mean'], vessel_thickness_mean_std['std']
    return vessel_images_mean_std, vessel_thickness_mean_std
