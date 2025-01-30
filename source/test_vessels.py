import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import argparse
from source.utils.testing_utils import test_vessels

parser = argparse.ArgumentParser()

parser.add_argument('--patients_info_xlsx_path', type=str, required=True,
                    help='Path to the XLSX file containing patients info with columns "ID", "Gender", "Lab Hb [gr/dL]",'
                         ' and "Lab RBC [M/microL]"')

parser.add_argument('--low_threshold', type=float, default=11.0,
                    help='Low threshold (e.g., HGB < 11.0) used for classification/ROC calculations in the general case.')

parser.add_argument('--males_threshold', type=float, default=13.5,
                    help='Low threshold for male patients (e.g., HGB < 13.5 for mild anemia).')

parser.add_argument('--females_threshold', type=float, default=12.0,
                    help='Low threshold for female patients (e.g., HGB < 12.0  for mild anemia).')

parser.add_argument('--use_thickness', action='store_true',
                    help='If set, model uses vessel thickness as an additional input.')

parser.add_argument('--load_existing_test_results', action='store_true',
                    help='If set, loads existing test results from JSON instead of running the model again.')

parser.add_argument('--test_graphs_dir', type=str, default='',
                    help='Directory where test graphs and results will be saved.')

parser.add_argument('--stem_kernel', type=int, default=2,
                    help='Kernel size for the stem convolution in the ConvNeXt-based model.')

parser.add_argument('--stem_stride', type=int, default=2,
                    help='Stride for the stem convolution in the ConvNeXt-based model.')

parser.add_argument('--block_kernel', type=int, default=7,
                    help='Kernel size for the depthwise convolutions in the ConvNeXt blocks.')

parser.add_argument('--bag_size', type=int, default=16,
                    help='Number of vessel samples in one bag.')

parser.add_argument('--num_params_multiplier', type=int, default=8,
                    help='Multiplier that expands the number of channels in the ConvNeXt model.')

parser.add_argument('--dropout_p', type=float, default=0.0,
                    help='Dropout probability used in the final MLP head.')

parser.add_argument('--random_crop', type=int, nargs='+', default=None,
                    help='If provided, applies a random crop of this size to the input vessel images. '
                         'Expects two integers, e.g., --random_crop 80 80.')

parser.add_argument('--vessels_length', type=int, default=80,
                    help='Width dimension (or second dimension) of vessel images if random_crop is not used.')

parser.add_argument('--num_frames', type=int, default=160,
                    help='Number of frames (height) of the input vessel images.')

parser.add_argument('--gt_key', type=str, default='HGB',
                    help='Key for the ground-truth value in the .xlsx file (e.g., "HGB").')

parser.add_argument('--save_path', type=str, default='./save_file/A_baseline',
                    help='Path to the folder where model checkpoints are saved (and/or loaded).')

parser.add_argument('--load_existing_samples', type=str, default=None,
                    help='Path to existing JSON file with filtered vessel samples.')

parser.add_argument('--load_existing_data_split', type=str, default=None,
                    help='Path to the folder containing the train/val/test splits.'
                         ' If none is given it will create new split.')

parser.add_argument('--bags_per_patient', type=int, default=70,
                    help='Number of "bags" (random sets of vessels) to sample per patient for evaluation.')

args = parser.parse_args()


def main():
    test_vessels(args)


if __name__ == '__main__':
    main()
