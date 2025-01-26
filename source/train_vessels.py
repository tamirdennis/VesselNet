import argparse
from source.utils.training_utils import train_vessels

parser = argparse.ArgumentParser()

parser.add_argument('--vessels_dataset_path', default="", type=str,
                    help='Path to a directory or file containing raw vessels data. Used when load_existing_samples is None.')

parser.add_argument('--patients_info_xlsx_path', type=str, required=True,
                    help='Path to the XLSX file containing patient info with columns "ID", "Gender", '
                         '"Lab Hb [gr/dL]", and "Lab RBC [M/microL]" (or other relevant columns).')

parser.add_argument('--vessels_length', type=int, default=80,
                    help='Defines the minimum vessel length. Vessels are cut or padded to this length if needed.')

parser.add_argument('--use_thickness', action='store_true',
                    help='If set, uses vessel thickness as an additional feature in the model.')

parser.add_argument('--load_existing_samples', type=str, default=None,
                    help='Path to existing JSON file(s) with preprocessed vessel samples. If not provided, '
                         'the code will load and filter from vessels_dataset_path.')

parser.add_argument('--train_val_test_split', nargs='+', type=float, default=[0.8, 0.1, 0.1],
                    help='Ratio split for training, validation, and test sets if no existing split is provided. '
                         'Example: --train_val_test_split 0.8 0.1 0.1')

parser.add_argument('--number_of_experiments', type=int, default=1,
                    help='Number of cross-validation folds or separate runs to execute. If > 1, multiple folds/runs.')

parser.add_argument('--min_num_vessels', type=int, default=0,
                    help='Minimum number of vessels required for a patient to be included.')

parser.add_argument('--stem_kernel', type=int, default=2,
                    help='Kernel size for the stem convolution in the ConvNeXt-based model.')

parser.add_argument('--stem_stride', type=int, default=2,
                    help='Stride for the stem convolution in the ConvNeXt-based model.')

parser.add_argument('--block_kernel', type=int, default=7,
                    help='Kernel size for the depthwise convolutions in the ConvNeXt blocks.')

parser.add_argument('--num_params_multiplier', type=int, default=8,
                    help='Multiplier that expands the number of channels in the ConvNeXt model layers.')

parser.add_argument('--dropout_p', type=float, default=0.0,
                    help='Dropout probability used in the final MLP head.')

parser.add_argument('--load_existing_data_split', type=str, default=None,
                    help='Path to an existing data split folder containing train_patients.txt, val_patients.txt, and test_patients.txt.')

parser.add_argument('--max_vessel_thickness', type=float, default=100,
                    help='Maximum allowed vessel thickness. Vessels above this thickness are filtered out.')

parser.add_argument('--min_vessel_thickness', type=float, default=0,
                    help='Minimum allowed vessel thickness. Vessels below this thickness are filtered out.')

parser.add_argument('--gt_key', type=str, default='HGB',
                    help='The lab measurement key to predict (e.g., "HGB", "RBC"). Default is "HGB".')

parser.add_argument('--num_frames', type=int, default=160,
                    help='Number of frames (height) for the input vessel images if random_crop is not used.')

parser.add_argument('--random_crop', type=int, nargs='+', default=None,
                    help='If provided, applies a random crop of this size to the input vessel images. '
                         'Expects two integers, e.g., --random_crop 80 80.')

parser.add_argument('--save_path', type=str, default='./save_file/A_baseline',
                    help='Folder path where model checkpoints and intermediate results will be saved.')

parser.add_argument('--workers', type=int, default=16,
                    help='Number of worker processes for data loading.')

parser.add_argument('--sampler_bins', nargs='+', default=None, type=float,
                    help='List of bin edges (floats) for stratified sampling. ')

parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size (number of bags) for training.')

parser.add_argument('--bag_size', type=int, default=16,
                    help='Number of vessel samples in one "bag" for MIL.')

parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay parameter for the optimizer.')

parser.add_argument('--epochs', type=int, default=150,
                    help='Total number of training epochs.')

parser.add_argument('--epoch_size', type=int, default=1600,
                    help='Number of training iterations per epoch (effectively the number of batches).')

parser.add_argument('--lr', type=float, default=1e-5,
                    help='Learning rate for the optimizer.')

args = parser.parse_args()


def main():
    """
    Main function to initiate the vessel training process.

    This sets up arguments, then calls train_vessels() with those arguments.
    """
    train_vessels(args)


if __name__ == '__main__':
    main()
