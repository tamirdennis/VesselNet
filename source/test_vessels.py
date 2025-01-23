import sys

sys.path.append('/home/tamirdenis/projects/VesselsNet')

import argparse
from source.utils.testing_utils import test_vessels

parser = argparse.ArgumentParser()

parser.add_argument('--patients_info_xlsx_path', type=str, required=True, )

parser.add_argument('--low_threshold', type=float, default=11.0)
parser.add_argument('--males_threshold', type=float, default=13.5)
parser.add_argument('--females_threshold', type=float, default=12.0)

parser.add_argument('--use_thickness', action='store_true',
                    help='')

parser.add_argument('--load_existing_test_results', action='store_true',
                    help='')

parser.add_argument('--test_graphs_dir', type=str, default='', help='')

parser.add_argument('--stem_kernel', type=int, default=2,
                    help='')
parser.add_argument('--stem_stride', type=int, default=2,
                    help='')
parser.add_argument('--block_kernel', type=int, default=7,
                    help='')
parser.add_argument('--bag_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--num_params_multiplier', type=int, default=8, )

parser.add_argument('--dropout_p', type=float, default=0.0)

parser.add_argument('--random_crop', type=int, nargs='+', default=None,
                    help='')

parser.add_argument('--num_frames', type=int, default=160,
                    help='')

parser.add_argument('--gt_dir', type=str, default='data/blood tests in excel',
                    help='')

parser.add_argument('--gt_key', type=str, default='HGB',
                    help='')

parser.add_argument('--save_path', type=str, default='./save_file/A_baseline',
                    help='save checkpoint directory')

parser.add_argument('--load_existing_samples', type=str, default=None,
                    help='')

parser.add_argument('--load_existing_data_split', type=str, default=None,
                    help='')

parser.add_argument('--bags_per_patient', type=int, default=70, help='number of bags per patient')

args = parser.parse_args()


def main():
    test_vessels(args)


if __name__ == '__main__':
    main()
