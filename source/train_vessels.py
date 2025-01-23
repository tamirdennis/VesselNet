import sys
sys.path.append('/home/tamirdenis/projects/VesselsNet')

import argparse
from source.utils.training_utils import train_vessels

parser = argparse.ArgumentParser()

parser.add_argument('--vessels_dataset_path', default="", type=str,
                    help='')
parser.add_argument('--vessels_length', type=int, default=80,
                    help='minimum vessel length - cut all vessels to this length')

parser.add_argument('--use_thickness', action='store_true',
                    help='If true, use thickness as a feature for VesselNet')

parser.add_argument('--load_existing_samples', type=str, default=None,
                    help='')

parser.add_argument('--annotations_dir', type=str, default=None,)

parser.add_argument('--train_val_test_split', nargs='+', type=float, default=[0.8,0.1,0.1],
                    help='train val test split')
parser.add_argument('--split_test_by_date_only', action='store_true', default=False,)

parser.add_argument('--convnext_type', type=str, default='b',
                    help='')
parser.add_argument('--number_of_experiments', type=int, default=1)
parser.add_argument('--min_num_vessels', type=int, default=0,)
parser.add_argument('--stem_kernel', type=int, default=2,
                    help='')
parser.add_argument('--stem_stride', type=int, default=2,
                    help='')
parser.add_argument('--block_kernel', type=int, default=7,
                    help='')
parser.add_argument('--num_params_multiplier', type=int, default=8,)

#TODO: remove this:
parser.add_argument('--dropout_p', type=float, default=0.0)


parser.add_argument('--load_existing_data_split', type=str, default=None,
                    help='')

parser.add_argument('--videos_info_xls', type=str, default='data/filtered_stable_v2/Videos Info.xlsx',
                    help='')

# additional arguments:
parser.add_argument('--gt_dir', type=str, default='data/blood tests in excel',
                    help='')

parser.add_argument('--gt_ranges', type=float, nargs='+', default=[5, 10.8],
                    help='gt ranges of the gt_key for the classifier')

parser.add_argument('--vit_patch', type=int, nargs='+', default=(1, 5),
                    help='')

parser.add_argument('--vit_embed_dim', type=int, default=768,
                    help='')

parser.add_argument('--model_depth', type=int, default=12,
                    help='')

parser.add_argument('--vivit_heads', type=int, default=3,
                    help='')

parser.add_argument('--vivit_head_dim', type=int, default=64,
                    help='')

parser.add_argument('--vivit_pool', type=str, default='cls',
                    help='')

#TODO: change this to the new version:
# parser.add_argument('--vessel_thin_score', type=float, default=0.0, help='')
#
# parser.add_argument('--max_vessel_thin_score', type=float, default=1.0, help='')
parser.add_argument('--max_vessel_thickness', type=float, default=100, help='')

parser.add_argument('--min_vessel_thickness', type=float, default=0, help='')


parser.add_argument('--gt_key', type=str, default='WBC',
                    help='')

parser.add_argument('--num_frames', type=int, default=160,
                    help='')

parser.add_argument('--random_crop', type=int, nargs='+', default=None,
                    help='')

parser.add_argument('--save_path', type=str, default='./save_file/A_baseline',
                    help='save checkpoint directory')

parser.add_argument('--workers', type=int, default=16,
                    help='load data workers')

parser.add_argument('--sampler_bins', nargs='+', default=None, type=float,
                    help='')

parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')

parser.add_argument('--bag_size', type=int, default=16, help='input batch size for training')

parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--momentum', type=float, default=0.95,
                    help='momentum')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs to train')
parser.add_argument('--epoch_size', type=int, default=1600,
                    help='')
parser.add_argument('--best_pred', type=int, default=1e5,
                    help='best pred')

parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate')

args = parser.parse_args()


def main():
    train_vessels(args)


if __name__ == '__main__':
    main()