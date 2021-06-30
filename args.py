import argparse
import time
import os
import json
import pandas as pd

parser = argparse.ArgumentParser()
# dataset and dataloader args
parser.add_argument('--max_len', type=int, default=30)
parser.add_argument('--mask_prob', type=float, default=0.1)
parser.add_argument('--sample_size', type=int, default=100)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--val_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--enable_sample', type=bool, default=True)

# model args
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--attn_heads',type=int, default=4)
parser.add_argument('--d_ffn', type=int, default=512)
parser.add_argument('--bert_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--eval_per_steps',type=int, default=1000)
parser.add_argument('--enable_res_parameter', type=bool, default=False)

# train args
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--num_epoch', type=int, default=30)
parser.add_argument('--metric_ks', type=list, default=[5,10,20,50])

args = parser.parse_args()
# other args

DATA = pd.read_csv('data/movielen_lenth_30_cold_10.csv', header=None).values + 1
num_item = DATA.max()
del DATA
args.num_item = int(num_item)

timestr = time.strftime("%y-%m-%d-%Hh%Mm%Ss")
args.save_path = timestr + '/'
os.mkdir(args.save_path)

config_file = open(args.save_path + 'args.json','w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()