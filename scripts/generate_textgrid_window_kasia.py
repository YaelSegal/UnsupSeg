import os
import sys
sys.path.append(os.getcwd())
from textgrid import *

import argparse
import glob

parser = argparse.ArgumentParser("Create files names list for feature extraction")

parser.add_argument("--wav_path",type=str, default="/data/segalya/ddk/buchwald/data_AB",help="The location of the wav file or the directory of wavs")
parser.add_argument("--textgrid_path", type=str, default="/data/segalya/ddk/buchwald/textgrids_AB", help="The location of the textgrid or the directory of textgrids")
parser.add_argument("--outpath", default="/data/segalya/ddk/UnsupSeg/window_buchwald/textgrids", help="The location of the wav file or the directory of wavs")
# parser.add_argument("--tier_name", help="the search window tier name", default="syllable")
parser.add_argument("--tier_name", help="the search window tier name", default="textgrid")
parser.add_argument("--new_tier_name", help="the new  tier name", default="window")

args = parser.parse_args()

def extract_textdrid_data(textgrid_filename, window, new_tier_name, outpath):
    textgrid = TextGrid()
    textgrid.read(textgrid_filename)
    # extract tier names
    tier_names = textgrid.getNames()
    region_list = []
    region_span = 1 # at list 1 sec between regions
    prev_start = 0
    if window in tier_names:
        current_region = []
        tier_index = tier_names.index(window)
        wav_file_data = ""
        output_data = ""
        x_min = ""
        x_max = ""
        # run over all intervals in the tier
        for interval in textgrid.getFirst(window):
            if re.search(r'\S', interval.mark):
                if x_min == interval.maxTime and x_max == interval.minTime:
                    continue
                x_min = interval.minTime
                x_max = interval.maxTime
                if not region_list and not current_region:
                    prev_start = x_min
                if x_min - prev_start > 1:
                    region_list.append(current_region)
                    current_region = []
                prev_start = x_min
                current_region.append([x_min, x_max, interval.mark])
        if current_region:
            region_list.append(current_region)


        if new_tier_name in tier_names:
            textgrid.delete(new_tier_name)
        # create new textgrid with window and burst
        total_min = textgrid.minTime
        total_max = textgrid.maxTime
        window_tier = IntervalTier(new_tier_name)
        tmp_x_min = 0
        for idx, region in enumerate(region_list):
            x_min = region[0][0] - 0.05
            x_max = region[-1][1] + 0.05
            window_tier.add(x_min, x_max, "window_{}".format(idx))
            tmp_x_min = x_max
        if tmp_x_min < total_max:
            window_tier.add(tmp_x_min, total_max, "")

        textgrid.append(window_tier)
        
        new_textgrid_base_name = os.path.basename(textgrid_filename).replace(".TextGrid", ".TextGrid")
        new_textgrid_name = os.path.join(outpath, new_textgrid_base_name)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        textgrid.write(new_textgrid_name)
        

print(args.textgrid_path)
for filename in os.listdir(args.textgrid_path):
    file_path = os.path.join(args.textgrid_path, filename)
    extract_textdrid_data(file_path, args.tier_name, args.new_tier_name, args.outpath)



