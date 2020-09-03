import os
import sys
sys.path.append(os.getcwd())
from textgrid import *

import argparse
import glob
import random
import soundfile
import librosa
random.seed(1234)

parser = argparse.ArgumentParser("Create files names list for feature extraction")

parser.add_argument("--wav_path",type=str, default="/data/segalya/ddk/buchwald/data_AB",help="The location of the wav file or the directory of wavs")
parser.add_argument("--textgrid_path", type=str, default="/data/segalya/ddk/UnsupSeg/window_buchwald/textgrids", help="The location of the textgrid or the directory of textgrids")
parser.add_argument("--outpath", default="/data/segalya/ddk/UnsupSeg/buchwald_proccess", help="The location of the wav file or the directory of wavs")
parser.add_argument("--tier_name", help="the search window tier name", default="window")

args = parser.parse_args()

TEST_SIZE, VAL_SIZE = 0.2, 0.2

textgrid_list = glob.glob(args.textgrid_path + "/*.TextGrid")

test_list = []
val_list = []
train_list = []
random.shuffle(textgrid_list)

for textgrid_name in textgrid_list:
    basename = textgrid_name.split("/")[-1]
    wav_name = os.path.join(args.wav_path, basename.replace(".TextGrid", ".WAV"))

    if len(val_list) < VAL_SIZE * len(textgrid_list):
        val_list.append([textgrid_name, wav_name])
        continue
    if len(test_list) < TEST_SIZE * len(textgrid_list):
        test_list.append([textgrid_name, wav_name])
        continue
    train_list.append([textgrid_name, wav_name])


def prepare_textgrid_wav(current_list, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for textgrid_file, wav_file in current_list:
        textgrid = TextGrid.fromFile(textgrid_file)
        window_tier = textgrid.getFirst(args.tier_name)
        basename = textgrid_file.split("/")[-1].replace(".TextGrid", "")

        wav, sr = soundfile.read(wav_file)
        if sr != 16000:
            wav = librosa.resample(wav, sr, 16000)
            sr = 16000
        windows = []
        for interval in window_tier:
            if not re.search(r'\S', interval.mark):
                continue
            windows.append([interval.minTime, interval.maxTime])
        
        values_tier = textgrid.getFirst("textgrid")
        if not values_tier:
            values_tier = textgrid.getFirst("syllable")

        win_intervals = []
        if len(windows) < 1:
            print("{} doesnt have window".format(textgrid_file))
            continue
        win_start = windows[0][0]
        win_end = windows[0][1]
        for idx, interval in enumerate(values_tier):
            if interval.minTime < win_start:
                continue
            if interval.maxTime > win_end:
                if len(win_intervals) > 1: # one item is just butterfly
                    new_wav_start = int(win_start*sr)
                    new_basename = basename + "_{}".format(new_wav_start)
                    new_wav_file_name = os.path.join(path, new_basename + ".wav")
                    new_txt_file_name = os.path.join(path, new_basename + ".phn")
                    wav_cut = wav[new_wav_start: int(win_end*sr)]
                    soundfile.write(new_wav_file_name, wav_cut, sr)

                    txt_cur = open(new_txt_file_name, "w")
                    for item in win_intervals:
                        if not re.search(r'\S', item.mark):
                            continue
                        start, end, mark = item.minTime, item.maxTime, item.mark
                        txt_cur.write("{} {} {}\n".format(int(start*sr)-new_wav_start, int(end*sr)-new_wav_start, mark))
                    txt_cur.close()

                if len(windows) > 1:
                    windows.pop(0)
                    win_start = windows[0][0]
                    win_end = windows[0][1]
                else:
                    break
                win_intervals = []
                continue
            win_intervals.append(interval)




prepare_textgrid_wav(val_list, args.outpath + "/val")
prepare_textgrid_wav(test_list, args.outpath + "/test")
prepare_textgrid_wav(train_list, args.outpath + "/train")