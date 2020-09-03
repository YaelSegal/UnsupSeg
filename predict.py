import argparse
import dill
from argparse import Namespace
import torch
import torchaudio
from utils import (detect_peaks, max_min_norm, replicate_first_k_frames, create_textgrid)
from next_frame_classifier import NextFrameClassifier
import os

def main(wav, ckpt, prominence, outpath):
    print(f"running inference on: {wav}")
    print(f"running inferece using ckpt: {ckpt}")
    print("\n\n", 90 * "-")

    ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
    hp = Namespace(**dict(ckpt["hparams"]))

    # load weights and peak detection params
    model = NextFrameClassifier(hp)
    weights = ckpt["state_dict"]
    weights = {k.replace("NFC.", ""): v for k,v in weights.items()}
    model.load_state_dict(weights)
    peak_detection_params = dill.loads(ckpt['peak_detection_params'])['cpc_1']
    if prominence is not None:
        print(f"overriding prominence with {prominence}")
        peak_detection_params["prominence"] = prominence

    # load data
    wav_name = wav.split("/")[-1]
    audio, sr = torchaudio.load(wav)
    assert sr == 16000, "model was trained with audio sampled at 16khz, please downsample."
    audio = audio[0]
    audio = audio.unsqueeze(0)

    # run inference
    preds = model(audio)  # get scores
    preds = preds[1][0]  # get scores of positive pairs
    num_features = preds.size(1)
    preds = replicate_first_k_frames(preds, k=1, dim=1)  # padding
    preds = 1 - max_min_norm(preds)  # normalize scores (good for visualizations)
    
    preds = detect_peaks(x=preds,
                         lengths=[preds.shape[1]],
                         prominence=peak_detection_params["prominence"],
                         width=peak_detection_params["width"],
                         distance=peak_detection_params["distance"])  # run peak detection on scores

    mult = audio.size(1)/num_features
    preds = preds[0] * mult / sr  # transform frame indexes to seconds

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    create_textgrid(preds, audio.size(1)/sr, os.path.join(outpath,wav_name.replace(".wav", ".TextGrid") ))
    print("predicted boundaries (in seconds):")
    print(preds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unsupervised segmentation inference script')
    parser.add_argument('--wav', help='path to wav file')
    parser.add_argument('--ckpt', help='path to checkpoint file')
    parser.add_argument('--prominence', type=float, default=None, help='prominence for peak detection (default: 0.05)')
    parser.add_argument('--output', type=str, default="/data/segalya/ddk/UnsupSeg/pred_textgrid", help='output_folder')
    args = parser.parse_args()
    main(args.wav, args.ckpt, args.prominence, args.output)
