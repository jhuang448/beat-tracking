import librosa
import os, argparse
from utils import predict

import torch
import torch.nn as nn
import torch.utils.data as data
from madmom.features import DBNBeatTrackingProcessor

from data import testDataset
from model import BeatTrackingModel, data_processing, eval_audio_transforms
import utils

audio_dir = "/import/c4dm-datasets/ballroom/BallroomData/"
annot_dir = "/import/c4dm-datasets/ballroom/jku-beat-annotations/"

hparams = {
        "n_cnn_layers": 2,
        "n_rnn_layers": 3,
        "rnn_dim": 25,
        "n_class": 2,
        "n_feats": 8,
        "dropout": 0.1,
        "stride": 1,
        "input_sample": 220500,
        "batch_size": 1
}

model = BeatTrackingModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout'], hparams['input_sample']
)

dbn = DBNBeatTrackingProcessor(
    min_bpm=60,
    max_bpm=240,
    transition_lambda=100,
    fps=(44100 // 1024),
    online=True)

dbn_downbeat = DBNBeatTrackingProcessor(
    min_bpm=15,
    max_bpm=80,
    transition_lambda=100,
    fps=(44100 // 1024),
    online=True)

def beatTracker(inputFile):
    state = utils.load_model(model, None, "checkpoints/dummy/checkpoint_1", False)

    y, _ = utils.load(inputFile, sr=44100, mono=True)
    x = torch.Tensor(y)

    x = eval_audio_transforms(x)
    x = nn.utils.rnn.pad_sequence(x, batch_first=True).unsqueeze(1).transpose(2, 3)

    # Predict
    all_outputs = model(x)

    # batch_num, _, output_length = all_outputs.shape

    total_length = all_outputs.shape[1]

    # print(all_outputs.shape)  # batch, length, classes
    _, _, num_classes = all_outputs.shape

    song_pred = all_outputs.data.numpy().reshape(-1, num_classes)
    # print(song_pred.shape) # total_length, num_classes

    beats_pred = song_pred[:total_length, 0]
    downbeats_pred = song_pred[:total_length, 1]
    # print(song_pred.shape)  # total_length, num_classes

    dbn.reset()
    predicted_beats = dbn.process_offline(beats_pred)

    dbn_downbeat.reset()
    predicted_downbeats = dbn.process_offline(downbeats_pred)

    return predicted_beats, predicted_downbeats

def main(args):

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    global hparams
    global model

    if 'cuda' in device:
        print("move model to gpu")
        model = utils.DataParallel(model)
        model.cuda()

    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)

    # print('model: ', model)
    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

    print("Loading full model from checkpoint " + str(args.load_model))

    state = utils.load_model(model, None, args.load_model, args.cuda)

    data_split = {"test": []}  # h5 files already saved
    shapes = {"output_frames": hparams['input_sample']}
    test_data = testDataset(sr=44100, hdf_dir="./hdf/", data_split=data_split,
                               partition="test", audio_dir=audio_dir, annot_dir=annot_dir, in_memory=False)

    results = utils.predict(args, model, test_data, device)

    print("Averaged F-measure:", results)


if __name__ == '__main__':
    ## EVALUATE PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader worker threads (default: 1)')
    parser.add_argument('--features', type=int, default=24,
                        help='Number of feature channels per layer')
    parser.add_argument('--audio_dir', type=str, default="/import/c4dm-05/jh008/jamendolyrics/",
                        help='Dataset path')
    parser.add_argument('--dataset', type=str, default="jamendo",
                        help='Dataset name')
    parser.add_argument('--hdf_dir', type=str, default="/import/c4dm-datasets/sepa_DALI/hdf/sepa=True/",
                        help='Dataset path')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='prediction path')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size")
    parser.add_argument('--sr', type=int, default=44100,
                        help="Sampling rate")
    parser.add_argument('--input_sample', type=int, default=110250,
                        help="Input samples")

    args = parser.parse_args()

    main(args)