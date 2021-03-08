import argparse, warnings

import torch
import torch.nn as nn
from madmom.features import DBNDownBeatTrackingProcessor

from data import testDataset
from model import BeatTrackingModel, eval_audio_transforms
import utils

audio_dir = "/import/c4dm-datasets/ballroom/BallroomData/"
annot_dir = "/import/c4dm-datasets/ballroom/jku-beat-annotations/"

hparams = {
        "n_rnn_layers": 3,
        "rnn_dim": 25,
        "n_class": 2,
        "n_feats": 8,
        "dropout": 0.1,
        "batch_size": 1
}

model = BeatTrackingModel(
        hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['dropout']
)

def beatTracker(inputFile):

    dbn_downbeat = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=(44100 // 1024))

    state = utils.load_model(model, None, "checkpoints/checkpoint_best", False)

    y, _ = utils.load(inputFile, sr=44100, mono=True)
    x = torch.Tensor(y)

    x = eval_audio_transforms(x)
    x = nn.utils.rnn.pad_sequence(x, batch_first=True).unsqueeze(1).transpose(2, 3)

    # predict
    all_outputs = model(x)

    _, _, num_classes = all_outputs.shape

    song_pred = all_outputs.reshape(-1, num_classes)
    song_pred = torch.sigmoid(song_pred).data.numpy()

    # dbn decoding
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        beat_info = dbn_downbeat(song_pred)
    predicted_beats = beat_info[:, 0]
    mask = (beat_info[:, 1] == 1)
    predicted_downbeats = beat_info[mask, 0]

    return predicted_beats, predicted_downbeats

def main(args):

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    global hparams
    global model

    if 'cuda' in device:
        print("move model to gpu")
        model = utils.DataParallel(model)
        model.cuda()

    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

    print("Loading full model from checkpoint " + str(args.load_model))

    # load model
    state = utils.load_model(model, None, args.load_model, args.cuda)

    data_split = {"test": []}  # h5 files already saved
    test_data = testDataset(sr=args.sr, hdf_dir="./hdf/", data_split=data_split,
                               partition="test", audio_dir=audio_dir, annot_dir=annot_dir, in_memory=False)

    # evaluate
    results = utils.predict(args, model, test_data, device)
    print("Averaged F-measure (beat, downbeat):", results[0], results[1])

    # print("Style-wise F-measures (beat, downbeat)", results[2], results[3])

    # uncomment for madmom evaluation
    # results = utils.predict_madmom(audio_dir, annot_dir, "data_split.npz")
    # print("Averaged F-measure (madmom):", results[0], results[1])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader worker threads (default: 1)')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size")
    parser.add_argument('--sr', type=int, default=44100,
                        help="Sampling rate")

    args = parser.parse_args()

    main(args)