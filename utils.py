import os

import torch
import torch.nn as nn
import numpy as np
import librosa
from madmom.features import DBNBeatTrackingProcessor
from madmom.features import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
import time
import warnings
from tqdm import tqdm
import mir_eval

from model import eval_audio_transforms

def load(path, sr=44100, mono=True, mode="numpy", offset=0.0, duration=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, curr_sr = librosa.load(path, sr=sr, mono=mono, res_type='kaiser_fast', offset=offset, duration=duration)

    if len(y.shape) == 1:
        # Expand channel dimension
        y = y[np.newaxis, :]

    if mode == "pytorch":
        y = torch.tensor(y)

    return y, curr_sr

def load_annot(beat_annot_file):

    with open(beat_annot_file, 'r') as f:
        beats_lines = f.read().splitlines()
    beats = []
    for line in beats_lines:
        time = float(line.split()[0])
        beat_no = int(line.split()[1])
        beats.append([time, beat_no])

    return np.array(beats)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def my_collate(batch):
    audio, beats = zip(*batch)
    audio = np.array(audio)
    beats = list(beats)
    return audio, beats

def set_lr(optim, lr):
    for g in optim.param_groups:
        g['lr'] = lr

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

def save_model(model, optimizer, state, path):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'state': state,
    }, path)

def load_model(model, optimizer, path, cuda):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # load state dict of wrapped module
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        # work-around for loading checkpoints where DataParallel was saved instead of inner module
        from collections import OrderedDict
        model_state_dict_fixed = OrderedDict()
        prefix = 'module.'
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith(prefix):
                k = k[len(prefix):]
            model_state_dict_fixed[k] = v
        model.load_state_dict(model_state_dict_fixed)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'state' in checkpoint:
        state = checkpoint['state']
    else:
        # older checkpoitns only store step, rest of state won't be there
        state = {"step": 0,
                 "worse_epochs": 0,
                 "epochs": checkpoint['epoch'],
                 "best_loss": np.Inf}
    return state

def validate(batch_size, model, criterion, dataloader, device):

    avg_time = 0.
    model.eval()
    total_loss = 0.
    data_len = len(dataloader.dataset) // batch_size

    with tqdm(total=data_len) as pbar:
        for batch_idx, _data in enumerate(dataloader):
            spectrograms, labels = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            t = time.time()

            output = model(spectrograms).squeeze(2)  # (batch, time, n_class)

            loss = criterion(output, labels)

            t = time.time() - t
            avg_time += (1. / float(batch_idx + 1)) * (t - avg_time)

            total_loss += loss.item()

            pbar.set_description("Current loss: {:.4f}".format(loss))
            pbar.update(1)

            if batch_idx == data_len:
                break

    return total_loss / data_len

def predict(args, model, test_data, device):

    dbn_downbeat = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=(44100 // 1024))

    dataloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             collate_fn=my_collate)
    model.eval()
    f_accum = 0.
    f_style_accum = {}
    f_accum_db = 0.
    f_style_accum_db = {}
    with tqdm(total=len(test_data)) as pbar, torch.no_grad():
        for example_num, _data in enumerate(dataloader):
            x, b = _data

            beats, downbeats, audio_name = b[0]
            style = audio_name.split('/')[0]

            x = move_data_to_device(x, device)
            x = x.squeeze(0)

            x = eval_audio_transforms(x)
            x = nn.utils.rnn.pad_sequence(x, batch_first=True).unsqueeze(1).transpose(2, 3)

            # predict
            all_outputs = model(x)

            _, total_length, num_classes = all_outputs.shape  # batch, length, classes

            song_pred = all_outputs.reshape(-1, num_classes)

            song_pred = torch.sigmoid(song_pred).data.numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                beat_info = dbn_downbeat(song_pred)
            predicted_beats = beat_info[:, 0]
            mask = (beat_info[:, 1] == 1)
            predicted_downbeats = beat_info[mask, 0]

            scores = mir_eval.beat.evaluate(np.array(beats), predicted_beats)
            f_accum += scores['F-measure']
            f_style_accum[style] = f_style_accum.get(style, []) + [scores['F-measure']]

            scores = mir_eval.beat.evaluate(np.array(downbeats), predicted_downbeats)
            f_accum_db += scores['F-measure']
            f_style_accum_db[style] = f_style_accum_db.get(style, []) + [scores['F-measure']]

            pbar.update(1)

    return f_accum / len(test_data), f_accum_db / len(test_data), f_style_accum, f_style_accum_db

def predict_madmom(audio_dir, annot_dir, split_file):

    db_proc = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=100)

    with open(split_file, "rb") as f:
        data_split = np.load(f, allow_pickle=True).item()

    f_accum = 0.
    f_accum_db = 0.
    with tqdm(total=len(data_split["test"])) as pbar:
        for audiofile in data_split["test"]:

            # fetch annotation
            audio_basename = os.path.basename(audiofile)
            beat_info = load_annot(os.path.join(annot_dir, audio_basename[:-4] + '.beats'))
            beats = beat_info[:, 0]
            mask = (beat_info[:, 1] == 1)
            downbeats = beat_info[mask, 0]

            # madmom prediction
            act = RNNDownBeatProcessor()(os.path.join(audio_dir, audiofile))
            beat_info = db_proc(act)

            predicted_beats = beat_info[:, 0]
            mask = (beat_info[:, 1] == 1)
            predicted_downbeats = beat_info[mask, 0]

            scores = mir_eval.beat.evaluate(np.array(beats), predicted_beats)
            f_accum += scores['F-measure']

            scores = mir_eval.beat.evaluate(np.array(downbeats), predicted_downbeats)
            f_accum_db += scores['F-measure']

            pbar.update(1)

    return f_accum / len(data_split["test"]), f_accum_db / len(data_split["test"])

def seed_torch(seed=0):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)