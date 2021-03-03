import os

import torch
import torch.nn as nn
import numpy as np
import librosa
from madmom.features import DBNBeatTrackingProcessor
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

def worker_init_fn(worker_id): # This is apparently needed to ensure workers have different random seeds and draw different examples!
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def my_collate(batch):
    audio, beats = zip(*batch)
    audio = np.array(audio)
    beats = list(beats)
    return audio, beats

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
        model = model.module  # save state dict of wrapped module
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'state': state,  # state of training loop (was 'step')
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

    # VALIDATE
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
            output = torch.sigmoid(output)

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

    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)

    if not os.path.exists('pics'):
        os.makedirs('pics')

    dbn = DBNBeatTrackingProcessor(
        min_bpm=60,
        max_bpm=180,
        transition_lambda=100,
        fps=(44100 // 1024),
        online=True)

    # PREPARE DATA
    dataloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             collate_fn=my_collate)
    model.eval()
    f_accum = 0.
    with tqdm(total=len(test_data)) as pbar, torch.no_grad():
        for example_num, _data in enumerate(dataloader):
            x, beats = _data

            x = move_data_to_device(x, device)
            x = x.squeeze(0)

            # x = x.squeeze(1)
            x = eval_audio_transforms(x)
            x = nn.utils.rnn.pad_sequence(x, batch_first=True).unsqueeze(1).transpose(2, 3)

            # Predict
            all_outputs = model(x)

            batch_num, _, output_length = all_outputs.shape

            total_length = all_outputs.shape[1]

            print(all_outputs.shape) # batch, length, classes
            _, _, num_classes = all_outputs.shape

            song_pred = torch.sigmoid(all_outputs).data.numpy().reshape(-1, num_classes)
            # print(song_pred.shape) # total_length, num_classes

            resolution = 1024 / args.sr

            song_pred = song_pred[:total_length, 0]
            # print(song_pred.shape)  # total_length, num_classes

            dbn.reset()
            predicted_beats = dbn.process_offline(song_pred)

            scores = mir_eval.beat.evaluate(np.array(beats), predicted_beats)
            f_accum += scores['F-measure']

            # write
            # with open(os.path.join(args.pred_dir, audio_name + "_align.csv"), 'w') as f:
            #     for word in word_align:
            #         f.write("{},{}\n".format(word[0] * resolution, word[1] * resolution))

            pbar.update(1)

    return f_accum / len(test_data)

def seed_torch(seed=0):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)