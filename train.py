import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import time
from tqdm import tqdm

from data import get_ballroom_folds, BallroomDataset
from utils import validate, seed_torch, set_lr
from utils import worker_init_fn, load_model

from model import BeatTrackingModel, data_processing

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

audio_dir = "/import/c4dm-datasets/ballroom/BallroomData/"
annot_dir = "/import/c4dm-datasets/ballroom/jku-beat-annotations/"

def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, batch_size):
    avg_time = 0.
    model.train()
    data_len = len(train_loader.dataset) // batch_size
    train_loss = 0.

    with tqdm(total=data_len) as pbar:
        for batch_idx, _data in enumerate(train_loader):
            spectrograms, labels = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            t = time.time()

            optimizer.zero_grad()

            output = model(spectrograms).squeeze(2)  # (batch, time, n_class)
            # output = torch.sigmoid(output)
            # print(output.shape, labels.shape)
            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()

            t = time.time() - t
            avg_time += (1. / float(batch_idx + 1)) * (t - avg_time)

            train_loss += loss.item()

            pbar.set_description("Current loss: {:.4f}".format(loss))
            pbar.update(1)

            if batch_idx == data_len:
                break

    return train_loss / data_len


def main(args):
    hparams = {
        "n_cnn_layers": 2,
        "n_rnn_layers": 3,
        "rnn_dim": 25,
        "n_class": 2,
        "n_feats": 8,
        "dropout": 0.1,
        "stride": 1,
        "learning_rate": args.lr,
        "input_sample": args.input_sample,
        "batch_size": args.batch_size
    }

    use_cuda = torch.cuda.is_available()
    seed_torch(2724)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    model = BeatTrackingModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout'], hparams['input_sample']
    ).to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    ### DATASET
    # data_split = get_ballroom_folds(audio_dir)
    data_split = {"train": [], "val": []}  # h5 files already saved

    shapes = {"output_frames": hparams['input_sample']}

    val_data = BallroomDataset(sr=44100, shapes=shapes, hdf_dir="./hdf/", data_split=data_split,
                            partition="val", audio_dir=audio_dir, annot_dir=annot_dir, in_memory=False)
    train_data = BallroomDataset(sr=44100, shapes=shapes, hdf_dir="./hdf/", data_split=data_split,
                               partition="train", audio_dir=audio_dir, annot_dir=annot_dir, in_memory=False)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_data,
                                   batch_size=hparams["batch_size"],
                                   shuffle=True,
                                   worker_init_fn=worker_init_fn,
                                   collate_fn=lambda x: data_processing(x),
                                   **kwargs)
    val_loader = data.DataLoader(dataset=val_data,
                                   batch_size=hparams["batch_size"],
                                   shuffle=False,
                                   collate_fn=lambda x: data_processing(x),
                                   **kwargs)

    optimizer = optim.Adam(model.parameters(), hparams['learning_rate'])
    # pos_weight = torch.Tensor([1., 1., 1.])
    criterion = nn.BCEWithLogitsLoss().to(device) # nonbeat:beat:downbeat
    # downbeat_flag = False
    # downbeat_epoch = -1

    # Set up training state dict that will also be saved into checkpoints
    # Set up training state dict that will also be saved into checkpoints
    state = {"step": 0,
             "worse_epochs": 0,
             "epochs": 0,
             "best_loss": np.Inf}

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_model is not None:
        state = load_model(model, None, args.load_model, args.cuda)

    # iter_meter = IterMeter()

    from torch.utils.tensorboard import SummaryWriter
    import datetime
    current = datetime.datetime.now()
    writer = SummaryWriter(os.path.join(args.log_dir + current.strftime("%m:%d:%H:%M")))

    while state["worse_epochs"] < 10: # or downbeat_flag == True:
        print("Training one epoch from epoch " + str(state["epochs"]))

        # if downbeat_flag == True: # increase the weight of the downbeat class
        #     pos_weight = torch.Tensor([1., 1., 0.4 * downbeat_epoch]) # update weight
        #     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device) # update criterion
        #     downbeat_epoch += 1

        lr = hparams['learning_rate'] / (((state["epochs"] // (20 * 1)) * 2) + 1)
        set_lr(optimizer, lr)
        writer.add_scalar("train/learning_rate", lr, state["epochs"])

        # train
        train_loss = train(model, device, train_loader, criterion, optimizer, None, state["epochs"], None, args.batch_size)
        writer.add_scalar("train/epoch_loss", train_loss, state["epochs"])

        val_loss = validate(args.batch_size, model, criterion, val_loader, device)
        print("VALIDATION FINISHED: LOSS: " + str(val_loss))
        writer.add_scalar("val/loss", val_loss, state["epochs"])

        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["epochs"]))
        if val_loss >= state["best_loss"]:
            state["worse_epochs"] += 1
        else:
            print("MODEL IMPROVED ON VALIDATION SET!")
            state["worse_epochs"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = checkpoint_path

        print("Saving model... best_epoch {} best_loss {}".format(state["best_checkpoint"], state["best_loss"]))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'state': state
        }, checkpoint_path)

        state["epochs"] += 1

        # if state["worse_epochs"] == 10:
        #     downbeat_flag = True
        #     downbeat_epoch = 1
        #     state["worse_epochs"] = 0 # do not stop here
        #
        # if downbeat_epoch == 11:
        #     downbeat_flag = False
        #     state["worse_epochs"] = 0

    writer.close()

if __name__ == '__main__':

    ## TRAIN PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--dummy', action='store_true',
                        help='Use dummy train/val sets (default: False)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of data loader worker threads (default: 1)')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Folder to write logs into')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Folder to write checkpoints into')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size")
    parser.add_argument('--sr', type=int, default=44100,
                        help="Sampling rate")
    parser.add_argument('--input_sample', type=int, default=220500,
                        help="Input sample")

    args = parser.parse_args()
    print(args)

    main(args)