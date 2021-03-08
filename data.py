import h5py
import os
import numpy as np
from sortedcontainers import SortedList
import librosa
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import load_annot, load

def get_ballroom_folds(audio_dir):
    train_list_all = []
    val_list_all = []
    test_list_all = []
    for dir in os.listdir(audio_dir):
        if os.path.isdir(os.path.join(audio_dir, dir)) and dir != "nada":
            audio_list = os.listdir(os.path.join(audio_dir, dir))
            audio_list = ["{}/{}".format(dir, audio_name) for audio_name in audio_list]

            total_len = len(audio_list)

            train_len = np.int(0.8 * total_len)
            train_list = np.random.choice(audio_list, train_len, replace=False)
            val_test_list = [elem for elem in audio_list if elem not in train_list]

            val_len = np.int(0.5 * len(val_test_list))
            val_list = np.random.choice(val_test_list, val_len, replace=False)

            test_list = [elem for elem in val_test_list if elem not in val_list]

            assert(len(train_list)+len(val_list)+len(test_list) == total_len)

            train_list_all += list(train_list)
            val_list_all += list(val_list)
            test_list_all += test_list

    return {"train": train_list_all, "val": val_list_all, "test": test_list_all}

class BallroomDataset(Dataset):
    def __init__(self, sr, shapes, hdf_dir, data_split, partition, audio_dir, annot_dir, in_memory=False):
        super(BallroomDataset, self).__init__()
        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        self.hdf_file = os.path.join(hdf_dir, "Ballroom_" + partition + ".hdf5")

        self.sr = sr
        self.shapes = shapes
        self.hop = (shapes["output_frames"] // 2)
        self.in_memory = in_memory

        self.audio_list = data_split[partition]

        # check if hdf5 file exists already
        if not os.path.exists(self.hdf_file):
            # create folder if it did not exist before
            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)

            # create hdf5 file
            with h5py.File(self.hdf_file, "w") as f:
                f.attrs["sr"] = sr

                print("Adding audio files to dataset (preprocessing)...")
                for idx, audio_name in enumerate(tqdm(self.audio_list)):

                    # load song
                    y, _ = load(path=os.path.join(audio_dir, audio_name), sr=self.sr, mono=True)

                    audio_basename = os.path.basename(audio_name)
                    beats = load_annot(os.path.join(annot_dir, audio_basename[:-4]+'.beats'))
                    annot_num = len(beats)

                    # add to hdf5
                    grp = f.create_group(str(idx))
                    grp.create_dataset("inputs", shape=y.shape, dtype=y.dtype, data=y)

                    grp.attrs["annot_num"] = annot_num
                    grp.attrs["input_length"] = y.shape[1]
                    grp.attrs["audio_name"] = audio_name.encode() # audio_basename[:-4]
                    # print(len(beats))

                    grp.create_dataset("beats", shape=(annot_num, 2), dtype=np.float, data=beats)

        with h5py.File(self.hdf_file, "r", libver='latest', swmr=True) as f:
            if f.attrs["sr"] != sr:
                raise ValueError(
                    "Tried to load existing HDF file, but sampling rate is not as expected. Did you load an out-dated HDF file?")

        # collect lengths of all audio files
        with h5py.File(self.hdf_file, "r") as f:
            # length of song
            lengths = [f[str(song_idx)].attrs["input_length"] for song_idx in range(len(f))]

            # subtract input_size from lengths and divide by hop size to determine number of starting positions
            lengths = [np.int(np.ceil(l / self.hop)) for l in lengths]

        self.lengths = lengths
        self.start_pos = SortedList(np.cumsum(lengths))
        self.length = self.start_pos[-1]

    def __getitem__(self, index):

        # open hdf5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_file, 'r', driver=driver)

        # find out which slice of targets we want to read
        song_idx = self.start_pos.bisect_right(index)
        if song_idx > 0:
            index = index - self.start_pos[song_idx - 1]

        # check length of audio signal
        audio_length = self.hdf_dataset[str(song_idx)].attrs["input_length"]
        annot_num = self.hdf_dataset[str(song_idx)].attrs["annot_num"]

        # determine position where to start targets
        start_target_pos = index * self.hop
        end_target_pos = start_target_pos + self.shapes["output_frames"]

        # read
        start_pos = start_target_pos
        end_pos = end_target_pos
        if end_pos > audio_length:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0

        # read audio and zero padding
        audio = self.hdf_dataset[str(song_idx)]["inputs"][0, start_pos:end_pos].astype(np.float32)
        if pad_back > 0:
            audio = np.pad(audio, (0, pad_back), mode="constant", constant_values=0.0)

        # find the beats within (start_target_pos, end_target_pos)
        beats_pos = self.hdf_dataset[str(song_idx)]["beats"][:, 0]
        try:
            first_beat_to_include = next(x for x, val in enumerate(list(beats_pos))
                                         if val > start_target_pos / self.sr)
        except StopIteration:
            first_beat_to_include = np.Inf

        try:
            last_beat_to_include = annot_num - 1 - next(
                x for x, val in enumerate(reversed(list(beats_pos)))
                if val < end_target_pos / self.sr)
        except StopIteration:
            last_beat_to_include = -np.Inf

        beats = np.array([])
        downbeats = np.array([])

        if first_beat_to_include <= last_beat_to_include:
            beats_info = self.hdf_dataset[str(song_idx)]["beats"][first_beat_to_include:last_beat_to_include + 1, :]
            beats = beats_info[:, 0] * self.sr - start_pos
            downbeat_mask = (beats_info[:,1]==1)
            downbeats = beats_info[downbeat_mask, 0] * self.sr - start_pos

        return audio, (beats, downbeats)

    def __len__(self):
        return self.length

class testDataset(Dataset):
    def __init__(self, sr, hdf_dir, data_split, partition, audio_dir, annot_dir, in_memory=False):
        super(testDataset, self).__init__()
        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        self.hdf_file = os.path.join(hdf_dir, "Ballroom_" + partition + ".hdf5")

        self.sr = sr
        self.in_memory = in_memory

        self.audio_list = data_split[partition]

        with h5py.File(self.hdf_file, "r") as f:
            self.length = len(f)

    def __getitem__(self, index):

        # open hdf5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_file, 'r', driver=driver)

        song_idx = index
        audio_name = self.hdf_dataset[str(song_idx)].attrs["audio_name"].decode()

        # read audio and zero padding
        audio = self.hdf_dataset[str(song_idx)]["inputs"][0, :].astype(np.float32)

        # find the beats within (start_target_pos, end_target_pos)
        beats_pos = self.hdf_dataset[str(song_idx)]["beats"][:, 0]
        downbeat_mask = (self.hdf_dataset[str(song_idx)]["beats"][:, 1] == 1)
        downbeats_pos = self.hdf_dataset[str(song_idx)]["beats"][downbeat_mask, 0]

        return audio, (beats_pos, downbeats_pos, audio_name)

    def __len__(self):
        return self.length