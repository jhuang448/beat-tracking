import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

n_fft = 2048
n_class = 3

# spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft)
mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_mels=128, n_fft=n_fft)
# freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)

def train_audio_transforms(waveform):
    s = mel_spectrogram(waveform)

    spec = mel_spectrogram(waveform).squeeze(0) # (n_mel, time)

    return spec.transpose(0, 1)

def eval_audio_transforms(waveform):
    w = waveform

    spec = mel_spectrogram(w).squeeze(0) # (n_mel, time)

    return spec.transpose(0, 1).unsqueeze(0)

def data_processing(data):
    spectrograms = []
    labels = []

    for (waveform, beats_info) in data:
        beats, downbeats = beats_info
        waveform = torch.Tensor(waveform)
        spec = train_audio_transforms(waveform)
        spectrograms.append(spec)

        beats_frames = torch.zeros(size=(spec.shape[0],n_class))
        beats_pos = (beats // (n_fft//2)).astype(int) # beat class
        beats_frames[beats_pos, 1] = 1
        downbeats_pos = (downbeats // (n_fft//2)).astype(int) # downbeat class
        beats_frames[downbeats_pos, 2] = 2

        label = torch.Tensor(beats_frames)
        labels.append(label)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels


class BidirectionalLSTM(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalLSTM, self).__init__()

        self.BiLSTM = nn.LSTM(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        # self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = self.layer_norm(x)
        x, _ = self.BiLSTM(x)
        x = self.dropout(x)
        return x

class BeatTrackingModel(nn.Module):

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=1, dropout=0.1, input_sample=220500):
        super(BeatTrackingModel, self).__init__()

        self.cnn = nn.Conv2d(1, n_feats, (1,3), stride=1, padding=(0,1))
        self.fully_connected = nn.Linear(n_feats * 128, rnn_dim)

        self.bilstm_layers = nn.Sequential(*[
            BidirectionalLSTM(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, n_class)  # birnn returns rnn_dim*2
        )

    def forward(self, x):
        # x = self.cnn_layers(x)
        # print(x.shape)
        x = F.relu(self.cnn(x))
        # print(x.shape)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)

        # print(x.shape)
        x = F.relu(self.fully_connected(x))

        x = self.bilstm_layers(x)

        x = self.classifier(x)

        return x
