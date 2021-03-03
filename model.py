import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

n_fft = 2048

# spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft)
mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_mels=128, n_fft=n_fft)

def train_audio_transforms(waveform):
    s = mel_spectrogram(waveform)
    energy = torch.sum(torch.pow(s, 2), dim=0, keepdim=True)

    spec = mel_spectrogram(waveform).squeeze(0) # (n_mel, time)
    spec_pad = F.pad(spec, (1, 0), "constant", 0)
    delta_spec = spec_pad[:, 1:] - spec_pad[:, :-1]
    # delta_spec[delta_spec<0] = 0

    spec_target = torch.cat((spec, delta_spec), dim=0).transpose(0, 1)

    return spec_target

def eval_audio_transforms(waveform):
    eval_spec_target = []

    # for i in range(waveform.shape[0]):
    w = waveform
    # s = spectrogram(w)

    spec = mel_spectrogram(w).squeeze(0) # (n_mel, time)
    spec_pad = F.pad(spec, (1, 0), "constant", 0)
    delta_spec = spec_pad[:, 1:] - spec_pad[:, :-1]
    delta_spec[delta_spec < 0] = 0

    spec_target = torch.cat((spec, delta_spec), dim=0).transpose(0, 1)

    return spec_target.unsqueeze(0)

def data_processing(data):
    spectrograms = []
    labels = []

    for (waveform, beats_info) in data:
        beats, downbeats = beats_info
        waveform = torch.Tensor(waveform)
        spec = train_audio_transforms(waveform)
        spectrograms.append(spec)

        beats_frames = torch.zeros(size=(spec.shape[0],))
        beats_pos = (beats // (n_fft//2)).astype(int) # beat class
        beats_frames[beats_pos] = 1
        downbeats_pos = (downbeats // (n_fft//2)).astype(int) # downbeat class
        # beats_frames[downbeats_pos] = 2

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

        # n residual cnn layers with filter size of 32
        # self.cnn_layers = nn.Sequential(
        #     nn.Conv2d(1, n_feats, 3, stride=stride, padding=3 // 2),
        #     nn.MaxPool2d(kernel_size=(2, 2)),
        #     nn.Conv2d(n_feats, n_feats, 3, stride=stride, padding=3 // 2),
        #     nn.MaxPool2d(kernel_size=(2, 2))
        # )

        # self.linear = nn.Linear(2048, rnn_dim)

        self.bilstm_layers = nn.Sequential(*[
            BidirectionalLSTM(rnn_dim=256 if i == 0 else rnn_dim * 2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, n_class)  # birnn returns rnn_dim*2
        )

    def forward(self, x):
        # x = self.cnn_layers(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)

        # x = self.linear(x)

        x = self.bilstm_layers(x)

        x = self.classifier(x)

        return x
