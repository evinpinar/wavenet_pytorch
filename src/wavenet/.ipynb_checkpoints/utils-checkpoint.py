import numpy as np
import torch
import librosa


def mulaw_encode(input, channels=256):
    n_q = channels - 1
    mu = torch.tensor(n_q, dtype=torch.float)
    audio = torch.tensor(input)
    # audio = torch.abs(torch.tensor(input))
    # audio_abs = torch.min(torch.abs(audio), 1.0)
    mag = torch.log1p(mu * torch.abs(audio)) / torch.log1p(mu)
    signal = torch.sign(audio) * mag
    out = ((signal + 1) / 2 * mu + 0.5).int()
    return out


def mulaw_decode(input, channels=256):
    n_q = channels - 1
    mu = torch.tensor(n_q, dtype=torch.float)
    audio = torch.tensor(input, dtype=torch.float)
    audio = (audio / mu) * 2 - 1
    out = torch.sign(audio) * (torch.exp(torch.abs(audio) * torch.log1p(mu)) - 1) / mu
    return out


def one_hot_encode(input, channels=256):
    size = input.size()[0]
    one_hot = torch.FloatTensor(size, channels)
    one_hot = one_hot.zero_()
    in1 = torch.tensor(input, dtype=torch.long)
    one_hot.scatter_(1, in1.unsqueeze(1), 1.0)
    return one_hot


def one_hot_decode(input):
    _, i = input.max(1)
    return torch.tensor(i)


def mel_resize(audio, S):
    factor = audio.size / S.shape[0]
    print(factor)
    mel = np.repeat(S, factor, axis=0)
    print(S.shape)
    print(mel.shape)
    mel_pad = audio.size - mel.shape[0]
    print(mel_pad)
    mel = np.pad(mel, [(0, mel_pad), (0, 0)], mode="constant", constant_values=0)
    print(mel.shape)
    return mel


def mel_resize2(audio_length, S):
    # Resize by the length of the audio
    factor = audio_length / S.shape[0]
    # print(factor)
    mel = np.repeat(S, factor, axis=0)
    # print(S.shape)
    # print(mel.shape)
    mel_pad = audio_length - mel.shape[0]
    # print(mel_pad)
    mel = np.pad(mel, [(0, mel_pad), (0, 0)], mode="constant", constant_values=0)
    # print("final mel shape:", mel.shape)
    return mel


def wav_to_spectrogram(wav, hparams):

    n_fft = ms_to_frames(hparams.fft_frame_size, hparams.sample_rate)
    hop_length = ms_to_frames(hparams.fft_hop_size, hparams.sample_rate)
    mel_spec = librosa.feature.melspectrogram(wav, sr= hparams.sample_rate,
                                              n_fft=n_fft,
                                              hop_length=hop_length,
                                              n_mels=hparams.num_mels,
                                              fmin=hparams.min_freq,
                                              fmax=hparams.max_freq)
    return librosa.power_to_db(mel_spec, ref=hparams.floor_freq).T


def ms_to_frames(ms, sample_rate):
    return int((ms / 1000) * sample_rate)


