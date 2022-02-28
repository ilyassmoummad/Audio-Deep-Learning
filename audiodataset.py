import os
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np


class AudioDataset(Dataset):
    def __init__(
        self,
        file_path,
        class_id,
        device,
        sampling_rate=16000,
        audio_target_duration=4,
        transform=None,
    ):
        self.file_path = file_path
        self.class_id = class_id
        self.device = device
        self.sampling_rate = sampling_rate
        self.audio_target_duration = audio_target_duration
        self.transform = transform

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        path = self.file_path[idx]
        waveform, sr = torchaudio.load(path)  # , normalization=True) # load audio
        resample = torchaudio.transforms.Resample(sr, self.sampling_rate)
        waveform = resample(waveform)
        audio_mono = torch.mean(waveform, dim=0, keepdim=True)  # Convert sterio to mono
        # audio_mono = audio_mono - audio_mono.mean() / audio_mono.std()
        wav_sample = audio_mono.shape[1]

        target_sample = self.audio_target_duration * self.sampling_rate
        tempData = torch.zeros([1, target_sample], dtype=torch.float32)

        diff_samples = target_sample - wav_sample

        if diff_samples < 0:
            tempData = audio_mono[:, -diff_samples:]
        elif diff_samples > 0:
            tempData[:, diff_samples:] = audio_mono
        else:
            tempData = audio_mono

        audio_mono = tempData
    
        melspec = torchaudio.transforms.MelSpectrogram(n_fft=1024, n_mels=64)
        melspecgram = melspec(audio_mono)
        melspecgram_norm = (melspecgram - melspecgram.mean()) / melspecgram.std()  # Noramalization

        return melspecgram_norm, torch.tensor(self.class_id[idx])

"""
if __name__ == "__main__":
    ANNOTATIONS_FILE = (
        "/users/local/i21moumm/soundata/urbansound8k/metadata/UrbanSound8K.csv"
    )
    AUDIO_DIR = "/users/local/i21moumm/soundata/urbansound8k/audio/"
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    df = pd.read_csv(ANNOTATIONS_FILE)
    df_train = df[df["fold"].isin(np.array(range(1, 9)))]
    train_files = df_train["slice_file_name"].values.tolist()
    train_fold = df_train["fold"].values
    train_labels = df_train["classID"].values.tolist()
    train_path = [
        os.path.join(AUDIO_DIR + "fold" + str(folder) + "/" + file)
        for folder, file in zip(train_fold, train_files)
    ]
    train_dataset = AudioDataset(
        file_path=train_path, class_id=train_labels, device=device
    )
    print(len(train_dataset))
    signal, label = train_dataset[0]
    print(signal.shape , signal.dtype)
"""
