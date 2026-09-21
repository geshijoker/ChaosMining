import glob
import pyarrow.parquet as pq
import numpy as np
import torch

class ParquetAudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, target_length=16000):
        parquet_files = glob.glob(f"{data_dir}/*.parquet")

        dfs = [pq.read_table(p).to_pandas() for p in parquet_files]
        self.df = pd.concat(dfs, ignore_index=True)

        self.labels = sorted(self.df["label"].unique())
        self.label_to_idx = {lab: i for i, lab in enumerate(self.labels)}
        self.target_length = target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw = row["audio"]["bytes"]

        waveform = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        waveform = torch.tensor(waveform) / 32768.0

        if len(waveform) < self.target_length:
            waveform = F.pad(waveform, (0, self.target_length - len(waveform)))
        else:
            waveform = waveform[:self.target_length]

        waveform = waveform.unsqueeze(0)

        label = self.label_to_idx[row["label"]]

        return waveform, label
