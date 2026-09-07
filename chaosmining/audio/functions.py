import os
import io
os.environ["HF_DATASETS_DISABLE_TORCHCODEC"] = "1"
import torch
import torchaudio
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from datasets import load_dataset
from datasets.features import Audio, Value, Features
import soundfile as sf

class HFChaosMiningAudioDataset(Dataset):
    def __init__(self, subset_name: str, split: str, target_length: int = 16000):
        if split not in ["train", "val"]:
            raise ValueError("split must be 'train' or 'val'")
        hf_split = "validation" if split == "val" else split

        # self.ds = load_dataset(
        #     path="geshijoker/chaosmining",
        #     name=subset_name,
        #     split=hf_split,
        #     columns=["label", "audio", "position"],
        #     features=Features({
        #         "label": Value("string"),
        #         "audio": Audio(sampling_rate=16000, decode=True), 
        #         "position": Value("int32")
        #     }),
        # )
        self.ds = load_dataset(
            "geshijoker/chaosmining",
            name=subset_name,
            split=hf_split,
            columns=["label", "audio", "position"],
            features=Features({
                "label": Value("string"),
                "audio": Audio(decode=False),  
                "position": Value("int32")
            }),
        )


        self.classes = sorted(set(self.ds["label"]))
        self.label_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.sample_rate = 16000
        self.target_length = target_length

        print(
            f"successfully load {split}  {len(self.ds)} samples, "
            f"{len(self.classes)} classes, sr={self.sample_rate}"
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        try:
            item = self.ds[idx]
            cla = self.label_to_idx[item["label"]]
            pos = int(item["position"])

            # waveform = torch.tensor(item["audio"]["array"], dtype=torch.float32).unsqueeze(0)

            # if waveform.shape[1] > self.target_length:
            #     waveform = waveform[:, :self.target_length]
            # elif waveform.shape[1] < self.target_length:
            #     pad = self.target_length - waveform.shape[1]
            #     waveform = torch.nn.functional.pad(waveform, (0, pad))
            audio_bytes = item["audio"]["bytes"]
            waveform_np, sr = sf.read(io.BytesIO(audio_bytes))  # [samples, channels]
            waveform = torch.tensor(waveform_np, dtype=torch.float32).T  # [channels, samples]
            if waveform.shape[1] > self.target_length:
                waveform = waveform[:, :self.target_length]
            elif waveform.shape[1] < self.target_length:
                pad = self.target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad))

            return waveform, cla, pos, self.sample_rate
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            raise e
# def collate_fn(batch):
#     tensors, targets = [], []
#     for waveform, label, *_ in batch:  
#         tensors.append(waveform)
#         targets.append(label)
#     return torch.stack(tensors), torch.tensor(targets)
def collate_fn(batch):
    tensors, targets, positions, sample_rates = [], [], [], []

    for waveform, label, pos, sr in batch:
        tensors.append(waveform)
        targets.append(label)
        positions.append(pos) 
        sample_rates.append(sr)  
    tensors = torch.stack(tensors)  # [batch_size, channels, length]
    targets = torch.tensor(targets)  # [batch_size]
    
    return tensors, targets, positions, sample_rates

def label_to_index(word, labels):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))

def index_to_label(index, labels):
    # Return the word corresponding to the index in labels
    return labels[index]