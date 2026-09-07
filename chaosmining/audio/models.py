import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torchaudio import models, transforms
from torchaudio.models import Wav2Vec2Model, RNNT, Conformer,emformer_rnnt_base
# from torchaudio.pipelines import WAV2VEC2_BASE, RNNT_BASE_LIBRISPEECH, CONFORMER_BASE_LIBRISPEECH
from torchaudio.functional import rnnt_loss

class AudioRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(AudioRNN, self).__init__()

        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size=80, stride=16)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.pool1 = nn.MaxPool1d(4)
        
        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=0.1)   
        # Fully connected layer
        self.fc1 = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        
        batch_size = x.size(0)
        x = x.transpose(1, 2)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size).to(x.device)

        # Passing in the input and hidden state into the model and obtaining outputs
        x, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(x, x.shape[-1])
        x = torch.squeeze(x)
        x = self.fc1(x)
        
        return x
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

class AudioLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(AudioLSTM, self).__init__()

        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size=80, stride=16)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.pool1 = nn.MaxPool1d(4)
        
        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=0.1)   
        # Fully connected layer
        self.fc1 = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        
        batch_size = x.size(0)
        x = x.transpose(1, 2)

        # Initializing hidden state for first input using method defined below
        h, c = self.init_hidden(batch_size)
        h = h.to(x.device)
        c = c.to(x.device)

        # Passing in the input and hidden state into the model and obtaining outputs
        x, (h, c) = self.lstm(x, (h, c))
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(x, x.shape[-1])
        x = torch.squeeze(x)
        x = self.fc1(x)
        
        return x
        
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        h0 = torch.randn(self.n_layers, batch_size, self.hidden_dim)
        c0 = torch.randn(self.n_layers, batch_size, self.hidden_dim)
        return h0, c0

class AudioTCN(nn.Module):
    def __init__(self, n_input=1, n_output=35, n_channel=32):
        super(AudioTCN, self).__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=16)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.max_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = torch.squeeze(x)
        x = self.fc1(x)
        return x

class AudioTrans(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(AudioTrans, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.conv1 = nn.Conv1d(input_size, self.hidden_dim, kernel_size=80, stride=16)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.pool1 = nn.MaxPool1d(4)

        #Defining the layers
        # RNN Layer
        encoder_layer = nn.TransformerEncoderLayer(self.hidden_dim, nhead=6)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Fully connected layer
        self.fc1 = nn.Linear(self.hidden_dim, output_size)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        
        batch_size = x.size(0)
        x = x.transpose(1, 2)

        # Initializing hidden state for first input using method defined below

        # Passing in the input and hidden state into the model and obtaining outputs
        x = self.transformer(x)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(x, x.shape[-1])
        x = torch.squeeze(x)
        x = self.fc1(x)
        
        return x
class AudioWav2Vec2(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=768, freeze_feature_extractor=False):
        super(AudioWav2Vec2, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.freeze_feature_extractor = freeze_feature_extractor

        # 1. Feature Extractor（BatchNorm1d适配[B,C,T]）
        class Wav2Vec2FeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=0),
                    nn.BatchNorm1d(512),
                    nn.GELU(),
                    nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(512),
                    nn.GELU(),
                    nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(512),
                    nn.GELU(),
                    nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(512),
                    nn.GELU(),
                    nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(512),
                    nn.GELU(),
                    nn.Conv1d(512, 512, kernel_size=2, stride=2, padding=0),
                    nn.BatchNorm1d(512),
                    nn.GELU(),
                    nn.Conv1d(512, 512, kernel_size=2, stride=2, padding=0),
                    nn.BatchNorm1d(512),
                    nn.GELU(),
                )
            
            def forward(self, x, lengths=None):
                x = self.layers(x)
                return x, lengths

        self.feature_extractor = Wav2Vec2FeatureExtractor()

        # 2. Encoder（残差修复+维度适配+lengths兼容，最终版）
        class Wav2Vec2Encoder(nn.Module):
            def __init__(self, embed_dim, num_layers, num_heads, ff_dim):
                super().__init__()
                self.proj = nn.Linear(512, embed_dim)
                # 核心修复：padding=same 保证残差长度一致
                self.pos_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=128, padding="same", groups=16)
                self.layer_norm = nn.LayerNorm(embed_dim)
                self.dropout = nn.Dropout(0.1)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
                    dropout=0.1, activation="gelu", batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            def forward(self, x, lengths=None):
                # 维度流转：[B,C,T] → [B,T,C] 适配Linear
                x = x.transpose(1, 2)
                x = self.proj(x)

                # 残差连接：保证维度完全匹配（再也不报错）
                x_feat = x.transpose(1, 2)
                x_pos = self.pos_conv(x_feat)
                x = (x_feat + x_pos).transpose(1, 2)

                # 归一化+dropout+Transformer
                x = self.layer_norm(x)
                x = self.dropout(x)
                x = self.transformer(x)
                return x
        
        self.encoder = Wav2Vec2Encoder(
            embed_dim=hidden_dim, num_layers=12, num_heads=12, ff_dim=3072
        )
        self.wav2vec2 = Wav2Vec2Model(
            feature_extractor=self.feature_extractor,
            encoder=self.encoder,
            aux=None
        )
        if self.freeze_feature_extractor:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.input_adapt = nn.Conv1d(input_size, 1, kernel_size=1) if input_size != 1 else nn.Identity()

    def forward(self, x):
        # [B,T] → [B,input_size,T] → [B,1,T]
        if len(x.shape) == 2:
            x = x.unsqueeze(1).repeat(1, self.input_size, 1)
        x = self.input_adapt(x)
        outputs, _ = self.wav2vec2(x)
        feat = outputs

        feat = feat.transpose(1, 2)
        feat_pool = self.pool(feat).squeeze(-1)
        out = self.fc(feat_pool)
        return out
class AudioRNNT(nn.Module):
    """
    RNNT encoder as feature extractor + classification head
    """

    def __init__(
        self,
        input_size,
        num_classes: int,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
    ):
        super().__init__()
        self.input_size = input_size
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # === Feature extractor===
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.mel_bn = nn.BatchNorm1d(n_mels)

        # === RNNT (only encoder used) ===
        self.rnnt = torchaudio.models.emformer_rnnt_base(
            num_symbols=num_classes
        )
        encoder_dim = 1024  # emformer_rnnt_base

        # === Pool + classifier ===
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(encoder_dim, num_classes)

    def _pad_to_min_length(self, waveform):
        min_len = self.n_fft
        if waveform.size(-1) < min_len:
            pad_len = min_len - waveform.size(-1)
            waveform = F.pad(waveform, (0, pad_len), mode='constant')
        return waveform

    def extract_features(self, waveform):
        if waveform.dim() == 4:
            waveform = waveform.squeeze(dim=1)
        elif waveform.dim() == 3:
            waveform = waveform.mean(dim=1)
        elif waveform.dim() == 2:
            pass
        else:
            raise ValueError(f"only support 2D/3D/4Dinput,now:{waveform.dim()}D")
        waveform = self._pad_to_min_length(waveform)
        x = self.mel(waveform)  
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = x.squeeze() if x.dim() > 3 else x

        # BatchNorm1d adapt（3D [B,n_mels,T]）
        x = F.relu(self.mel_bn(x))
        x = x.transpose(1, 2)  # [B,T,n_mels] → adapt RNNT encoder
        lengths = torch.full(
            (x.size(0),),
            x.size(1),
            dtype=torch.long,
            device=x.device,
        )
        return x, lengths

    def forward(self, inputs):
        if isinstance(inputs, dict):
            waveform = inputs["waveform"]
        elif isinstance(inputs, (list, tuple)):
            waveform = inputs[0]
        else:
            waveform = inputs
        feats, lengths = self.extract_features(waveform)
        enc, enc_lens = self.rnnt.transcribe(feats, lengths)
        enc = enc.transpose(1, 2)       
        pooled = self.pool(enc).squeeze(-1)
        logits = self.fc(pooled)

        return logits

# -------------------------- AudioConformer--------------------------
class AudioConformer(nn.Module):
    def __init__(self, input_size, output_size, conformer_dim=256, n_layers=8, n_heads=4, ffn_dim=1024, conv_kernel_size=31):
        super(AudioConformer, self).__init__()

        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=16000, n_mels=80, n_fft=400, hop_length=160
        )
        self.mel_bn = nn.BatchNorm1d(80)
        self.mel_dropout = nn.Dropout(0.1)

        # Conformer
        self.conformer = Conformer(
            input_dim=80,
            num_heads=n_heads,
            ffn_dim=ffn_dim,
            num_layers=n_layers,
            depthwise_conv_kernel_size=conv_kernel_size,
            dropout=0.1
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(80, output_size)

    def forward(self, x):
        # x: [B, C, T] or [C, T] 
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, T] → [B, T]
        if x.dim() == 3:
            x = x.mean(dim=1)  # [B, T]

        mel = self.mel_transform(x)        # [B, n_mels, T]
        mel = F.relu(self.mel_bn(mel))    # BatchNorm1d
        mel = self.mel_dropout(mel)
        mel = mel.transpose(1, 2)         # [B, T, n_mels] adapt Conformer

        # Conformer
        lengths = torch.full((mel.size(0),), mel.size(1), dtype=torch.long, device=mel.device)
        conformer_out, _ = self.conformer(mel, lengths)  # [B, T, input_dim]

        # Pool + classifier
        conformer_feat = conformer_out.transpose(1, 2)   # [B, input_dim, T]
        feat_pool = self.pool(conformer_feat).squeeze(-1)  # [B, input_dim]
        return self.fc(feat_pool)