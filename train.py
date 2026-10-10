import os
import csv
import math
import wave
import struct
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import DarjaTTSModel, text_to_sequence

def hz_to_mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def create_mel_filterbank(sample_rate=22050, n_fft=1024, n_mels=80, f_min=0.0, f_max=8000.0):
    """
    Creates triangular mel-filterbank matrix of shape (n_fft // 2 + 1, n_mels).
    """
    n_freqs = n_fft // 2 + 1
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max if f_max is not None else sample_rate / 2)

    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = torch.tensor([mel_to_hz(m.item()) for m in mel_pts])
    bin_pts = torch.floor((n_fft + 1) * hz_pts / sample_rate).long()

    fb = torch.zeros((n_freqs, n_mels), dtype=torch.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_pts[m - 1].item()
        f_m = bin_pts[m].item()
        f_m_plus = bin_pts[m + 1].item()

        if f_m > f_m_minus:
            for k in range(f_m_minus, f_m):
                fb[k, m - 1] = (k - f_m_minus) / (f_m - f_m_minus)
        if f_m_plus > f_m:
            for k in range(f_m, f_m_plus):
                fb[k, m - 1] = (f_m_plus - k) / (f_m_plus - f_m)

    return fb


# Global Mel filterbank instance
MEL_FB = create_mel_filterbank()


def wave_to_mel(filepath, sample_rate=22050, n_mels=80, n_fft=1024, hop_length=256):
    """
    Reads WAV file and computes accurate log-mel spectrogram features using STFT.
    """
    with wave.open(filepath, 'rb') as wf:
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    int_samples = struct.unpack(f"<{n_frames * n_channels}h", raw_bytes)
    audio = torch.tensor([float(s) / 32768.0 for s in int_samples[::n_channels]], dtype=torch.float32)

    window = torch.hann_window(n_fft)
    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, return_complex=True)
    magnitude = torch.abs(stft) # shape: (n_fft // 2 + 1, num_frames)

    # Project to mel scale: (num_frames, n_fft // 2 + 1) x (n_fft // 2 + 1, n_mels) -> (num_frames, n_mels)
    mel_spec = torch.matmul(magnitude.transpose(0, 1), MEL_FB)
    log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))

    return log_mel


class DarjaSpeechDataset(Dataset):
    """
    Custom Dataset for Algerian Darja speech-text pairs.
    """
    def __init__(self, csv_file, audio_dir):
        self.audio_dir = audio_dir
        self.data = []
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            header = next(reader)
            for row in reader:
                if len(row) >= 5:
                    wav_name, _, norm_arabic, _, norm_arabizi = row[:5]
                    self.data.append({
                        "wav_name": wav_name,
                        "text_arabic": norm_arabic,
                        "text_arabizi": norm_arabizi
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        wav_path = os.path.join(self.audio_dir, item["wav_name"])
        mel_spec = wave_to_mel(wav_path)

        text_seq = text_to_sequence(item["text_arabic"])

        # Calculate proportional duration target per character
        seq_len = len(text_seq)
        mel_len = mel_spec.size(0)
        durations = torch.full((seq_len,), fill_value=max(1, mel_len / seq_len), dtype=torch.float32)

        return text_seq, mel_spec, durations


def collate_fn(batch):
    texts, mels, durations = zip(*batch)

    max_text_len = max(t.size(0) for t in texts)
    max_mel_len = max(m.size(0) for m in mels)

    padded_texts = torch.zeros(len(texts), max_text_len, dtype=torch.long)
    padded_mels = torch.zeros(len(mels), max_mel_len, 80, dtype=torch.float32)
    padded_durs = torch.zeros(len(durations), max_text_len, dtype=torch.float32)

    for i in range(len(batch)):
        padded_texts[i, :texts[i].size(0)] = texts[i]
        padded_mels[i, :mels[i].size(0), :] = mels[i]
        padded_durs[i, :durations[i].size(0)] = durations[i]

    return padded_texts, padded_mels, padded_durs, max_mel_len


def train(epochs=5, batch_size=2, lr=1e-3, checkpoint_path="darja_tts_checkpoint.pt"):
    print(f"Starting Algerian Darja TTS Fine-Tuning ({epochs} Epochs)...")

    dataset = DarjaSpeechDataset("dataset/metadata.csv", "dataset/wavs")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = DarjaTTSModel()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mel_criterion = nn.MSELoss()
    dur_criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for text_batch, mel_batch, dur_batch, max_mel_len in dataloader:
            optimizer.zero_grad()

            mel_pred, dur_pred = model(text_batch, target_durations=dur_batch, max_mel_len=max_mel_len)

            loss_mel = mel_criterion(mel_pred, mel_batch)
            loss_dur = dur_criterion(dur_pred, dur_batch)
            loss = loss_mel + 0.1 * loss_dur

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab': model.encoder.embedding.num_embeddings
    }, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train lightweight Algerian Darja TTS model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
