import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabulary: space + Arabic letters + basic Latin/Arabizi characters
VOCAB = " _أإآابتثجحخدذرزسشصضطظعغفقكلمنهويىة1234567890abcdefghijklmnopqrstuvwxyz"
CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

def text_to_sequence(text: str):
    """Converts normalized string text to tensor of token IDs."""
    sequence = [CHAR_TO_ID.get(char, CHAR_TO_ID['_']) for char in text]
    return torch.tensor(sequence, dtype=torch.long)


class TextEncoder(nn.Module):
    """
    Lightweight 1D-CNN based Text Encoder for fast CPU execution.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=5, padding=2)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        emb = self.embedding(x) # (batch_size, seq_len, embed_dim)
        h = emb.transpose(1, 2) # (batch_size, embed_dim, seq_len)

        h = F.relu(self.conv1(h))
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = self.dropout(h)

        h = F.relu(self.conv2(h))
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)

        return h.transpose(1, 2) # (batch_size, seq_len, hidden_dim)


class DurationPredictor(nn.Module):
    """
    Predicts duration (number of mel frames) for each input character.
    """
    def __init__(self, in_dim=256, filter_dim=256, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, filter_dim, kernel_size=kernel_size, padding=1)
        self.norm1 = nn.LayerNorm(filter_dim)
        self.conv2 = nn.Conv1d(filter_dim, filter_dim, kernel_size=kernel_size, padding=1)
        self.norm2 = nn.LayerNorm(filter_dim)
        self.linear = nn.Linear(filter_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, in_dim)
        h = x.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = F.relu(self.conv2(h))
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        out = self.linear(h.transpose(1, 2)).squeeze(-1) # (batch_size, seq_len)
        return F.relu(out)


class LengthRegulator(nn.Module):
    """
    Expands encoder output sequence length according to character durations.
    """
    def __init__(self):
        super().__init__()

    def forward(self, encoder_out, durations, max_mel_len=None):
        # encoder_out: (batch_size, seq_len, hidden_dim)
        # durations: (batch_size, seq_len)
        batch_size, seq_len, hidden_dim = encoder_out.shape
        repeated_list = []

        for b in range(batch_size):
            rep_b = []
            for i in range(seq_len):
                dur = int(torch.round(durations[b, i]).item())
                dur = max(1, dur)
                rep_b.append(encoder_out[b, i:i+1].expand(dur, -1))
            rep_b_cat = torch.cat(rep_b, dim=0)
            repeated_list.append(rep_b_cat)

        # Find max frame length in batch or pad to max_mel_len
        target_len = max_mel_len if max_mel_len is not None else max(x.size(0) for x in repeated_list)
        padded_batch = []
        for rep in repeated_list:
            if rep.size(0) < target_len:
                pad = torch.zeros(target_len - rep.size(0), hidden_dim, device=encoder_out.device)
                rep = torch.cat([rep, pad], dim=0)
            else:
                rep = rep[:target_len]
            padded_batch.append(rep.unsqueeze(0))

        return torch.cat(padded_batch, dim=0)


class MelDecoder(nn.Module):
    """
    Converts length-regulated hidden representations into 80-channel log mel-spectrograms.
    """
    def __init__(self, in_dim=256, n_mels=80, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=5, padding=2)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mel_out = nn.Linear(hidden_dim, n_mels)

    def forward(self, x):
        # x shape: (batch_size, mel_len, in_dim)
        h = x.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = F.relu(self.conv2(h))
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        mel_pred = self.mel_out(h.transpose(1, 2)) # (batch_size, mel_len, n_mels)
        return mel_pred


class DarjaTTSModel(nn.Module):
    """
    Complete Lightweight FastSpeech2-like Neural TTS Model for Algerian Darja.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, n_mels=80, hidden_dim=256):
        super().__init__()
        self.encoder = TextEncoder(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.duration_predictor = DurationPredictor(in_dim=hidden_dim, filter_dim=hidden_dim)
        self.length_regulator = LengthRegulator()
        self.decoder = MelDecoder(in_dim=hidden_dim, n_mels=n_mels, hidden_dim=hidden_dim)

    def forward(self, text_seq, target_durations=None, max_mel_len=None):
        # text_seq: (batch_size, text_len)
        enc_out = self.encoder(text_seq)

        predicted_durations = self.duration_predictor(enc_out)

        # During training, use target_durations if available
        durations = target_durations if target_durations is not None else predicted_durations

        expanded_enc = self.length_regulator(enc_out, durations, max_mel_len=max_mel_len)
        mel_pred = self.decoder(expanded_enc)

        return mel_pred, predicted_durations


if __name__ == "__main__":
    # Smoke test model architecture
    model = DarjaTTSModel()
    sample_text = text_to_sequence("واش راك لاباس").unsqueeze(0)
    mel_out, dur_out = model(sample_text)
    print("Model Output Shape:", mel_out.shape)
    print("Duration Output Shape:", dur_out.shape)
