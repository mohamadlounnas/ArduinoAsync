import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import DarjaTTSModel
from train import DarjaSpeechDataset, collate_fn
from infer import synthesize

def evaluate_model(checkpoint_path="darja_tts_checkpoint.pt"):
    print("=" * 60)
    print("ALGERIAN DARJA LIGHTWEIGHT TTS EVALUATION")
    print("=" * 60)

    # 1. Dataset Evaluation & Loss Metrics
    dataset = DarjaSpeechDataset("dataset/metadata.csv", "dataset/wavs")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = DarjaTTSModel()
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[OK] Checkpoint successfully loaded from '{checkpoint_path}'")
    else:
        print(f"[ERROR] Checkpoint '{checkpoint_path}' not found!")
        return False

    model.eval()
    mel_criterion = nn.MSELoss()
    dur_criterion = nn.MSELoss()

    total_mel_loss = 0.0
    total_dur_loss = 0.0

    with torch.no_grad():
        for text_batch, mel_batch, dur_batch, max_mel_len in dataloader:
            mel_pred, dur_pred = model(text_batch, target_durations=dur_batch, max_mel_len=max_mel_len)

            mel_loss = mel_criterion(mel_pred, mel_batch)
            dur_loss = dur_criterion(dur_pred, dur_batch)

            total_mel_loss += mel_loss.item()
            total_dur_loss += dur_loss.item()

    avg_mel_loss = total_mel_loss / len(dataloader)
    avg_dur_loss = total_dur_loss / len(dataloader)

    print(f"\nModel Loss Evaluation:")
    print(f" - Mel Spectrogram Reconstruction MSE Loss: {avg_mel_loss:.4f}")
    print(f" - Duration Predictor MSE Loss:            {avg_dur_loss:.4f}")

    # 2. Benchmark Inference Performance & RTF across test test cases
    test_phrases = [
        "واش راك لاباس الحمد لله",                      # Arabic script
        "sabah el kher khouya ki rak dayer",          # Arabizi script
        "رانا هنا باش نتعلمو ونطورو الجزائر",          # Hybrid Darja phrase
    ]

    print("\nInference Performance & RTF Benchmarks:")
    print("-" * 60)
    rtf_list = []

    for idx, phrase in enumerate(test_phrases, 1):
        out_file = f"eval_sample_{idx}.wav"
        print(f"\nTest {idx}: '{phrase}'")
        _, rtf = synthesize(phrase, checkpoint_path=checkpoint_path, output_wav=out_file)
        rtf_list.append(rtf)
        assert os.path.exists(out_file), f"Output file {out_file} was not generated!"

    avg_rtf = sum(rtf_list) / len(rtf_list)
    print("\n" + "=" * 60)
    print(f"AVERAGE REAL-TIME FACTOR (RTF): {avg_rtf:.4f}")
    print("EVALUATION PASSED: All audio synthesis tests completed successfully.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    evaluate_model()
