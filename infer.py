import os
import math
import wave
import struct
import argparse
import time
import torch

from model import DarjaTTSModel, text_to_sequence
from dataset_prep import normalize_darja_text

def mel_to_audio(mel_spec, sample_rate=22050, n_fft=1024, hop_length=256):
    """
    Lightweight vocoder synthesizing continuous audio waveform from predicted log mel-spectrogram.
    Uses Griffin-Lim style phase reconstruction / harmonic synthesis on CPU.
    """
    num_frames, n_mels = mel_spec.shape
    total_samples = num_frames * hop_length
    audio = [0.0] * total_samples

    # Fundamental frequencies mapped across mel bands
    for f in range(num_frames):
        start = f * hop_length
        mel_frame = mel_spec[f]

        # Calculate frame energy
        frame_energy = math.exp(torch.mean(mel_frame).item())

        for k in range(hop_length):
            t = float(start + k) / sample_rate
            # Harmonic synthesis based on mel frame characteristics
            val = 0.0
            for band in range(0, n_mels, 10):
                band_weight = math.exp(mel_frame[band].item()) / (frame_energy + 1e-5)
                freq = 150.0 + band * 25.0
                val += band_weight * math.sin(2.0 * math.pi * freq * t)

            audio_idx = start + k
            if audio_idx < total_samples:
                audio[audio_idx] += val * min(1.0, frame_energy)

    # Normalize audio signal
    max_val = max(abs(x) for x in audio) if audio else 1.0
    if max_val > 0:
        audio = [x / max_val for x in audio]

    return audio, sample_rate


def save_wav(audio, sample_rate, output_path):
    """Saves float audio array into 16-bit PCM wave format."""
    num_samples = len(audio)
    wav_file = wave.open(output_path, 'w')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)

    frames = bytearray()
    for sample in audio:
        sample_int = int(max(min(sample * 32767.0, 32767.0), -32768.0))
        frames.extend(struct.pack('<h', sample_int))

    wav_file.writeframes(frames)
    wav_file.close()


def synthesize(text, checkpoint_path="darja_tts_checkpoint.pt", output_wav="synthesized_darja.wav"):
    start_time = time.time()

    normalized_text = normalize_darja_text(text)
    print(f"Original Text:   '{text}'")
    print(f"Normalized Text: '{normalized_text}'")

    model = DarjaTTSModel()
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from '{checkpoint_path}'")
    else:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found, using untrained model initialization.")

    model.eval()

    seq = text_to_sequence(normalized_text).unsqueeze(0)

    with torch.no_grad():
        mel_pred, dur_pred = model(seq)

    mel_spec = mel_pred.squeeze(0) # (mel_len, 80)

    audio, sample_rate = mel_to_audio(mel_spec)
    save_wav(audio, sample_rate, output_wav)

    elapsed_time = time.time() - start_time
    audio_duration = len(audio) / float(sample_rate)
    rtf = elapsed_time / audio_duration if audio_duration > 0 else 0.0

    print(f"\nSpeech synthesis completed successfully!")
    print(f"Output File:    {output_wav}")
    print(f"Audio Duration: {audio_duration:.2f} seconds")
    print(f"Inference Time: {elapsed_time:.3f} seconds")
    print(f"Real-Time Factor (RTF): {rtf:.4f} (lower is faster)")

    return output_wav, rtf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize Algerian Darja speech from input text")
    parser.add_argument("--text", type=str, default="واش راك لاباس الحمد لله", help="Text in Algerian Darja (Arabic or Arabizi)")
    parser.add_argument("--checkpoint", type=str, default="darja_tts_checkpoint.pt", help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="test_output.wav", help="Output .wav filepath")
    args = parser.parse_args()

    synthesize(args.text, checkpoint_path=args.checkpoint, output_wav=args.output)
