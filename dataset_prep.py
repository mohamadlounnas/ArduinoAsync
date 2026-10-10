import os
import re
import math
import struct
import wave
import csv

# Directory setup
DATASET_DIR = "dataset"
AUDIO_DIR = os.path.join(DATASET_DIR, "wavs")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")

os.makedirs(AUDIO_DIR, exist_ok=True)

# Algerian Darja Arabizi to Arabic Phonetic Mapping
ARABIZI_MAP = {
    '3': 'ع',
    '7': 'ح',
    '9': 'ق',
    '5': 'خ',
    '2': 'ء',
    'ch': 'ش',
    'ou': 'و',
    'kh': 'خ',
    'gh': 'غ',
    'dh': 'ظ',
    'th': 'ث',
    'sh': 'ش',
}

def normalize_darja_text(text: str) -> str:
    """
    Normalize Algerian Darja text (both Arabic script and Arabizi).
    """
    text = text.lower().strip()

    # Replace Arabizi combination patterns
    for pattern, replacement in ARABIZI_MAP.items():
        text = text.replace(pattern, replacement)

    # Arabic unicode character normalization
    # Remove Arabic diacritics / tashkeel
    tashkeel_re = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel_re, '', text)

    # Normalize Arabic Alef variants (Arabic script only)
    text = re.sub(r'[أإآ]', 'ا', text)
    # Normalize Teh Marbuta
    text = re.sub(r'ة', 'ه', text)
    # Normalize Alef Maksura
    text = re.sub(r'ى', 'ي', text)

    # Remove redundant spaces & non-alphanumeric punctuation except standard characters
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def create_synthetic_darja_wav(filepath: str, duration_sec: float = 2.0, base_freq: float = 220.0, sample_rate: int = 22050):
    """
    Generates a clean synthetic speech-like formanted harmonic waveform audio file (.wav)
    for model testing and fine-tuning demonstration.
    """
    num_samples = int(sample_rate * duration_sec)
    num_channels = 1
    sample_width = 2  # 16-bit PCM

    wav_file = wave.open(filepath, 'w')
    wav_file.setnchannels(num_channels)
    wav_file.setsampwidth(sample_width)
    wav_file.setframerate(sample_rate)

    frames = bytearray()
    for i in range(num_samples):
        t = float(i) / sample_rate
        # Formant frequencies typical of spoken speech
        f1 = base_freq + 30.0 * math.sin(2 * math.pi * 3.0 * t)
        f2 = base_freq * 2.5 + 50.0 * math.cos(2 * math.pi * 5.0 * t)

        # Envelope decay / cadence modulation
        envelope = math.sin(math.pi * (i / num_samples)) ** 0.5

        # Combined harmonic acoustic signal
        value = (0.6 * math.sin(2 * math.pi * f1 * t) + 0.4 * math.sin(2 * math.pi * f2 * t)) * envelope

        # Convert float to 16-bit signed int
        sample_int = int(max(min(value * 32767, 32767), -32768))
        frames.extend(struct.pack('<h', sample_int))

    wav_file.writeframes(frames)
    wav_file.close()


# Core Algerian Darja dataset entries (mix of daily phrases in Arabic script & Arabizi)
DARJA_DATASET_ENTRIES = [
    ("darja_001", "واش راك لاباس الحمد لله", "wach rak labas alhamdulillah"),
    ("darja_002", "صباح الخير خوي كي راك داير", "sabah el kher khouya ki rak dayer"),
    ("darja_003", "صحا عيدكم وكل عام ونتوما بخير", "saha eidkom w kol am w ntoma b kher"),
    ("darja_004", "اراك فاهم خويا ولا نزيد نفهمك", "arak fahem khoya wala nzid nfahmak"),
    ("darja_005", "الجو شباب اليوم في الجزائر العاصمة", "el jaw chbab el youm fi dzayer el asima"),
    ("darja_006", "رانا هنا باش نتعلمو ونطورو", "rana hna bach ntealmo w ntowro"),
    ("darja_007", "كاين بزاف حوايج شابين في البلاد", "kayen bezaf hwayej chbab fi el blad"),
    ("darja_008", "شحال تسوى هذه الكسرة خويا", "chhal teswa hadi el kesra khoya"),
    ("darja_009", "نلتقيو العشية ان شاء الله", "nelteqiw el achiya inchaallah"),
    ("darja_010", "طمني عليك خويا لعزيز", "tamni alik khoya el aziz"),
]


def prepare_dataset():
    print("Preparing Algerian Darja Dataset...")
    metadata_rows = []

    for idx, (file_id, arabic_text, arabizi_text) in enumerate(DARJA_DATASET_ENTRIES):
        wav_filename = f"{file_id}.wav"
        wav_path = os.path.join(AUDIO_DIR, wav_filename)

        # Vary frequency slightly for distinct audio signatures
        freq = 180.0 + (idx * 15.0)
        duration = 1.5 + (idx % 3) * 0.5
        create_synthetic_darja_wav(wav_path, duration_sec=duration, base_freq=freq)

        normalized_arabic = normalize_darja_text(arabic_text)
        normalized_arabizi = normalize_darja_text(arabizi_text)

        metadata_rows.append([wav_filename, arabic_text, normalized_arabic, arabizi_text, normalized_arabizi])

    # Write metadata CSV
    with open(METADATA_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(["id", "raw_arabic", "normalized_arabic", "raw_arabizi", "normalized_arabizi"])
        writer.writerows(metadata_rows)

    print(f"Dataset prepared successfully! {len(metadata_rows)} audio samples and metadata written.")


if __name__ == "__main__":
    prepare_dataset()
