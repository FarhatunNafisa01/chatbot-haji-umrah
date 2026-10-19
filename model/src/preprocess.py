# preprocess.py
# Membersihkan dataset dan membagi menjadi train/val/test (80/10/10, stratified)

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Path (sesuaikan jika dijalankan dari folder lain)
DATA_DIR = Path("data")
SUMBER = DATA_DIR / "dataset.csv"
KELUARAN = DATA_DIR / "processed"
SEED = 42


def bersihkan(teks: str) -> str:
    """Rapikan spasi berlebih. Huruf besar/kecil dibiarkan karena
    tokenizer IndoBERT sudah melakukan lowercase sendiri."""
    teks = str(teks).strip()
    return re.sub(r"\s+", " ", teks)


def main():
    KELUARAN.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMBER)
    df["text"] = df["text"].apply(bersihkan)
    df = df.dropna(subset=["text", "intent"]).drop_duplicates("text")
    print(f"Total data: {len(df)} baris, {df['intent'].nunique()} intent")

    # Buat pemetaan intent <-> id (diurutkan agar konsisten)
    daftar_intent = sorted(df["intent"].unique())
    label2id = {nama: i for i, nama in enumerate(daftar_intent)}
    id2label = {i: nama for nama, i in label2id.items()}
    df["label"] = df["intent"].map(label2id)

    # Split 80% train, 20% sisanya -> dibagi dua jadi val dan test
    train, temp = train_test_split(
        df, test_size=0.20, stratify=df["label"], random_state=SEED
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=SEED
    )

    train.to_csv(KELUARAN / "train.csv", index=False)
    val.to_csv(KELUARAN / "val.csv", index=False)
    test.to_csv(KELUARAN / "test.csv", index=False)

    with open(KELUARAN / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f,
                  ensure_ascii=False, indent=2)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")


if __name__ == "__main__":
    main()