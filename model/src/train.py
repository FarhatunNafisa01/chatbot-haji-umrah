# train.py
# Fine-tuning IndoBERT untuk klasifikasi intent chatbot haji & umrah
#
# Di Colab, jalankan dulu:
#   !pip install -q transformers datasets accelerate scikit-learn seaborn

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)

# ---------- Konfigurasi ----------
NAMA_MODEL = "indobenchmark/indobert-base-p1"
DATA_DIR = Path("data/processed")
OUT_DIR = Path("model/saved_model")
HASIL_DIR = Path("model/results")
MAX_LEN = 64          # pertanyaan chatbot pendek, 64 token sudah cukup
SEED = 42
set_seed(SEED)

OUT_DIR.mkdir(parents=True, exist_ok=True)
HASIL_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Muat data & label ----------
with open(DATA_DIR / "label_map.json", encoding="utf-8") as f:
    peta = json.load(f)
label2id = peta["label2id"]
id2label = {int(k): v for k, v in peta["id2label"].items()}

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

# ---------- Tokenisasi ----------
tokenizer = AutoTokenizer.from_pretrained(NAMA_MODEL)


def tokenisasi(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)


def ke_dataset(df):
    ds = Dataset.from_pandas(df[["text", "label"]])
    return ds.map(tokenisasi, batched=True, remove_columns=["text"])


ds_train, ds_val, ds_test = map(ke_dataset, (train_df, val_df, test_df))

# ---------- Model ----------
model = AutoModelForSequenceClassification.from_pretrained(
    NAMA_MODEL,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)


def hitung_metrik(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, pred),
        "f1_macro": f1_score(labels, pred, average="macro"),
    }


# ---------- Pelatihan ----------
args = TrainingArguments(
    output_dir=str(OUT_DIR / "checkpoints"),
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=26,
    eval_strategy="epoch",          # di transformers versi lama: evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    save_total_limit=2,
    logging_steps=10,
    report_to="none",
    fp16=torch.cuda.is_available(),
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=hitung_metrik,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

# ---------- Evaluasi akhir pada data test ----------
keluaran = trainer.predict(ds_test)
pred = np.argmax(keluaran.predictions, axis=-1)
benar = keluaran.label_ids

nama_kelas = [id2label[i] for i in range(len(id2label))]
laporan = classification_report(
    benar, pred, labels=list(range(len(nama_kelas))),
    target_names=nama_kelas, output_dict=True, zero_division=0,
)
print(classification_report(
    benar, pred, labels=list(range(len(nama_kelas))),
    target_names=nama_kelas, zero_division=0,
))

metrik = {
    "accuracy": accuracy_score(benar, pred),
    "f1_macro": f1_score(benar, pred, average="macro"),
    "per_kelas": laporan,
}
with open(HASIL_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrik, f, ensure_ascii=False, indent=2)

# Confusion matrix
cm = confusion_matrix(benar, pred, labels=list(range(len(nama_kelas))))
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=nama_kelas, yticklabels=nama_kelas)
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix - Data Uji")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(HASIL_DIR / "confusion_matrix.png", dpi=150)

# ---------- Simpan model final ----------
trainer.save_model(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))
with open(OUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f,
              ensure_ascii=False, indent=2)

print(f"\nAkurasi test: {metrik['accuracy']:.4f} | F1 macro: {metrik['f1_macro']:.4f}")