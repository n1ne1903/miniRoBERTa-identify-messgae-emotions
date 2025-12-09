from datasets import load_dataset
import re, string, os
from pathlib import Path
from bs4 import BeautifulSoup
import py_vncorenlp
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

# ===== Mapping 7→5 lớp =====
MAP5 = {
    "Enjoyment": "joy",
    "Sadness": "sadness",
    "Anger": "anger",
    "Disgust": "anger",
    "Fear": "fear",
    "Surprise": "surprise",
    "Other": None
}
CLASSES = ["joy", "sadness", "anger", "fear", "surprise"]
label2id = {c: i for i, c in enumerate(CLASSES)}

def _clean_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = BeautifulSoup(s, "html.parser").get_text()
    s = s.lower()
    s = re.sub(r'\d+', '', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\W+', ' ', s)
    return s.strip()

def _ensure_vncorenlp(save_dir) -> py_vncorenlp.VnCoreNLP:
    # ép về str để tránh lỗi Path trên một số phiên bản
    save_dir = str(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    jar = os.path.join(save_dir, "VnCoreNLP-1.1.1.jar")
    if not os.path.exists(jar):
        py_vncorenlp.download_model(save_dir=save_dir)
    return py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)

class ClsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        assert len(texts) == len(labels), "texts và labels phải cùng độ dài"
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text if isinstance(text, str) else str(text),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class UITVSMECForClassification(Dataset):
    """
    - Tự phát hiện 'Sentence' (str) hoặc 'Sentences' (list[str]).
    - Nếu 'Sentences' → join.
    - Clean theo batch.
    - Word segmentation: bật/tắt bằng use_wseg, hoặc truyền segmenter đã có.
    """
    def __init__(self,
                 split: str = "train",
                 save_dir = "/content/drive/MyDrive/vncorenlp",
                 max_len: int = 128,
                 use_wseg: bool = True,
                 segmenter=None):
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.use_wseg = bool(use_wseg)
        self._external_segmenter = segmenter  # dùng lại nếu truyền vào
        save_dir = Path(save_dir)  # vẫn chấp nhận Path ở ngoài

        # 1) Load dataset split
        ds = load_dataset("tridm/UIT-VSMEC")[split]

        # 2) Map nhãn về 5 lớp
        def to_5_labels(ex):
            emo = ex.get("Emotion")
            mapped = MAP5.get(emo, None)
            ex["labels"] = -1 if mapped is None else label2id[mapped]
            return ex
        ds = ds.map(to_5_labels)
        ds = ds.filter(lambda ex: ex["labels"] != -1)

        # 3) Chọn cột văn bản
        cols = ds.column_names
        text_col = "Sentence" if "Sentence" in cols else ("Sentences" if "Sentences" in cols else None)
        if text_col is None:
            raise KeyError(f"Không tìm thấy cột văn bản trong {cols}")

        # 3.1) Chuẩn hoá về 'Sentence' (str)
        def normalize_batch(batch):
            src = batch[text_col]
            out = []
            for item in src:
                if isinstance(item, list):
                    out.append(" ".join([x for x in item if x is not None]))
                else:
                    out.append("" if item is None else str(item))
            return {"Sentence": out}
        ds = ds.map(normalize_batch, batched=True, desc="Normalize text")

        # 4) Clean theo batch
        def clean_batch(batch):
            texts = batch["Sentence"]
            return {"Sentence": [_clean_text(t) for t in texts]}
        ds = ds.map(clean_batch, batched=True, desc="Clean text")

        # 5) Word segmentation (optional, robust)
        if self.use_wseg:
            seg = self._external_segmenter
            if seg is None:
                try:
                    seg = _ensure_vncorenlp(save_dir)
                except Exception as e:
                    print(f"[WARN] Không khởi tạo được VnCoreNLP ({e}). Fallback: tắt use_wseg.")
                    self.use_wseg = False
            if self.use_wseg and seg is not None:
                def wseg_batch(batch):
                    texts = batch["Sentence"]
                    try:
                        segged = seg.word_segment(texts)  # list input
                    except Exception:
                        segged = [seg.word_segment(t) for t in texts]
                    norm = []
                    for x in segged:
                        # một số bản trả list token → join
                        if isinstance(x, list):
                            norm.append(" ".join(x))
                        else:
                            norm.append(x)
                    return {"Sentence": norm}
                ds = ds.map(wseg_batch, batched=True, batch_size=64, desc="Word segmentation")

        # 6) Gói vào torch Dataset
        self.dataset = ClsDataset(
            texts=ds["Sentence"],
            labels=ds["labels"],
            tokenizer=self.tokenizer,
            max_len=max_len
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
