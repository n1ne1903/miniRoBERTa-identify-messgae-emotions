from pathlib import Path
import torch

# Xác định số GPU và thiết bị
N_GPUS = torch.cuda.device_count()
if N_GPUS > 0:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

N_WORKERS = 4
CKPT_DIR = Path(__file__).parent/"pretrain/checkpoints"
N_CKPT_SAMPLES = 100_000

# Các tham số cho model và training
MAX_LEN = 128
VOCAB_SIZE = 64001  # PhoBERT-base vocab size
N_LAYERS = 12
N_HEADS = 12
HIDDEN_SIZE = 768
MLP_SIZE = 3072
MAX_LR = 2e-5
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
WEIGHT_DECAY = 0.01
N_WARM_STEPS = 1000

# Tham số cho fine-tuning
NUM_LABELS = 5
CLASSIFIER_DROPOUT = 0.1

# Thư mục cho VnCoreNLP
SAVE_DIR = "/content/drive/MyDrive/sentiment/vncorenlp"  # Thay bằng path thực