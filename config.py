import torch

class Config:
    def __init__(self):
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
            print("🚀 Using Device: NVIDIA GPU (CUDA)")
        elif torch.backends.mps.is_available():
            self.DEVICE = torch.device("mps")
            print("🍎 Using Device: Apple Silicon GPU (MPS)")
        else:
            self.DEVICE = torch.device("cpu")
            print("🐢 Using Device: CPU (Slow)")

        # 2. 训练参数
        self.MODEL_NAME = "distilbert-base-uncased"  # 使用 DistilBERT
        self.MAX_LEN = 128
        self.BATCH_SIZE = 32  # DistilBERT 显存占用小，Batch Size 可以开大一点
        self.EPOCHS = 20
        self.LEARNING_RATE = 2e-5

        # 3. 调试开关
        # 【重要】初次运行建议设为 1000 测试流程。
        # 确认无误后改为 None 以跑全量数据 (12万条)。
        self.SAMPLE_SIZE = None
        self.CACHE_FILE = "files/agnews_test_enhanced.csv"
        self.SAVE_CHECKPOINT_INTERVAL = 10
        self.SAVE_CHECKPOINT_DIR = "/Users/buhaozhe/KG-Project/checkpoints"