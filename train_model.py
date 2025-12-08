import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import Config
from kg_enhancer import KGEnhancer
from dataset import KGNewsDataset

# 设置回退机制，防止某些不兼容 MPS 的算子导致报错
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
config = Config()


def load_and_process_data(split_name, cache_filename):
    """
    通用数据加载与处理函数
    :param split_name: huggingface dataset 的键名，例如 'train' 或 'test'
    :param cache_filename: 本地保存的 csv 文件名
    """
    # 1. 检查是否有本地缓存
    if os.path.exists(cache_filename):
        print(f"✅ 发现缓存文件: {cache_filename}，直接读取中...")
        return pd.read_csv(cache_filename)

    print(f"📥 正在为 [{split_name}] 集下载/加载 AG News 数据...")
    dataset = load_dataset("ag_news")
    data_split = dataset[split_name]

    # 2. 采样逻辑 (如果 config.SAMPLE_SIZE 有值，则只取部分数据用于调试)
    if config.SAMPLE_SIZE:
        print(f"⚠️ 调试模式: 仅使用 [{split_name}] 的前 {config.SAMPLE_SIZE} 条数据。")
        # 确保不会越界
        select_range = range(min(config.SAMPLE_SIZE, len(data_split)))
        data_split = data_split.select(select_range)

    df = pd.DataFrame(data_split)

    # 3. 知识图谱增强
    enhancer = KGEnhancer()

    print(f"🧠 开始对 [{split_name}] 集进行知识图谱增强 (CPU密集型)...")
    tqdm.pandas(desc=f"KG Processing ({split_name})")
    df['enhanced_text'] = df['text'].progress_apply(enhancer.enhance_text)

    # 4. 保存缓存
    print(f"💾 [{split_name}] 处理完成，保存缓存到 {cache_filename}")
    df.to_csv(cache_filename, index=False)
    return df


def train():
    writer = SummaryWriter(log_dir="runs/agnews_distilbert_kg")

    # 定义文件名
    train_file = "files/agnews_train_enhanced.csv"
    test_file = "files/agnews_test_enhanced.csv"

    print("=" * 40)
    print("正在加载训练集...")
    train_df = load_and_process_data(split_name="train", cache_filename=train_file)

    print("=" * 40)
    print("正在加载测试集...")
    test_df = load_and_process_data(split_name="test", cache_filename=test_file)

    print("=" * 40)
    print(f"📊 最终数据集统计:\n训练集大小: {len(train_df)}\n测试集大小: {len(test_df)}")

    # 初始化 Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)

    # 构建 Dataset (直接使用全量数据，不再进行 sample/drop 切分)
    training_set = KGNewsDataset(train_df, tokenizer, config.MAX_LEN, config)
    # 这里将 Test 集作为验证/测试集
    validation_set = KGNewsDataset(test_df, tokenizer, config.MAX_LEN, config)

    # DataLoader 配置
    train_params = {'batch_size': config.BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    val_params = {'batch_size': config.BATCH_SIZE, 'shuffle': False, 'num_workers': 0}

    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **val_params)

    # --- 2. 模型初始化 ---
    model = DistilBertForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=4)
    model.to(config.DEVICE)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # --- 3. 训练循环 ---
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        print(f"\nTraining Epoch {epoch + 1}/{config.EPOCHS}")

        loop = tqdm(training_loader, leave=True)
        for data in loop:
            ids = data['ids'].to(config.DEVICE, dtype=torch.long)
            mask = data['mask'].to(config.DEVICE, dtype=torch.long)
            targets = data['targets'].to(config.DEVICE, dtype=torch.long)

            outputs = model(ids, attention_mask=mask, labels=targets)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(training_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        writer.add_scalar("Train/Loss", avg_loss, epoch + 1)

        # --- 4. 验证/测试循环 ---
        model.eval()
        val_targets = []
        val_predictions = []

        print("Running Evaluation on Test Set...")
        with torch.no_grad():
            for data in tqdm(validation_loader, desc="Validating"):
                ids = data['ids'].to(config.DEVICE, dtype=torch.long)
                mask = data['mask'].to(config.DEVICE, dtype=torch.long)
                targets = data['targets'].to(config.DEVICE, dtype=torch.long)

                outputs = model(ids, attention_mask=mask)
                _, preds = torch.max(outputs.logits, dim=1)

                val_targets.extend(targets.cpu().numpy())
                val_predictions.extend(preds.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_predictions)
        print(f"🏆 Test Set Accuracy: {val_acc:.4f}")
        writer.add_scalar("Test/Accuracy", val_acc, epoch + 1)


        if (epoch + 1) % config.SAVE_CHECKPOINT_INTERVAL == 0:
            if not os.path.exists(config.SAVE_CHECKPOINT_DIR):
                os.makedirs(config.SAVE_CHECKPOINT_DIR, exist_ok=True)

            save_path = os.path.join(config.SAVE_CHECKPOINT_DIR, f"distil-bert_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Checkpoint saved: {save_path}")


if __name__ == "__main__":
    # 如果你想先单独生成数据，可以取消下面两行的注释
    # load_and_process_data("train", "agnews_train_enhanced.csv")
    # load_and_process_data("test", "agnews_test_enhanced.csv")

    # 启动训练 (训练函数内部会自动调用数据加载)
    train()