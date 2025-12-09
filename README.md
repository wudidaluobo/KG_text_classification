<<<<<<< HEAD
# Knowledge Graph Enhanced Text Classification (AG News)

This project implements a text classification pipeline using **DistilBERT**, enhanced with external knowledge from a Knowledge Graph (KG). It links entities in the text to Wikidata using `spacy-entity-linker` and appends their descriptions to the input to improve model context.

## 🚀 Features

* **KG Enhancement:** Automatically detects entities in news text and appends descriptions from Wikidata (via Spacy).
* **Model:** Uses `distilbert-base-uncased` for efficient and accurate classification.
* **Hardware Acceleration:** Supports **CUDA** (NVIDIA), **MPS** (Apple Silicon M1/M2/M3), and CPU.
* **Smart Caching:** Saves processed/enhanced datasets to CSV to avoid re-running expensive KG lookups.
* **TensorBoard Support:** Logs training loss and test accuracy for visualization.

## 📂 Project Structure

* `train_model.py`: Main entry point. Handles data loading, training loop, validation, and saving checkpoints.
* `kg_enhancer.py`: Contains the `KGEnhancer` class which uses Spacy to link entities and retrieve knowledge.
* `dataset.py`: Custom PyTorch `Dataset` class (`KGNewsDataset`) for tokenization and data handling.
* `config.py`: Central configuration for hyperparameters (Batch size, LR, Epochs) and device selection.

## 🛠️ Installation

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Spacy Model**
    Required for basic NLP tasks.
    ```bash
    python -m spacy download en_core_web_sm
    ```

3.  **Download Knowledge Base**
    Required for `spacy-entity-linker` (downloads SQLite KB file).
    ```bash
    python -m spacy_entity_linker "download_knowledge_base"
    ```

## ⚙️ Configuration

You can modify training parameters in `config.py`:

* `SAMPLE_SIZE`: Set to `1000` for debugging or `None` to train on the full dataset.
* `BATCH_SIZE`: Default is `32`.
* `EPOCHS`: Default is `20`.
* `DEVICE`: Automatically detects CUDA, MPS, or CPU.

## ▶️ Usage

**Start Training:**
```bash
python train_model.py

