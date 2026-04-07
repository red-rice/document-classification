## Clone the repository

   ```bash
git clone https://github.com/forhadreza43/document-classification-multimodal.git
   ```

## 📁 Project Structure

```
└── 📁Project
    └── 📁checkpoints
        ├── bert_margin_star.pt
        ├── layoutlmv3_margin_star.pt
    └── 📁rvl-cdip-box
        └── 📁imagesa
        └── .........
        └── 📁imagesz
    └── 📁rvl-cdip-o-box
        ├── documentcloud_4_ood__pdf3_documentcloud_00015_ood.tsv
        ├── .....................................................
        ├── websearch_9__pdf3_websearch_9_00008_ood.tsv
    └── 📁rvl-cdip-text
        └── 📁imagesa
        └── .........
        └── 📁imagesz
        ├── text_test.txt
        ├── text_train.txt
        ├── text_val.txt
    └── 📁rvl-cdip
        └── 📁imagesa
        └── .........
        └── 📁imagesz   
        └── 📁labels
            ├── test.txt
            ├── train.txt
            ├── val.txt
        ├── readme.txt
    └── 📁rvl-cdip-o
        ├── documentcloud_4_ood__pdf3_documentcloud_00015_ood.tif
        ├── .....................................................
        ├── websearch_9__pdf3_websearch_9_00008_ood.tif
    └── 📁rvl-cdip-o-text
        ├── documentcloud_4_ood__pdf3_documentcloud_00015_ood.txt
        ├── .....................................................
        ├── websearch_9__pdf3_websearch_9_00008_ood.txt
    └── 📁src
        ├── classification_result_multimodal.py
        ├── classification_result.py
        ├── config.py
        ├── data_multimodal_ood.py
        ├── data_multimodal.py
        ├── data.py
        ├── evaluate_multimodal.py
        ├── generate_ocr_boxes_tesseract_ood.py
        ├── generate_ocr_boxes_tesseract.py
        ├── generate_ocr_boxes_train10k_val_test.py
        ├── generate_text_from_image.py
        ├── knn_ood_multimodal.py
        ├── knn_ood.py
        ├── loss.py
        ├── losses_extra.py
        ├── losses.py
        ├── metrics.py
        ├── model_multimodal.py
        ├── model.py
        ├── novelty_detection_result_99PRE_multimodal.py
        ├── novelty_detection_result_99PRE.py
        ├── novelty_detection_result_multimodal.py
        ├── novelty_detection_result_old.py
        ├── novelty_detection_result.py
        ├── ocr_missing_text.py
        ├── rejection_effectiveness_multimodal.py
        ├── rejection_effectiveness.py
        ├── sampler.py
        ├── test.py
        ├── train_multimodal.py
        ├── train.py
    ├── .gitignore
    ├── check_norms.py
    ├── check_shapes.py
    ├── command.txt
    ├── Deep-metric-learning-for-end-to-end-document-classificati_2025_Neurocomputin.pdf
    ├── diagnose_mm.py
    ├── dual.txt
    ├── inspect_ckpts.py
    ├── inspect_mm_data.py
    ├── instruction.txt
    ├── main.py
    ├── ocr_quality.py
    ├── ocr_stats.py
    ├── README.md
    ├── test_bert_on_box.py
    ├── test_mm_no_image.py
    ├── test.txt
    ├── text-paper.txt
    ├── train.txt
    └── val.txt
```

Add all the datasets along with project structure.

## Install dependencies and packages

**CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && pip install transformers numpy scikit-learn tqdm Pillow pytesseract faiss-cpu
```

**GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip install transformers numpy scikit-learn tqdm Pillow pytesseract faiss-gpu
```

|               Package                 |                Purpose                |
|---------------------------------------|---------------------------------------|
| `torch`, `torchvision`, `torchaudio`  | PyTorch deep learning framework       |
| `transformers`                        | Hugging Face BERT / LayoutLMv3 models |
| `numpy`                               | Numerical arrays                      |
| `scikit-learn`                        | KNN, metrics, train/test splits       |
| `tqdm`                                | Progress bars                         |
| `Pillow`                              | Image loading (PIL)                   |
| `pytesseract`                         | Tesseract OCR Python wrapper          |
| `faiss-cpu` / `faiss-gpu`             | Fast similarity search for KNN*       |


> **Note:** `pytesseract` requires the Tesseract binary installed separately.
> Windows: download from https://github.com/UB-Mannheim/tesseract/wiki and install to `C:\Program Files\Tesseract-OCR\`.

# A. One-time preparation

## 1) Generate missing OCR text for RVL-CDIP

Run once if some `.txt` files are missing in `rvl-cdip-text`.

```bash
python src/generate_text_from_image.py --project_root "F:\Project" --tesseract_cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## 2) Generate OCR boxes for RVL-CDIP

```bash
python src/generate_ocr_boxes_tesseract.py --project_root "F:\Project" --missing_only --tesseract_cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## 3) Generate OCR boxes for OOD

```bash
python src/generate_ocr_boxes_tesseract_ood.py --project_root "F:\Project" --missing_only --tesseract_cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```
---

# B. Text-only

## 1) Train

```bash
python src/train.py --project_root "F:\Project" --save_dir checkpoints --loss margin_star --use cpu --train_samples 5000 --val_samples 1000
python src/train.py --project_root "F:\Project" --save_dir checkpoints --loss margin_star --use gpu --train_samples 50000 --val_samples 6250
python src/train.py --project_root "F:\Project" --save_dir checkpoints --loss margin_star --use gpu --train_samples 100000 --val_samples 12500
python src/train.py --project_root "F:\Project" --save_dir checkpoints --loss margin_star --use gpu --train_samples full --val_samples full

```

## 2) Closed-set classification

```bash
python src/classification_result.py --project_root "F:\Project" --ckpt checkpoints/bert_margin_star.pt --use cpu --test_samples 1000
python src/classification_result.py --project_root "F:\Project" --ckpt checkpoints/bert_margin_star.pt --use gpu --test_samples 6250
python src/classification_result.py --project_root "F:\Project" --ckpt checkpoints/bert_margin_star.pt --use gpu --test_samples 12500
python src/classification_result.py --project_root "F:\Project" --ckpt checkpoints/bert_margin_star.pt --use gpu --test_samples full

```

## 3) Novelty detection

```bash
python src/novelty_detection_result.py --project_root "F:\Project" --ckpt_dir checkpoints --use cpu --train_samples 5000 --val_samples 1000 --test_samples 1000 --tpr 0.95
python src/novelty_detection_result.py --project_root "F:\Project" --ckpt_dir checkpoints --use gpu --train_samples 50000 --val_samples 6250 --test_samples 6250 --tpr 0.95
python src/novelty_detection_result.py --project_root "F:\Project" --ckpt_dir checkpoints --use gpu --train_samples 100000 --val_samples 12500 --test_samples 12500 --tpr 0.95
python src/novelty_detection_result.py --project_root "F:\Project" --ckpt_dir checkpoints --use gpu --train_samples full --val_samples full --test_samples full --tpr 0.95

```

## 4) Precision-threshold

```bash
python src/novelty_detection_result_99PRE.py --project_root "F:\Project" --ckpt checkpoints/bert_margin_star.pt --use cpu --train_samples 5000 --val_samples 1000 --test_samples 1000 --target_pre 99.0
python src/novelty_detection_result_99PRE.py --project_root "F:\Project" --ckpt checkpoints/bert_margin_star.pt --use gpu --train_samples 50000 --val_samples 6250 --test_samples 6250 --target_pre 99.0
python src/novelty_detection_result_99PRE.py --project_root "F:\Project" --ckpt checkpoints/bert_margin_star.pt --use gpu --train_samples 100000 --val_samples 12500 --test_samples 12500 --target_pre 99.0
python src/novelty_detection_result_99PRE.py --project_root "F:\Project" --ckpt checkpoints/bert_margin_star.pt --use gpu --train_samples full --val_samples full --test_samples full --target_pre 99.0

```

## 5) KNN* rejection effectiveness

```bash
python src/rejection_effectiveness.py --project_root "F:\Project" --ckpt checkpoints/bert_margin_star.pt --use cpu --train_samples 5000 --val_samples 1000 --test_samples 1000 --target_pre 99.0
python src/rejection_effectiveness.py --project_root "F:\Project" --ckpt checkpoints/bert_margin_star.pt --use gpu --train_samples 50000 --val_samples 6250 --test_samples 6250 --target_pre 99.0
python src/rejection_effectiveness.py --project_root "F:\Project" --ckpt checkpoints/bert_margin_star.pt --use gpu --train_samples 100000 --val_samples 12500 --test_samples 12500 --target_pre 99.0
python src/rejection_effectiveness.py --project_root "F:\Project" --ckpt checkpoints/bert_margin_star.pt --use gpu --train_samples full --val_samples full --test_samples full --target_pre 99.0

```

# C. Multimodal

## 1) Train

```bash
python src/train_multimodal.py --project_root "F:\Project" --save_dir checkpoints --loss margin_star --use cpu --train_samples 5000 --val_samples 1000
python src/train_multimodal.py --project_root "F:\Project" --save_dir checkpoints --loss margin_star --use gpu --train_samples 50000 --val_samples 6250
python src/train_multimodal.py --project_root "F:\Project" --save_dir checkpoints --loss margin_star --use gpu --train_samples 100000 --val_samples 12500
python src/train_multimodal.py --project_root "F:\Project" --save_dir checkpoints --loss margin_star --use gpu --train_samples full --val_samples full

```

## 2) Closed-set classification

```bash
python src/classification_result_multimodal.py --project_root "F:\Project" --ckpt checkpoints/layoutlmv3_margin_star.pt --use cpu --test_samples 1000
python src/classification_result_multimodal.py --project_root "F:\Project" --ckpt checkpoints/layoutlmv3_margin_star.pt --use gpu --test_samples 6250
python src/classification_result_multimodal.py --project_root "F:\Project" --ckpt checkpoints/layoutlmv3_margin_star.pt --use gpu --test_samples 12500
python src/classification_result_multimodal.py --project_root "F:\Project" --ckpt checkpoints/layoutlmv3_margin_star.pt --use gpu --test_samples full

```

## 3) Novelty detection

```bash
python src/novelty_detection_result_multimodal.py --project_root "F:\Project" --ckpt_dir checkpoints --use cpu --train_samples 5000 --val_samples 1000 --test_samples 1000 --tpr 0.95
python src/novelty_detection_result_multimodal.py --project_root "F:\Project" --ckpt_dir checkpoints --use gpu --train_samples 50000 --val_samples 6250 --test_samples 6250 --tpr 0.95
python src/novelty_detection_result_multimodal.py --project_root "F:\Project" --ckpt_dir checkpoints --use gpu --train_samples 100000 --val_samples 12500 --test_samples 12500 --tpr 0.95
python src/novelty_detection_result_multimodal.py --project_root "F:\Project" --ckpt_dir checkpoints --use gpu --train_samples full --val_samples full --test_samples full --tpr 0.95

```

## 4) Precision-threshold

```bash
python src/novelty_detection_result_99PRE_multimodal.py --project_root "F:\Project" --ckpt checkpoints/layoutlmv3_margin_star.pt --use cpu --train_samples 5000 --val_samples 1000 --test_samples 1000 --target_pre 99.0
python src/novelty_detection_result_99PRE_multimodal.py --project_root "F:\Project" --ckpt checkpoints/layoutlmv3_margin_star.pt --use gpu --train_samples 50000 --val_samples 6250 --test_samples 6250 --target_pre 99.0
python src/novelty_detection_result_99PRE_multimodal.py --project_root "F:\Project" --ckpt checkpoints/layoutlmv3_margin_star.pt --use gpu --train_samples 100000 --val_samples 12500 --test_samples 12500 --target_pre 99.0
python src/novelty_detection_result_99PRE_multimodal.py --project_root "F:\Project" --ckpt checkpoints/layoutlmv3_margin_star.pt --use gpu --train_samples full --val_samples full --test_samples full --target_pre 99.0

```

## 5) KNN* rejection effectiveness

```bash
python src/rejection_effectiveness_multimodal.py --project_root "F:\Project" --ckpt checkpoints/layoutlmv3_margin_star.pt --use cpu --train_samples 5000 --val_samples 1000 --test_samples 1000 --target_pre 99.0
python src/rejection_effectiveness_multimodal.py --project_root "F:\Project" --ckpt checkpoints/layoutlmv3_margin_star.pt --use gpu --train_samples 50000 --val_samples 6250 --test_samples 6250 --target_pre 99.0
python src/rejection_effectiveness_multimodal.py --project_root "F:\Project" --ckpt checkpoints/layoutlmv3_margin_star.pt --use gpu --train_samples 100000 --val_samples 12500 --test_samples 12500 --target_pre 99.0
python src/rejection_effectiveness_multimodal.py --project_root "F:\Project" --ckpt checkpoints/layoutlmv3_margin_star.pt --use gpu --train_samples full --val_samples full --test_samples full --target_pre 99.0

```
