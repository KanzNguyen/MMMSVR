# Multimodal Video Retrieval System

**Published at SOICT 2024** (International Symposium on Information and Communication Technology)

---

## Overview

This repository contains code for a multimodal video retrieval system. The pipeline supports querying large video archives using **text, image, or audio** inputs by integrating semantic search (CLIP, BEiT-3, BLIP-2), object-level search (YOLOv10 + CLIP), OCR-based search, and ASR-based audio search. Results are fused using **Reciprocal Rank Fusion (RRF)** with dynamic temporal weighting.

---

## Repository Structure
```text
Preprocessing & Segmentation
    Uniform video segmentation.ipynb     # Segment video into fixed-length clips
    Video segmentation.ipynb             # Segment video by news story boundaries

Feature Extraction
    BEiT3 feature extraction.ipynb       # Extract frame features using BEiT-3
    BLIP2 feature extraction.ipynb       # Extract frame features using BLIP-2
    CLIP feature extraction for segmented objects.ipynb  # Extract CLIP features for detected objects
    Object extraction.ipynb              # Detect and crop objects using YOLOv10
    Object feature extraction.ipynb      # Extract CLIP features from cropped objects
    Audio extraction.ipynb               # Extract audio from video
    Audio feature extraction.ipynb       # Extract text features
    OCR extraction.ipynb                 # Extract on-screen text via OCR

Indexing
    BEiT3 FAISS indexing.ipynb           # Build FAISS index from BEiT-3 features
    BLIP2 FAISS Indexing.ipynb           # Build FAISS index from BLIP-2 features
    CLIP FAISS indexing.ipynb            # Build FAISS index from CLIP-ViT-H14 features
    Object indexing.ipynb                # Build FAISS index for object-level features
    Index audio for elasticsearch.ipynb  # Index ASR transcripts in Elasticsearch
    Index ocr for elasticsearch.ipynb    # Index OCR transcripts in Elasticsearch
    Audio elastic search.ipynb           # Text-based audio search via Elasticsearch
    OCR elastic search.ipynb             # Text-based OCR search via Elasticsearch

Search
    main.ipynb                           # Unified search: all modalities + RRF fusion
```

---

## Notebooks

All notebooks are designed to run on **Kaggle** (free GPU).

1. **Segmentation** — segment videos into clips
2. **Feature Extraction** — extract features from each modality
3. **Indexing** — build FAISS / Elasticsearch indexes
4. **Search** — run queries via `main.ipynb`

---

## Acknowledgements

- [CLIP (OpenAI)](https://github.com/openai/CLIP)
- [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3)
- [BLIP-2 (Salesforce LAVIS)](https://github.com/salesforce/LAVIS)
- [YOLOv10](https://github.com/THU-MIG/yolov10)
- [Wav2Vec 2.0](https://huggingface.co/facebook/wav2vec2-base-960h)
