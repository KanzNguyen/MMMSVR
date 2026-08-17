# Multimodal Video Retrieval System

**Published at SOICT 2024** — 📄 [MMMSVR: An Advanced Video Retrieval and Question Answering System](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_11) *(Springer, CCIS)*

---

## Overview

This repository contains code for a multimodal video retrieval system. The pipeline supports querying large video archives using **text, image, or audio** inputs by integrating semantic search (CLIP, BEiT-3, BLIP-2), object-level search (YOLOv10 + CLIP), OCR-based search, and ASR-based audio search. Results are fused using **Reciprocal Rank Fusion (RRF)** with dynamic temporal weighting.

> **Scope:** This repository covers the **retrieval** component of the MMMSVR system. The question-answering module described in the paper is not included here.

---

## Repository Structure
```text
Preprocessing & Segmentation
    Uniform_video_segmentation.ipynb     # Segment video into fixed-length clips
    Video_segmentation.ipynb             # Segment video by news story boundaries

Feature Extraction
    BEiT3_feature_extraction.ipynb       # Extract frame features using BEiT-3
    BLIP2_feature_extraction.ipynb       # Extract frame features using BLIP-2
    CLIP_feature_extraction_for_segmented_objects.ipynb  # Extract CLIP features for detected objects
    Object_extraction.ipynb              # Detect and crop objects using YOLOv10
    Object_feature_extraction.ipynb      # Extract CLIP features from cropped objects
    Audio_extraction.ipynb               # Extract audio from video
    Audio_feature_extraction.ipynb       # Extract text features
    OCR_extraction.ipynb                 # Extract on-screen text via OCR

Indexing
    BEiT3_FAISS_indexing.ipynb           # Build FAISS index from BEiT-3 features
    BLIP2_FAISS_indexing.ipynb           # Build FAISS index from BLIP-2 features
    CLIP_FAISS_indexing.ipynb            # Build FAISS index from CLIP-ViT-H14 features
    Object_indexing.ipynb                # Build FAISS index for object-level features
    Index_audio_for_elasticsearch.ipynb  # Index ASR transcripts in Elasticsearch
    Index_ocr_for_elasticsearch.ipynb    # Index OCR transcripts in Elasticsearch
    Audio_elastic_search.ipynb           # Text-based audio search via Elasticsearch
    OCR_elastic_search.ipynb             # Text-based OCR search via Elasticsearch

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

## Results

Retrieval quality on **35 evaluation queries**, comparing each single-modality index against the
**RRF fusion** of all three. Fusion outperforms every individual model on both metrics.

| Method | MRR ↑ | AP@20 ↑ |
|---|---|---|
| CLIP-ViT-H14 | 0.559 | 0.109 |
| BLIP-2 | 0.381 | 0.083 |
| BEiT-3 | 0.645 | 0.104 |
| **RRF fusion (CLIP + BLIP-2 + BEiT-3)** | **0.699** | **0.130** |

> Evaluation uses a **subset of queries** from the AI Challenge HCMC preliminary round. Ground-truth
> relevance was assigned by the team based on the correct answers known after the competition; the
> queries are **not redistributed here** due to the competition's data terms, and the official
> challenge evaluation closed after the competition. Metrics are reported for reference only.

## Acknowledgements

- [CLIP (OpenAI)](https://github.com/openai/CLIP)
- [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3)
- [BLIP-2 (Salesforce LAVIS)](https://github.com/salesforce/LAVIS)
- [YOLOv10](https://github.com/THU-MIG/yolov10)
- [Wav2Vec 2.0](https://huggingface.co/facebook/wav2vec2-base-960h)
