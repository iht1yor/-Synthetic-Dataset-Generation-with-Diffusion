# Synthetic Dataset Generation with Diffusion Models for Rare Skin Lesion Classification

**Computer Vision Course – Fall 2025**  
**Midterm Project Proposal**

---

## 1. Title & Team

**Project Title:** Improving Rare Skin Disease Detection Using AI-Generated Images

**Team Name:** MedVision Team

**Team Members:**
- Khojiev Ikhtiyor (Team Lead) – 220343@centralasian.uz
- Ibrohim Shukrullayev – 220010@centralasian.uz
- Fayziev Abdulxamid – 220107@centralasian.uz

**Repository:** https://github.com/medvision-cau/skin-lesion-project

---

## 2. Abstract

Medical datasets have a big problem - rare diseases appear much less than common ones. In skin cancer detection, dangerous diseases like melanoma appear in only 11% of images, while benign cases make up 67%. This causes AI models to perform poorly on rare but important cases.

We will solve this by using Stable Diffusion to create synthetic images of rare skin lesions. We'll fine-tune the model on HAM10000 dataset and generate around 800 new images for each minority class, then train a classifier using both real and synthetic data together.

Our goal is to improve F1-score on rare classes by 8-10% while keeping overall accuracy stable. Deliverables include trained models, evaluation results, and reusable code.

---

## 3. Problem & Motivation

### Why This Matters

Skin cancer is very common and early detection increases survival rates from 27% to 99%. However, building AI systems is difficult because most datasets have way more benign lesions than dangerous ones.

HAM10000 dataset breakdown:
- Benign nevi: 67%
- Melanoma: 11%
- Dermatofibroma: only 1.1%

When we train on this imbalanced data, models become good at recognizing common lesions but terrible at detecting rare diseases that could kill patients.

### Our Approach

Recent diffusion models like Stable Diffusion can generate photorealistic images from text. We'll use this to create high-quality synthetic images of rare skin diseases to balance the training data.

### Goals

1. Generate 800 synthetic images per rare class (melanoma, dermatofibroma, vascular)
2. Achieve FID score below 30 (measures realism)
3. Improve F1-score on minority classes by 8%+
4. Keep performance on common classes stable

---

## 4. Related Work

**Key Papers:**

| Method | Dataset | Result | Limitations |
|--------|---------|--------|-------------|
| GANs (Khosla 2023) | HAM10000 | +6% F1 | Mode collapse, unstable training |
| StyleGAN2 (Bissoto 2020) | ISIC 2018 | Moderate | Complex tuning |
| RandAugment (traditional) | General | +2-3% | Limited diversity |
| Stable Diffusion (Rombach 2022) | LAION-5B | High quality | Not tested on medical images |

**What We'll Do Differently:**
- Use Stable Diffusion (more stable than GANs)
- Apply LoRA for efficient fine-tuning
- Use textual inversion for medical terms
- Only add synthetic data for minority classes

---

## 5. Data & Resources

### Dataset: HAM10000

- Source: Harvard Dataverse (free for research)
- Size: 10,015 dermoscopic images, 7 classes
- Split: 70% train / 15% val / 15% test

**Class Distribution:**
- Melanocytic nevi: 6,705 (67%)
- Melanoma: 1,113 (11%)
- Benign keratosis: 1,099 (11%)
- Basal cell carcinoma: 514 (5%)
- Actinic keratoses: 327 (3%)
- Vascular lesions: 142 (1.4%) ← target
- Dermatofibroma: 115 (1.1%) ← target

### Compute

- **Platform:** Google Colab Pro (A100 GPU)
- **Estimated time:** ~20 GPU hours total
- **Backup:** Kaggle if Colab times out

### Software

- Python 3.10, PyTorch 2.0
- Hugging Face Diffusers (Stable Diffusion)
- timm (classifiers), scikit-learn (metrics)

### Ethics

HAM10000 is public and anonymized. No IRB needed. All synthetic images will be marked "AI-generated" and used for research only.

---

## 6. Method

### Baselines

1. **Real-only:** ResNet-50 on original data
2. **Oversampling:** Duplicate rare class images
3. **RandAugment:** Best traditional augmentation

### Our Approach (3 Steps)

**Step 1: Fine-tune Stable Diffusion**
- Start with pretrained Stable Diffusion v1.5
- Use LoRA (updates only 5M params vs 860M full model)
- Train on minority class images only (~960 images)
- Learn custom tokens: `<melanoma>`, `<dermatofibroma>`
- Prompts: "A dermoscopic image of melanoma with asymmetric borders"

**Step 2: Generate Synthetic Images**
- Generate 800 images per rare class (2,400 total)
- Filter by FID score (remove FID > 50)
- Check diversity with LPIPS metric

**Step 3: Train Classifier**
- Use EfficientNet-B0 (lighter than ResNet-50)
- Common classes: 100% real data
- Rare classes: 50% real + 50% synthetic
- Train 50 epochs with focal loss

### Architecture

```
HAM10000 minority classes 
  → Fine-tune Stable Diffusion with LoRA
  → Generate 2,400 synthetic images
  → Mix with real data (50/50 for rare classes)
  → Train EfficientNet-B0 classifier
  → Evaluate on held-out test set
```

---

## 7. Experiments & Metrics

### Synthetic Image Quality

- **FID Score:** Lower = more realistic (target: <30)
- **Inception Score:** Higher = better (target: >3.0)
- **Visual check:** Rate 50 random images manually

### Classification Performance

- **F1-Score** (main metric): Better for imbalanced data
- **Balanced Accuracy:** Average per-class recall
- **Overall Accuracy:** Make sure we don't hurt common classes
- **Confusion Matrix:** See misclassifications

### Success Criteria

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| Macro F1 | 0.62 | 0.70 | 0.75 |
| Minority F1 | 0.45 | 0.55 | 0.62 |
| Overall Acc | 0.82 | ≥0.82 | 0.85 |
| FID | N/A | <30 | <20 |

Run each experiment 3 times with different seeds, report mean ± std.

---

## 8. Risks & Mitigations

**Problem 1: Low quality synthetic images**
- Mitigation: Use pretrained Stable Diffusion, apply LoRA
- Backup: Switch to simpler augmentation project

**Problem 2: Colab timeouts**
- Mitigation: Save checkpoints every 30 min
- Backup: Use Kaggle (30h/week free)

**Problem 3: Synthetic data hurts performance**
- Mitigation: Start with 25% synthetic, monitor validation
- Backup: Use synthetic only for analysis

**Problem 4: Not enough time**
- Mitigation: Focus on core experiments first
- Built-in 1 week buffer

---

## 9. Timeline & Roles

### Weekly Plan

| Week | Task | Owner | Deadline |
|------|------|-------|----------|
| W1 (Oct 21-27) | Setup + download data | Everyone | Oct 27 |
| W2 (Oct 28-Nov 3) | Baseline training | Ikhtiyor | Nov 3 |
| W3 (Nov 4-10) | Fine-tune Stable Diffusion | Abdulxamid | Nov 10 |
| W4 (Nov 11-17) | Generate synthetic images | Abdulxamid + Ibrohim | Nov 17 |
| W5 (Nov 18-24) | Train hybrid classifier | Ikhtiyor | Nov 24 |
| W6 (Nov 25-Dec 1) | Experiments + comparison | Everyone | Dec 1 |
| W7 (Dec 2-8) | Visualizations | Ibrohim | Dec 8 |
| W8 (Dec 9-15) | Final report + poster | Everyone | Dec 15 |
| W9 (Dec 16-22) | Buffer + presentation | Everyone | Dec 22 |

### Responsibilities

**Ikhtiyor:** Baselines, classifier training, results analysis  
**Abdulxamid:** Diffusion fine-tuning, synthetic generation, code docs  
**Ibrohim:** Visualizations, error analysis, poster design  
**Shared:** Report writing, weekly updates

---

## 10. Expected Outcomes

### Deliverables

1. Code repository with documentation
2. Trained models (diffusion + classifier)
3. Final report (8-10 pages)
4. Poster presentation
5. Demo: generate synthetic images + show results

### Stretch Goals (if time permits)

- Test on ISIC 2019 dataset
- Get medical student feedback
- Release synthetic dataset publicly

---

## 11. Ethics & Compliance

- HAM10000 is public (CC BY-NC-SA 4.0)
- Already anonymized, no IRB needed
- Synthetic images marked as "AI-generated"
- For research only, not medical diagnosis
- Limitation: Dataset mostly light-skinned patients

---

## 12. References

1. Rombach et al. (2022). High-resolution image synthesis with latent diffusion models. CVPR.

2. Tschandl et al. (2018). The HAM10000 dataset. Scientific Data, 5(1).

3. Mazurowski et al. (2023). Segment anything model for medical image analysis. Medical Image Analysis, 89.

4. Khosla et al. (2023). GAN-based synthetic data augmentation for skin lesion classification. Neural Computing and Applications, 35(4).

5. Bissoto et al. (2020). Skin lesion synthesis with GANs. OR 2.0 Workshop.

6. Gal et al. (2022). Textual inversion for personalized text-to-image generation. arXiv:2208.01618.

7. Cubuk et al. (2020). RandAugment. CVPR Workshops.

8. Hu et al. (2021). LoRA: Low-rank adaptation of large language models. arXiv:2106.09685.

---

## Team Contribution Statement

Each member contributes equally (~33%):

**Ikhtiyor:** Baseline models, classifier training, experiments, results  
**Abdulxamid:** Diffusion fine-tuning, image generation, quality metrics  
**Ibrohim:** Visualizations, error analysis, poster, documentation  

All collaborate on final report and presentation.

---

**File:** CV25_Proposal_MedVision.pdf  
**Date:** October 21, 2025  
**Contact:** 220343@centralasian.uz
