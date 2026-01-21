🎭 Vision-LLM for Facial Emotion Recognition of Compound Expressions (FER-CE)
📌 Project Overview

This project explores the use of Vision–Language Large Models (Vision-LLMs) for Facial Emotion Recognition of Compound Expressions (FER-CE).
Unlike traditional FER systems limited to basic emotions, this work focuses on compound emotions that naturally occur in real human interactions (e.g., happily surprised, sadly angry, fearfully disgusted).

By combining visual perception and linguistic reasoning, Vision-LLMs enable not only emotion classification but also interpretable textual explanations, bridging the gap between prediction and human understanding.

🧠 Motivation

Conventional FER approaches based on CNNs struggle with:

subtle facial cues,

overlapping Action Units (AUs),

ambiguous emotional mixtures.

Vision-LLMs (BLIP-2, LLaVA, Qwen-VL, InternVL) provide a paradigm shift by:

unifying vision and language,

generating natural language justifications,

explaining emotional nuances beyond discrete labels.

This project investigates whether Vision-LLMs can outperform Vision-only models and provide explainable multimodal predictions for compound emotions.

📂 Dataset
RAF-CE Dataset

Source: http://whdeng.cn/RAF/model4.html

Content:

Real-world facial images

14 compound emotion classes

Action Unit (AU) annotations

Challenges:

Class imbalance

Subtle facial variations

High inter-class similarity

🔄 Methodology
1️⃣ Data Preparation & Alignment

Face detection and alignment

Image normalization

Data augmentation:

illumination changes

rotations

light occlusions

Analysis of class distribution

2️⃣ Vision-LLM Training for Compound Emotions

Vision-LLMs combine:

Visual Encoder: CLIP, ViT-L, EVA

LLM: Vicuna, LLaMA, Qwen

Vision-Language Alignment: Q-Former, projection layers, cross-attention

Models Used

BLIP-2 (Zero-shot & LoRA fine-tuning)

CLIP

ResNet-EfficientNet-Vit

Learning Objectives

Multi-class classification (14 compound emotions)

Textual explanation generation, e.g.:

“The person appears happily surprised due to a smiling mouth and raised eyebrows.”

Visual Prompt Engineering

Example prompt:

Describe the emotional state and explain which facial cues
(eyes, eyebrows, mouth, muscle tension) contribute to it.

3️⃣ Multimodal Interpretation (Vision + Language)
Visual Explainability

Grad-CAM on visual encoders

Heatmaps highlighting key facial regions

Implicit AU localization (eyes, eyebrows, mouth)

Linguistic Explainability

Analysis of generated explanations

Extraction of mentioned facial cues

Coherence between text and visual attention

Alignment Validation

Ensuring that highlighted regions correspond to the predicted compound emotion.

🧪 Baseline Models (Vision-Only)

ResNet

Vision Transformer (ViT)

Swin Transformer

📊 Evaluation Metrics
Classification Metrics

Accuracy

Macro F1-Score (critical due to class imbalance)

Confusion Matrix

Textual Evaluation

BLEU

ROUGE

Optional Multimodal Alignment Metrics

CLIPScore

Faithfulness Score (text ↔ heatmap consistency)

📈 Key Contributions

End-to-end Vision-LLM pipeline for compound FER

Benchmark between Vision-Only and Vision-LLM models

Multimodal explainability linking facial regions to linguistic justifications

Interpretation of subtle emotional mixtures beyond discrete labels

🧰 Technologies Used

Python 3

Deep Learning: PyTorch, TensorFlow

Vision Models: CLIP, ViT,Resnet,EfficientNet

Vision-LLMs: BLIP-2, Clip

XAI: Grad-CAM

Visualization: Matplotlib, Seaborn

Optional Interface: Streamlit

Environment: Kaggle / Google Colab / GitHub


🚀 How to Run
git clone https://github.com/your-repo/Vision-LLM-FER-CE.git
cd Vision-LLM-FER-CE



(Optional)

streamlit run app/streamlit_app.py

🔮 Future Work

Few-shot learning for rare compound emotions

Temporal modeling (video-based FER)

AU-guided prompting

Real-time emotion analysis

Integration with HCI and affective computing systems

👥 Contributors

Hedil Jlassi

Hadil Souilem

📚 Academic Context

This project was developed as part of the AI / Data Mining / Computer Vision curriculum and follows research-oriented practices in multimodal learning and explainable AI.
