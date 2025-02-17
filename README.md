# CS 7643 - Image Captioning
## How Image Captioning Evolved Over Time  

This project takes a deep dive into **image captioning**, a task where a model looks at an image and generates a description in natural language. The goal is not just to compare models, but to **understand how vision and NLP have evolved** over time to improve image captioning.  

We explore two key approaches:  
1. **CNN+LSTM with attention** - A more traditional encoder-decoder setup.  
2. **ViT+GPT-2** - A transformer-based approach that reflects modern advancements.  

These two architectures show **how we moved from CNNs and sequential models to transformers** for solving complex AI tasks.

---

## **The Evolution of Image Captioning**  
### **Early Approaches: CNN + LSTM**  
Before transformers became mainstream, image captioning relied on a **CNN + RNN (LSTM/GRU) setup**.  
- **CNNs** extract features from an image.  
- **LSTMs** generate a sequence of words based on those features.  
- **Attention mechanisms** help the model focus on different parts of the image as it generates text.  

While this approach worked well, it had **limitations**:  
- **LSTMs struggle with long-range dependencies**, which affects text coherence.  
- **CNNs compress images into a single feature vector**, potentially losing spatial relationships.

### **The Shift to Transformers: ViT + GPT-2**  
With the rise of **Vision Transformers (ViT) and powerful language models like GPT-2**, image captioning took a big leap forward:  
- **ViT treats an image as a sequence of patches**, learning deep contextual relationships.  
- **GPT-2 handles language generation more effectively than LSTMs**, allowing for more natural, fluent captions.  
- **Self-attention replaces recurrence**, making training more parallelizable and scalable.

This shift **bridged the gap between vision and NLP**, enabling **better feature representation, improved text generation, and faster training**.

---

## **How Image Captioning Works**
At a high level, the image captioning process follows a **sequence-to-sequence (seq2seq) framework**:

1. **The encoder extracts image features**  
   - **CNN (ResNet) or ViT** processes the image into feature embeddings.  
2. **The decoder generates captions**  
   - **LSTM (for CNN-based model) or GPT-2 (for ViT-based model)** predicts words one by one.  
3. **Attention mechanisms help the model focus**  
   - The model learns to "look" at different parts of the image while generating the caption.  

This combination of **vision and NLP** is what makes modern image captioning work so well.

---

## **Models Used**

### **1. CNN+LSTM with Attention**
- **Encoder:** A pretrained **ResNet** extracts image features.  
- **Decoder:** **LSTM with attention** generates text.  
- **Why Attention?** Instead of relying on a single compressed feature vector, attention allows the model to dynamically focus on different parts of the image while generating each word.  

### **2. ViT + GPT-2**
- **Encoder:** **Vision Transformer (ViT)**, which processes an image as a series of patches.  
- **Decoder:** **GPT-2**, a powerful transformer-based language model.  
- **Why Transformers?** They allow for **better feature extraction and more natural text generation** compared to CNNs and LSTMs.  

---

## **Training and Experiments**

### **1. Dataset**
The models were trained on **Flickr8k**, a dataset containing 8,000 images with multiple human-annotated captions.

### **2. Training Strategy**
- **CNN+LSTM:** Used **SGD optimizer** and **cross-entropy loss**.  
- **ViT+GPT-2:** Used **AdamW optimizer** and **cross-entropy loss**.  
- **Early stopping** was applied to prevent overfitting.  
- **Hyperparameters like learning rate and dropout** were tuned for better performance.

### **3. Key Challenges**
- **Data Limitations** – Transformer-based models require massive datasets. Training on **Flickr8k** made it challenging to generalize well.  
- **Computational Cost** – The **ViT-GPT2 model took ~15 hours for 5 epochs**, requiring optimizations.  
- **Text Coherence** – LSTMs produced **generic captions**, while transformers **generated richer descriptions**.  

---

## **Results**
| Model | Training Loss | Validation Loss | Caption Quality |
|--------|----------------|----------------|----------------|
| CNN+LSTM | 3.40 | 3.53 | Decent but often generic captions |
| ViT+GPT-2 | **2.23** | **2.34** | More descriptive and natural |

- **ViT+GPT-2 outperformed CNN+LSTM in both loss and quality**.  
- **CNN+LSTM struggled with longer sentences and fine details**.  
- **Transformer-based models handled context better, leading to more human-like captions**.

While CNN+LSTM still has **practical use cases**, **transformers are the future** of deep learning for vision-language tasks.

For more details, check out the **full report** in this repository, which goes deeper into the evolution of image captioning.
