Image Captioning with Seq2Seq

### Overview

It is a multimodal deep learning system that generates natural language descriptions for images using a Sequence-to-Sequence (Seq2Seq) architecture. It combines computer vision and natural language processing to translate visual features into meaningful textual captions.

The system follows a staged pipeline: image feature extraction using a pre-trained CNN, followed by sequence modeling using an encoder-decoder architecture. The final result is a model capable of describing unseen images in human-readable form.

Live Demo:
[https://image-caption-generator-a2fjhthbbwmmqvpexhzxp5.streamlit.app/](https://image-caption-generator-a2fjhthbbwmmqvpexhzxp5.streamlit.app/)

---

### Purpose

The goal of this project is to bridge the gap between visual understanding and language generation. Instead of treating images and text separately, the system learns how to connect them.

This is not just about generating captions; it is about teaching a model to “see” and “describe” at the same time, which turns out to be harder than it sounds.

---

### Key Features

* Image caption generation using deep learning
* Pre-trained ResNet50 for feature extraction
* Seq2Seq architecture with Encoder-Decoder model
* LSTM/GRU-based sequence modeling
* Vocabulary building and text preprocessing
* Greedy Search and Beam Search for inference
* Evaluation using BLEU score and other metrics
* Streamlit-based deployment for real-time interaction

---

### Technologies Used

* Python
* PyTorch
* Torchvision (ResNet50)
* NLP preprocessing techniques
* Streamlit (for deployment)
* Kaggle GPU environment (T4 x2)

---

## System Architecture

The system is divided into four major stages:

1. Feature Extraction
2. Text Preprocessing
3. Seq2Seq Model Design
4. Training and Inference

---

## Project Flow

### 1. Feature Extraction Pipeline

* Images from Flickr30k dataset are processed using a pre-trained ResNet50
* The final classification layer is removed
* Each image is converted into a 2048-dimensional feature vector
* Features are cached and stored in a `.pkl` file

Why this matters:
Training a CNN + RNN together is expensive. So instead of doing that, the system “remembers” image features once and reuses them. Efficient and practical.

---

### 2. Text Preprocessing & Vocabulary

* Captions are loaded from dataset
* Text is cleaned and tokenized
* Special tokens are added:

  * `<start>`
  * `<end>`
  * `<pad>`
* Vocabulary is built based on word frequency
* Captions are converted into numerical sequences

This step ensures the model understands language in a structured way instead of raw text.

---

### 3. Seq2Seq Model Architecture

#### Encoder

* Takes 2048-dim image feature vector
* Passes through a Linear layer
* Outputs a fixed-size hidden representation (e.g., 512)

#### Decoder

* Uses LSTM or GRU
* Takes word embeddings as input
* Uses encoder output as initial hidden state
* Generates one word at a time
* Final layer maps to vocabulary size

In simple terms:
The encoder understands the image, and the decoder explains it word by word.

---

### 4. Training Phase

* Loss Function: CrossEntropy Loss
* Padding tokens are ignored during loss calculation
* Optimizer: Adam
* Model learns by comparing predicted captions with ground truth

Training involves multiple epochs where:

* Loss is tracked
* Model gradually improves caption quality

---

### 5. Inference (Caption Generation)

Two decoding strategies are implemented:

#### Greedy Search

* Selects the most probable word at each step
* Fast but not always optimal

#### Beam Search

* Explores multiple candidate sequences
* Produces better and more coherent captions

The model stops generating when `<end>` token is reached.

---

## Evaluation Metrics

* BLEU-4 Score (measures similarity with ground truth captions)
* Precision, Recall, F1-score (token-level evaluation)
* Optional: METEOR / ROUGE

---

## Output Examples

The system generates:

* Input image
* Ground truth caption
* Model-generated caption

It also provides:

* Loss curves (training vs validation)
* Quantitative performance metrics

---

## Real-World Applications

This system can be used in multiple industries:

* Assistive Technology (for visually impaired users)
* Social Media Platforms (automatic caption generation)
* E-commerce (auto product descriptions from images)
* Digital Asset Management (image tagging and indexing)
* Surveillance Systems (scene understanding)
* Healthcare (medical image description assistance)

Organizations that can benefit:

* Tech companies working on AI products
* E-commerce platforms
* Accessibility-focused organizations
* Media and content platforms

---

## How to Run the Project

### Local Setup

1. Clone the repository

```bash id="2jv9yx"
git clone <your-repo-link>
cd neural-storyteller
```

2. Install dependencies

```bash id="9c9q4m"
npm install
```

3. Run the application

```bash id="6fd3cs"
npm start
```

---

### Streamlit Deployment

If running locally with Streamlit:

```bash id="c2j1du"
streamlit run app.py
```

---

### Live Application

Access the deployed version:
[https://image-caption-generator-a2fjhthbbwmmqvpexhzxp5.streamlit.app/](https://image-caption-generator-a2fjhthbbwmmqvpexhzxp5.streamlit.app/)

---

## Challenges Faced

* Handling large-scale image feature extraction efficiently
* Building a balanced vocabulary without noise
* Preventing overfitting during training
* Generating meaningful captions instead of repetitive phrases
* Implementing Beam Search correctly

---

## Future Improvements

* Use Transformer-based models instead of RNNs
* Add attention mechanism for better context understanding
* Improve caption diversity
* Support multilingual caption generation
* Optimize inference speed for real-time systems


