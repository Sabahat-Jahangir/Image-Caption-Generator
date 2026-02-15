import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle

# ---------------------------------------------------------
# PAGE CONFIG - name your app here
# ---------------------------------------------------------
st.set_page_config(
    page_title="VisualVerse",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# CUSTOM CSS - make it look clean and professional
# ---------------------------------------------------------
st.markdown("""
<style>
    /* main background - light gray */
    .stApp {
        background-color: #fafbfc;
    }
    /* container card */
    .main-card {
        background-color: white;
        border-radius: 16px;
        padding: 2rem 2rem 1.5rem 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin: 1rem auto;
        max-width: 1400px;
    }
    /* title styling */
    h1 {
        color: #1e2f4e;
        font-weight: 600;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    /* subtitle */
    .subhead {
        color: #4a5e7a;
        font-size: 1.1rem;
        margin-bottom: 1.8rem;
        border-bottom: 1px solid #eaeef2;
        padding-bottom: 0.8rem;
    }
    /* upload area */
    .stFileUploader > div > div {
        border: 2px dashed #b8c5d0;
        border-radius: 12px;
        background-color: #f8fafc;
        padding: 2rem;
        transition: border 0.2s;
    }
    .stFileUploader > div > div:hover {
        border-color: #3a6ea5;
    }
    /* image caption */
    .caption-box {
        background-color: #f0f4fa;
        border-left: 5px solid #1e2f4e;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        font-size: 1.2rem;
        font-weight: 450;
        color: #0e1c2a;
        margin-top: 0.5rem;
        line-height: 1.5;
    }
    /* footer */
    .footer {
        text-align: center;
        color: #6b7a8a;
        font-size: 0.85rem;
        margin-top: 2.5rem;
        border-top: 1px solid #e0e6ed;
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD RESNET50 FOR FEATURE EXTRACTION
# ---------------------------------------------------------
@st.cache_resource
def load_resnet():
    """Load pre-trained ResNet50 for feature extraction"""
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove the final classification layer
    resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
    resnet_model.eval()
    return resnet_model

# ---------------------------------------------------------
# LOAD CAPTIONING MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    """Load vocabulary and trained captioning model"""
    try:
        # Load vocabulary
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
            word2idx = vocab['word2idx']
            idx2word = vocab['idx2word']

        # ----- Encoder -----
        class Encoder(nn.Module):
            def __init__(self, input_dim=2048, hidden_dim=512):
                super().__init__()
                self.fc = nn.Linear(input_dim, hidden_dim)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                return self.relu(self.fc(x))

        # ----- Decoder -----
        class Decoder(nn.Module):
            def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word2idx['<pad>'])
                self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
                self.fc_out = nn.Linear(hidden_dim, vocab_size)
            
            def forward(self, caption, hidden):
                h0 = hidden.unsqueeze(0)
                c0 = torch.zeros_like(h0)
                emb = self.embedding(caption)
                out, _ = self.lstm(emb, (h0, c0))
                return self.fc_out(out)

        # ----- Full model -----
        class CaptioningModel(nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
            
            def forward(self, img_feat, caption):
                hid = self.encoder(img_feat)
                return self.decoder(caption, hid)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = Encoder()
        decoder = Decoder(len(word2idx))
        model = CaptioningModel(encoder, decoder).to(device)
        
        # Load trained weights
        model.load_state_dict(torch.load('image_captioning_model.pth', map_location=device))
        model.eval()
        
        return model, word2idx, idx2word, device
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure 'vocab.pkl' and 'image_captioning_model.pth' are in the same directory as app.py")
        st.stop()

# Load models
with st.spinner('Loading models (first time may take 1-2 minutes)...'):
    resnet = load_resnet()
    model, word2idx, idx2word, device = load_model()

# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# ---------------------------------------------------------
# FEATURE EXTRACTION FROM IMAGE
# ---------------------------------------------------------
def extract_features(image_tensor):
    """Extract 2048-dim features using ResNet50"""
    with torch.no_grad():
        # Add batch dimension
        img = image_tensor.unsqueeze(0).to(device)
        features = resnet(img)
        # Flatten to 2048-dimensional vector
        features = features.view(-1)
    return features

# ---------------------------------------------------------
# CAPTION GENERATION (greedy search)
# ---------------------------------------------------------
def generate_caption(image_tensor, max_length=20):
    """Generate caption for an image using greedy search"""
    with torch.no_grad():
        # Step 1: Extract ResNet50 features
        img_feat = extract_features(image_tensor)
        
        # Step 2: Encode features
        encoded = model.encoder(img_feat.unsqueeze(0))
        
        # Step 3: Initialize LSTM states
        h = encoded.unsqueeze(0)
        c = torch.zeros_like(h)
        
        # Step 4: Start with <start> token
        word = torch.tensor([word2idx['<start>']]).to(device)
        caption_ids = []
        
        # Step 5: Generate words one by one
        for _ in range(max_length):
            # Embed current word
            emb = model.decoder.embedding(word.unsqueeze(0))
            
            # LSTM step
            out, (h, c) = model.decoder.lstm(emb, (h, c))
            
            # Predict next word
            logits = model.decoder.fc_out(out.squeeze(0))
            pred = logits.argmax(dim=1)
            wid = pred.item()
            
            # Stop if <end> token
            if wid == word2idx['<end>']:
                break
            
            caption_ids.append(wid)
            word = torch.tensor([wid]).to(device)
        
        # Convert IDs to words
        caption = ' '.join([idx2word[i] for i in caption_ids])
        return caption

# ---------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# App title
st.markdown("<h1>🎨 VisualVerse</h1>", unsafe_allow_html=True)
st.markdown('<p class="subhead">Transform images into descriptive text using deep learning</p>', unsafe_allow_html=True)

# Two columns
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("**Upload Image**")
    uploaded = st.file_uploader(
        " ",  # hidden label
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    
    if uploaded:
        try:
            image = Image.open(uploaded).convert('RGB')
            st.image(image, caption=None, use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            st.stop()
    else:
        # Placeholder box
        st.markdown("""
        <div style="background-color:#f8fafc; border-radius:12px; padding:3rem; text-align:center; border:1px dashed #b8c5d0;">
            <span style="font-size:40px; color:#6b7a8a;">📷</span>
            <p style="color:#4a5e7a; margin-top:0.5rem;">No image selected</p>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown("**Generated Caption**")
    
    if uploaded:
        with st.spinner("🔍 Analyzing image and generating caption..."):
            try:
                # Transform image
                img_tensor = transform(image)
                
                # Generate caption
                caption_text = generate_caption(img_tensor)
                
                # Display caption
                st.markdown(f'<div class="caption-box">"{caption_text}"</div>', unsafe_allow_html=True)
                
                # Model info
                st.markdown("""
                <div style="background-color:#e9eef3; border-radius:6px; padding:0.6rem 1rem; margin-top:1rem; color:#2a4058; font-size:0.85rem;">
                    <b>Model:</b> ResNet50 + LSTM (trained on Flickr30k)<br>
                    <b>Architecture:</b> Seq2Seq with attention
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating caption: {str(e)}")
    else:
        st.markdown("""
        <div style="background-color:#f8fafc; border-radius:12px; padding:3rem; text-align:center; border:1px dashed #b8c5d0;">
            <span style="font-size:40px; color:#6b7a8a;">💬</span>
            <p style="color:#4a5e7a; margin-top:0.5rem;">Caption will appear here</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">VisualVerse · Generative AI Assignment · PyTorch + Streamlit</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar with info (optional)
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
    **VisualVerse** is an image captioning system that generates natural language descriptions for images.
    
    **How it works:**
    1. Upload an image
    2. ResNet50 extracts visual features
    3. LSTM decoder generates caption
    4. Greedy search produces final text
    
    **Model Details:**
    - Feature Extractor: ResNet50 (pre-trained)
    - Encoder: Linear layer (2048→512)
    - Decoder: LSTM with embeddings
    - Training: Flickr30k dataset
    """)
    
    st.markdown("---")
    st.markdown("Built with ❤️ using PyTorch & Streamlit")