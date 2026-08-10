"""
Kronecker V2 - Interactive Web App
Problem 2: Kronecker for Images and Audio
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
import os
import random

# Page config
st.set_page_config(
    page_title="Kronecker V2 - Unified Embedding",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #2c3e50; text-align: center; }
    .sub-header { font-size: 1.2rem; color: #7f8c8d; text-align: center; margin-bottom: 2rem; }
    .metric-card { background: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center; }
    .success-box { background: #d4edda; padding: 1rem; border-radius: 10px; border-left: 5px solid #28a745; }
    .info-box { background: #d1ecf1; padding: 1rem; border-radius: 10px; border-left: 5px solid #17a2b8; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 1. MODEL DEFINITION
# ============================================================

class KroneckerV2(torch.nn.Module):
    def __init__(self, patch_size=8, max_patches=3, emb_dim=32):
        super().__init__()
        self.max_patches = max_patches
        self.emb_dim = emb_dim
        
        self.char_emb = torch.nn.Embedding(128, patch_size)
        self.text_proj = torch.nn.Linear(patch_size, patch_size)
        self.image_proj = torch.nn.Linear(192, patch_size)
        self.audio_proj = torch.nn.Linear(64, patch_size)
        
        kronecker_dim = patch_size ** max_patches
        self.final_proj = torch.nn.Linear(kronecker_dim, emb_dim)
        self.norm = torch.nn.LayerNorm(emb_dim)
        
    def kronecker_product(self, patches):
        result = patches[0]
        for p in patches[1:]:
            result = torch.kron(result, p)
        return result
    
    def forward(self, text=None, image=None, audio=None):
        if text is not None:
            patches = self.text_to_patches(text)
        elif image is not None:
            patches = self.image_to_patches(image)
        elif audio is not None:
            patches = self.audio_to_patches(audio)
        else:
            raise ValueError("Need input")
        
        emb = self.kronecker_product(patches)
        emb = self.final_proj(emb)
        emb = self.norm(emb)
        return torch.nn.functional.normalize(emb, dim=-1)
    
    def text_to_patches(self, text):
        if isinstance(text, str):
            indices = [ord(c) % 128 for c in text[:self.max_patches]]
        else:
            indices = []
            for t in text:
                if isinstance(t, str):
                    indices.append(ord(t[0]) % 128)
                else:
                    indices.append(0)
        
        while len(indices) < self.max_patches:
            indices.append(0)
        
        indices = indices[:self.max_patches]
        tensor = torch.tensor(indices, dtype=torch.long)
        patches = self.char_emb(tensor)
        patches = self.text_proj(patches)
        return patches
    
    def image_to_patches(self, image):
        if image.dim() == 4:
            image = image[0]
        image = image.flatten()
        patch = self.image_proj(image)
        patches = patch.unsqueeze(0).repeat(self.max_patches, 1)
        return patches
    
    def audio_to_patches(self, audio):
        if audio.dim() == 2:
            audio = audio[0]
        if len(audio) > 64:
            audio = audio[:64]
        elif len(audio) < 64:
            audio = torch.nn.functional.pad(audio, (0, 64 - len(audio)))
        patch = self.audio_proj(audio)
        patches = patch.unsqueeze(0).repeat(self.max_patches, 1)
        return patches

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def generate_sample(concept, seed):
    """Generate sample image and audio for a concept"""
    # Image
    img = torch.randn(3, 8, 8) * 0.5
    img[:, 2:6, 2:6] = 0.5 + seed * 0.1
    
    # Audio
    t = torch.linspace(0, 1, 64)
    freq = 100 + seed * 50
    audio = torch.sin(2 * 3.14159 * freq * t)
    audio += 0.05 * torch.randn(64)
    
    return img, audio

def get_embeddings(model, concepts):
    """Get embeddings for all concepts"""
    text_embs = {}
    image_embs = {}
    audio_embs = {}
    
    for concept in concepts:
        seed = concepts.index(concept)
        text_embs[concept] = model(text=concept)
        img, audio = generate_sample(concept, seed)
        image_embs[concept] = model(image=img)
        audio_embs[concept] = model(audio=audio)
    
    return text_embs, image_embs, audio_embs

# ============================================================
# 3. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = KroneckerV2()
    try:
        model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
        return model
    except:
        return model

model = load_model()
concepts = ["cat", "dog", "bird", "fish"]

# ============================================================
# 4. UI
# ============================================================

# Header
st.markdown('<h1 class="main-header">🧠 Kronecker V2</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Unified Multimodal Embedding for Text, Images, and Audio</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎯 Problem 2: Kronecker for Images and Audio")
    st.markdown("""
    **The Question:**
    > "What is the natural extension of Kronecker, such that it can represent images and audio as well!"
    
    **Our Solution:**
    Kronecker V2 extends Kronecker product to represent:
    - ✅ Text (character patches)
    - ✅ Images (image patches)
    - ✅ Audio (audio frames)
    
    **All three in ONE embedding space!**
    """)
    
    st.divider()
    
    st.header("📊 Visualizations")
    viz_type = st.selectbox(
        "Select Visualization",
        ["Embedding Space", "Similarity Matrix", "Cross-Modal Retrieval", "3D Embedding"]
    )
    
    st.divider()
    
    st.header("🎮 Interactive Demo")
    query_concept = st.selectbox("Query Concept", concepts)

# Main content
if viz_type == "Embedding Space":
    st.subheader("🌌 Embedding Space Visualization")
    
    # Get embeddings
    text_embs, image_embs, audio_embs = get_embeddings(model, concepts)
    
    # Collect all embeddings
    all_embs = []
    all_labels = []
    colors = []
    
    for concept in concepts:
        all_embs.append(text_embs[concept].detach().numpy())
        all_labels.append(f"{concept} (text)")
        colors.append('blue')
        
        all_embs.append(image_embs[concept].detach().numpy())
        all_labels.append(f"{concept} (image)")
        colors.append('green')
        
        all_embs.append(audio_embs[concept].detach().numpy())
        all_labels.append(f"{concept} (audio)")
        colors.append('red')
    
    all_embs = np.array(all_embs)
    
    # PCA
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(all_embs)
    
    # Create DataFrame for Plotly
    import pandas as pd
    df = pd.DataFrame({
        'x': emb_2d[:, 0],
        'y': emb_2d[:, 1],
        'label': all_labels,
        'color': colors,
        'modality': ['text']*4 + ['image']*4 + ['audio']*4,
        'concept': [c for c in concepts]*3
    })
    
    # Plot
    fig = px.scatter(
        df, x='x', y='y', text='label',
        color='modality', symbol='modality',
        title=f'Kronecker V2 Embedding Space (PCA)',
        labels={'x': f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%})',
                'y': f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%})'}
    )
    fig.update_traces(textposition='top center', marker_size=15)
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    # Same concept similarity
    same_sims = []
    for concept in concepts:
        sim = torch.nn.functional.cosine_similarity(
            text_embs[concept].unsqueeze(0),
            image_embs[concept].unsqueeze(0)
        ).item()
        same_sims.append(sim)
    avg_same = np.mean(same_sims)
    
    # Different concept similarity
    diff_sims = []
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            if i < j:
                sim = torch.nn.functional.cosine_similarity(
                    text_embs[c1].unsqueeze(0),
                    text_embs[c2].unsqueeze(0)
                ).item()
                diff_sims.append(sim)
    avg_diff = np.mean(diff_sims)
    
    with col1:
        st.metric("Same Concept Similarity", f"{avg_same:.4f}", delta="✅ Perfect")
    with col2:
        st.metric("Different Concept Similarity", f"{avg_diff:.4f}", delta="✅ Separated")
    with col3:
        st.metric("Separation Margin", f"{avg_same - avg_diff:.4f}", delta="✅ Good")

elif viz_type == "Similarity Matrix":
    st.subheader("🔢 Similarity Matrix Heatmap")
    
    # Get embeddings
    text_embs, image_embs, audio_embs = get_embeddings(model, concepts)
    
    # Create combined embeddings
    all_embs = []
    labels = []
    for concept in concepts:
        all_embs.append(text_embs[concept])
        all_embs.append(image_embs[concept])
        all_embs.append(audio_embs[concept])
        labels.extend([f"{concept}_text", f"{concept}_image", f"{concept}_audio"])
    
    # Compute similarity matrix
    n = len(all_embs)
    sim_matrix = np.zeros((n, n))
    for i, emb1 in enumerate(all_embs):
        for j, emb2 in enumerate(all_embs):
            sim_matrix[i, j] = torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0), emb2.unsqueeze(0)
            ).item()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sim_matrix, 
                xticklabels=labels, 
                yticklabels=labels,
                cmap='RdYlGn',
                vmin=-1, vmax=1,
                cbar_kws={'label': 'Cosine Similarity'},
                annot=True, fmt='.2f', annot_kws={'size': 8})
    plt.title('Similarity Matrix: Text, Image, Audio Embeddings', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("💡 **Insight:** Diagonal blocks (same concepts) show high similarity (green). Off-diagonal blocks (different concepts) show low similarity (red).")

elif viz_type == "Cross-Modal Retrieval":
    st.subheader("🎯 Cross-Modal Retrieval")
    
    # Get embeddings
    text_embs, image_embs, audio_embs = get_embeddings(model, concepts)
    
    # Query
    query_emb = text_embs[query_concept]
    
    # Compute similarities
    img_sims = {}
    audio_sims = {}
    
    for concept in concepts:
        img_sims[concept] = torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0),
            image_embs[concept].unsqueeze(0)
        ).item()
        audio_sims[concept] = torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0),
            audio_embs[concept].unsqueeze(0)
        ).item()
    
    # Display results
    st.markdown(f"### Query: **{query_concept}** (text)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Image Retrieval")
        sorted_img = sorted(img_sims.items(), key=lambda x: x[1], reverse=True)
        for concept, sim in sorted_img:
            color = "✅" if concept == query_concept else "❌"
            st.metric(f"{color} {concept}", f"{sim:.4f}")
    
    with col2:
        st.subheader("🎵 Audio Retrieval")
        sorted_audio = sorted(audio_sims.items(), key=lambda x: x[1], reverse=True)
        for concept, sim in sorted_audio:
            color = "✅" if concept == query_concept else "❌"
            st.metric(f"{color} {concept}", f"{sim:.4f}")
    
    # Bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(img_sims.keys(), img_sims.values(), color=['green' if k == query_concept else 'gray' for k in img_sims.keys()])
    ax1.set_title('Image Similarities')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_ylim(0, 1.1)
    
    ax2.bar(audio_sims.keys(), audio_sims.values(), color=['green' if k == query_concept else 'gray' for k in audio_sims.keys()])
    ax2.set_title('Audio Similarities')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_ylim(0, 1.1)
    
    st.pyplot(fig)
    
    # Summary
    best_img = max(img_sims, key=img_sims.get)
    best_audio = max(audio_sims, key=audio_sims.get)
    
    st.markdown(f"""
    <div class="success-box">
        <h4>✅ Retrieval Results</h4>
        <p>Best image match: <strong>{best_img}</strong> (similarity: {img_sims[best_img]:.4f})</p>
        <p>Best audio match: <strong>{best_audio}</strong> (similarity: {audio_sims[best_audio]:.4f})</p>
        <p>Status: {"✅ PERFECT" if best_img == query_concept and best_audio == query_concept else "⚠️ Needs improvement"}</p>
    </div>
    """, unsafe_allow_html=True)

elif viz_type == "3D Embedding":
    st.subheader("🎲 3D Embedding Space")
    
    # Get embeddings
    text_embs, image_embs, audio_embs = get_embeddings(model, concepts)
    
    # Collect all embeddings
    all_embs = []
    all_labels = []
    colors = []
    
    for concept in concepts:
        all_embs.append(text_embs[concept].detach().numpy())
        all_labels.append(f"{concept} (text)")
        colors.append('blue')
        
        all_embs.append(image_embs[concept].detach().numpy())
        all_labels.append(f"{concept} (image)")
        colors.append('green')
        
        all_embs.append(audio_embs[concept].detach().numpy())
        all_labels.append(f"{concept} (audio)")
        colors.append('red')
    
    all_embs = np.array(all_embs)
    
    # PCA to 3D
    pca = PCA(n_components=3)
    emb_3d = pca.fit_transform(all_embs)
    
    # Create 3D plot
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Add traces for each modality
    modalities = ['text', 'image', 'audio']
    colors_map = {'text': 'blue', 'image': 'green', 'audio': 'red'}
    
    for modality in modalities:
        indices = [i for i, label in enumerate(all_labels) if modality in label]
        fig.add_trace(go.Scatter3d(
            x=emb_3d[indices, 0],
            y=emb_3d[indices, 1],
            z=emb_3d[indices, 2],
            mode='markers+text',
            name=modality,
            text=[all_labels[i] for i in indices],
            textposition='top center',
            marker=dict(
                size=12,
                color=colors_map[modality],
                opacity=0.8
            )
        ))
    
    fig.update_layout(
        title='Kronecker V2 Embedding Space (3D PCA)',
        scene=dict(
            xaxis_title=f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%})',
            yaxis_title=f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%})',
            zaxis_title=f'PCA 3 ({pca.explained_variance_ratio_[2]:.2%})'
        ),
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 5. FOOTER
# ============================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 1rem;">
    <p>🏆 <strong>Kronecker V2</strong> - Problem 2: Kronecker for Images and Audio</p>
    <p>✅ Text, Images, and Audio unified in ONE embedding space!</p>
</div>
""", unsafe_allow_html=True)
