"""
Kronecker V2 - Animated Visualization
Problem 2: Kronecker for Images and Audio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
import seaborn as sns
import os
from tqdm import tqdm

# Create output directory
os.makedirs('output', exist_ok=True)
os.makedirs('animations', exist_ok=True)

print("=" * 70)
print("🎯 KRONECKER V2 - ANIMATED VISUALIZATION")
print("   Problem 2: Kronecker for Images and Audio")
print("=" * 70)

# ============================================================
# 1. DATASET
# ============================================================

class MultimodalDataset(Dataset):
    def __init__(self, num_samples=200):
        self.num_samples = num_samples
        self.concepts = ["cat", "dog", "bird", "fish"]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        concept = random.choice(self.concepts)
        seed = self.concepts.index(concept)
        
        # Image
        img = torch.randn(3, 8, 8) * 0.5
        img[:, 2:6, 2:6] = 0.5 + seed * 0.1
        
        # Audio
        t = torch.linspace(0, 1, 64)
        freq = 100 + seed * 50
        audio = torch.sin(2 * 3.14159 * freq * t)
        audio += 0.05 * torch.randn(64)
        
        return {'text': concept, 'image': img, 'audio': audio}

# ============================================================
# 2. KRONECKER V2 MODEL
# ============================================================

class KroneckerV2(nn.Module):
    def __init__(self, patch_size=8, max_patches=3, emb_dim=32):
        super().__init__()
        self.max_patches = max_patches
        self.emb_dim = emb_dim
        
        self.char_emb = nn.Embedding(128, patch_size)
        self.text_proj = nn.Linear(patch_size, patch_size)
        self.image_proj = nn.Linear(192, patch_size)
        self.audio_proj = nn.Linear(64, patch_size)
        
        kronecker_dim = patch_size ** max_patches
        self.final_proj = nn.Linear(kronecker_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        
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
        return F.normalize(emb, dim=-1)
    
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
            audio = F.pad(audio, (0, 64 - len(audio)))
        patch = self.audio_proj(audio)
        patches = patch.unsqueeze(0).repeat(self.max_patches, 1)
        return patches

# ============================================================
# 3. TRAIN WITH HISTORY
# ============================================================

def train_with_history():
    print("\n🚀 Training Kronecker V2 with History...")
    
    model = KroneckerV2()
    dataset = MultimodalDataset(150)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    losses = []
    embeddings_history = []
    sim_history = []
    
    # Get fixed test samples
    test_concepts = ["cat", "dog", "bird", "fish"]
    test_texts = []
    test_images = []
    test_audios = []
    
    for concept in test_concepts:
        seed = test_concepts.index(concept)
        test_texts.append(concept)
        
        img = torch.randn(3, 8, 8) * 0.5
        img[:, 2:6, 2:6] = 0.5 + seed * 0.1
        test_images.append(img)
        
        t = torch.linspace(0, 1, 64)
        freq = 100 + seed * 50
        audio = torch.sin(2 * 3.14159 * freq * t)
        test_audios.append(audio)
    
    print("Training epochs...")
    for epoch in range(15):
        total_loss = 0
        count = 0
        
        for batch in loader:
            texts = batch['text']
            images = batch['image']
            audios = batch['audio']
            
            optimizer.zero_grad()
            loss = 0
            
            for i in range(len(texts)):
                t_emb = model(text=texts[i])
                i_emb = model(image=images[i])
                a_emb = model(audio=audios[i])
                
                loss += (1 - F.cosine_similarity(t_emb.unsqueeze(0), i_emb.unsqueeze(0)))
                loss += (1 - F.cosine_similarity(t_emb.unsqueeze(0), a_emb.unsqueeze(0)))
                loss += (1 - F.cosine_similarity(i_emb.unsqueeze(0), a_emb.unsqueeze(0)))
                
                for j in range(len(texts)):
                    if i != j:
                        t_emb2 = model(text=texts[j])
                        diff = F.cosine_similarity(t_emb.unsqueeze(0), t_emb2.unsqueeze(0))
                        loss += max(0, 0.3 - diff)
            
            loss = loss / len(texts)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
        
        avg_loss = total_loss / count
        losses.append(avg_loss)
        
        # Save embeddings every 2 epochs
        if epoch % 2 == 0:
            epoch_embeddings = []
            for concept in test_concepts:
                idx = test_concepts.index(concept)
                text_emb = model(text=test_texts[idx]).detach().numpy()
                image_emb = model(image=test_images[idx]).detach().numpy()
                audio_emb = model(audio=test_audios[idx]).detach().numpy()
                epoch_embeddings.extend([text_emb, image_emb, audio_emb])
            embeddings_history.append(epoch_embeddings)
            
            # Also save similarity matrix
            all_embs = []
            for idx, concept in enumerate(test_concepts):
                all_embs.append(model(text=test_texts[idx]))
                all_embs.append(model(image=test_images[idx]))
                all_embs.append(model(audio=test_audios[idx]))
            
            sim_matrix = np.zeros((12, 12))
            for i, emb1 in enumerate(all_embs):
                for j, emb2 in enumerate(all_embs):
                    sim_matrix[i, j] = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
            sim_history.append(sim_matrix)
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    torch.save(model.state_dict(), 'model.pt')
    print("✅ Model saved!")
    return model, losses, embeddings_history, sim_history, test_concepts

# ============================================================
# 4. ANIMATIONS
# ============================================================

def create_animations(losses, embeddings_history, sim_history, concepts):
    print("\n🎬 Creating Animations...")
    
    # ============================================================
    # ANIMATION 1: Training Loss
    # ============================================================
    print("\n📊 1. Creating Loss Animation...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], 'b-', linewidth=2)
    scatter, = ax.plot([], [], 'ro', markersize=8)
    ax.set_xlim(0, len(losses))
    ax.set_ylim(0, max(losses) * 1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Animation')
    ax.grid(True, alpha=0.3)
    
    def update_loss(frame):
        line.set_data(range(frame+1), losses[:frame+1])
        scatter.set_data([frame], [losses[frame]])
        return line, scatter
    
    anim_loss = FuncAnimation(fig, update_loss, frames=len(losses), 
                             interval=200, repeat=False)
    anim_loss.save('animations/training_loss.gif', writer='pillow', fps=5)
    print("   ✅ Saved: animations/training_loss.gif")
    plt.close()
    
    # ============================================================
    # ANIMATION 2: Embedding Space Evolution
    # ============================================================
    print("\n📊 2. Creating Embedding Space Animation...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = {'text': '#3498db', 'image': '#2ecc71', 'audio': '#e74c3c'}
    markers = {'text': 'o', 'image': 's', 'audio': '^'}
    labels = []
    
    # Create all points
    all_points = []
    for i, emb in enumerate(embeddings_history):
        # PCA for this step
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(np.array(emb))
        all_points.append(emb_2d)
    
    def update_embedding(frame):
        ax.clear()
        points = all_points[frame]
        
        # Plot each point
        for i, (x, y) in enumerate(points):
            modality = 'text' if i % 3 == 0 else 'image' if i % 3 == 1 else 'audio'
            concept = concepts[i // 3]
            ax.scatter(x, y, c=colors[modality], marker=markers[modality], 
                      s=200, alpha=0.8, label=f'{concept} ({modality})' if i < 3 else '')
            ax.annotate(f'{concept} {modality}', (x, y), fontsize=8, ha='center', va='bottom')
        
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_title(f'Embedding Space - Step {frame+1}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        return []
    
    anim_embed = FuncAnimation(fig, update_embedding, frames=len(all_points),
                              interval=500, repeat=False)
    anim_embed.save('animations/embedding_space.gif', writer='pillow', fps=2)
    print("   ✅ Saved: animations/embedding_space.gif")
    plt.close()
    
    # ============================================================
    # ANIMATION 3: Similarity Matrix Evolution
    # ============================================================
    print("\n📊 3. Creating Similarity Matrix Animation...")
    
    labels = []
    concepts_labels = []
    for concept in concepts:
        concepts_labels.extend([f'{concept}_text', f'{concept}_image', f'{concept}_audio'])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def update_sim(frame):
        ax.clear()
        sim_matrix = sim_history[frame]
        
        # Create heatmap
        im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_xticks(range(len(concepts_labels)))
        ax.set_yticks(range(len(concepts_labels)))
        ax.set_xticklabels(concepts_labels, rotation=90, fontsize=8)
        ax.set_yticklabels(concepts_labels, fontsize=8)
        ax.set_title(f'Similarity Matrix - Step {frame+1}')
        
        # Add colorbar
        if frame == 0:
            plt.colorbar(im, ax=ax, label='Cosine Similarity')
        
        return []
    
    anim_sim = FuncAnimation(fig, update_sim, frames=len(sim_history),
                            interval=500, repeat=False)
    anim_sim.save('animations/similarity_matrix.gif', writer='pillow', fps=2)
    print("   ✅ Saved: animations/similarity_matrix.gif")
    plt.close()
    
    print("\n🎬 All animations saved in 'animations/' folder!")
    print("   - training_loss.gif")
    print("   - embedding_space.gif")
    print("   - similarity_matrix.gif")

# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":
    # Train with history
    model, losses, embeddings_history, sim_history, concepts = train_with_history()
    
    # Create animations
    create_animations(losses, embeddings_history, sim_history, concepts)
    
    print("\n" + "=" * 70)
    print("🎉 KRONECKER V2 - ANIMATIONS COMPLETE!")
    print("=" * 70)
    print("📁 Animations saved in 'animations/' folder:")
    print("   - training_loss.gif (Loss decreasing over time)")
    print("   - embedding_space.gif (Embeddings converging)")
    print("   - similarity_matrix.gif (Similarities evolving)")
    print("\n🎯 Problem 2: SOLVED ✅")
