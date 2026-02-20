import numpy as np
import matplotlib.pyplot as plt
from word2vec import SkipGram, Vocabulary
import pickle
import os

def load_model():
    data = np.load('model_data.npz')
    
    model = SkipGram(
        vocab_size=int(data['vocab_size']),
        embedding_dim=int(data['embedding_dim'])
    )
    model.W_in = data['W_in']
    model.W_out = data['W_out']
    
    loss_history = data['loss_history']
    
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    return model, vocab, loss_history


def plot_loss(loss_history, output_dir='visualizations'):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(output_path, dpi=300)
    print(f"Loss plot saved as '{output_path}'")
    plt.close()


def compute_similarity(model, vocab, word1, word2):
    if word1 not in vocab.word2idx or word2 not in vocab.word2idx:
        return None
    
    idx1 = vocab.word2idx[word1]
    idx2 = vocab.word2idx[word2]
    
    vec1 = model.W_in[idx1]
    vec2 = model.W_in[idx2]
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 > 0 and norm2 > 0:
        return np.dot(vec1, vec2) / (norm1 * norm2)
    return None


def plot_embeddings_2d(model, vocab, word_categories):
    from sklearn.manifold import TSNE
    
    all_words = []
    all_categories = []
    for category, words in word_categories.items():
        for word in words:
            if word in vocab.word2idx:
                all_words.append(word)
                all_categories.append(category)
    
    if len(all_words) < 2:
        print("Not enough words found in vocabulary for 2D plot")
        return
    
    word_indices = [vocab.word2idx[w] for w in all_words]
    embeddings = np.array([model.W_in[idx] for idx in word_indices])
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_words)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    colors = {
        'Royalty': '#e74c3c',
        'Medical': '#3498db',
        'Technology': '#2ecc71',
        'Food': '#f39c12',
        'Education': '#9b59b6',
    }
    
    plt.figure(figsize=(14, 10))
    
    for category in colors.keys():
        mask = [cat == category for cat in all_categories]
        if any(mask):
            indices = [i for i, m in enumerate(mask) if m]
            plt.scatter(
                [embeddings_2d[i, 0] for i in indices],
                [embeddings_2d[i, 1] for i in indices],
                s=150,
                alpha=0.7,
                c=colors[category],
                label=category,
                edgecolors='black',
                linewidth=1
            )
    
    for i, word in enumerate(all_words):
        plt.annotate(
            word,
            xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    plt.xlabel('t-SNE Dimension 1', fontsize=13)
    plt.ylabel('t-SNE Dimension 2', fontsize=13)
    plt.title('Word Embeddings in 2D (t-SNE with Category Colors)', fontsize=15, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    output_path = os.path.join('visualizations', 'embeddings_2d.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"2D embeddings plot saved as '{output_path}'")
    plt.close()


def main():
    print("=" * 60)
    print("Word2Vec Visualizations")
    print("=" * 60)
    print()
    
    print("Loading model and vocabulary...")
    model, vocab, loss_history = load_model()
    print(f"Loaded model with {model.vocab_size} words, {model.embedding_dim} dimensions")
    print()
    
    print("1. Generating loss plot...")
    plot_loss(loss_history)
    print()
    
    print("2. Testing semantic relationships (cosine similarity):")
    print("-" * 60)
    
    word_pairs = [
        ('king', 'queen', 'Royalty pair'),
        ('king', 'prince', 'Royalty family'),
        ('doctor', 'hospital', 'Medical domain'),
        ('doctor', 'nurse', 'Medical professionals'),
        ('programmer', 'computer', 'Tech domain'),  
        ('programmer', 'engineer', 'Tech professionals'),
        ('chef', 'cook', 'Food professionals'),
        ('teacher', 'student', 'Education pair'),
        ('king', 'doctor', 'Cross-domain'),
        ('programmer', 'chef', 'Cross-domain'),
    ]
    
    print(f"\n{'Word 1':<15} {'Word 2':<15} {'Similarity':<12} {'Category'}")
    print("-" * 65)
    
    for word1, word2, category in word_pairs:
        sim = compute_similarity(model, vocab, word1, word2)
        if sim is not None:
            print(f"{word1:<15} {word2:<15} {sim:.4f}       {category}")
    
    print()
    print("-" * 60)
    print()
    
    print("3. Generating 2D embedding visualization...")
    word_categories = {
        'Royalty': ['king', 'queen', 'prince', 'princess', 'castle', 'throne'],
        'Medical': ['doctor', 'nurse', 'hospital', 'patient', 'surgeon'],
        'Technology': ['programmer', 'engineer', 'computer', 'software', 'code'],
        'Food': ['chef', 'cook', 'restaurant', 'food', 'kitchen'],
        'Education': ['teacher', 'student', 'school', 'professor', 'university']
    }
    plot_embeddings_2d(model, vocab, word_categories)
    print()
    
    print("=" * 60)
    print("All visualizations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
