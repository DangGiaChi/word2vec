import numpy as np
from word2vec import SkipGram, Vocabulary

np.random.seed(42)

corpus_text = """
the quick brown fox jumps over the lazy dog
the dog barks at the cat
the cat sleeps on the mat
a brown cat and a black dog play together
the fox runs through the forest
dogs and cats are popular pets
the lazy cat rests under the tree
a quick rabbit hops near the fox
the brown rabbit eats fresh grass
birds fly over the tall tree
the dog and cat are friends
cats enjoy sleeping in sunny spots
the quick fox hunts in the forest
rabbits hop around the garden
dogs play fetch with their owners
"""

def preprocess_text(text):
    sentences = text.strip().lower().split('\n')
    
    tokenized = []
    for sentence in sentences:
        words = sentence.strip().split()
        if words:
            tokenized.append(words)
    
    return tokenized


def generate_training_pairs(sentences, vocab, window_size=2):
    pairs = []
    
    for sentence in sentences:
        indices = vocab.encode(sentence)
        
        for center_pos, center_idx in enumerate(indices):
            for offset in range(-window_size, window_size + 1):
                context_pos = center_pos + offset
                
                if offset == 0 or context_pos < 0 or context_pos >= len(indices):
                    continue
                
                context_idx = indices[context_pos]
                pairs.append((center_idx, context_idx))
    
    return pairs


def train(model, pairs, epochs=10, verbose=True):
    num_pairs = len(pairs)
    
    for epoch in range(epochs):
        np.random.shuffle(pairs)
        
        total_loss = 0
        for center_idx, context_idx in pairs:
            loss = model.train_step(center_idx, context_idx, num_neg_samples=5)
            total_loss += loss
        
        avg_loss = total_loss / num_pairs
        
        if verbose and (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")


def main():
    print("=" * 60)
    print("Word2Vec Skip-Gram with Negative Sampling")
    print("=" * 60)
    print()
    
    sentences = preprocess_text(corpus_text)
    print(f"Number of sentences: {len(sentences)}")

    vocab = Vocabulary(min_count=1)
    vocab.build(sentences)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample words: {list(vocab.word2idx.keys())[:10]}")
    
    print("\nGenerating training pairs...")
    pairs = generate_training_pairs(sentences, vocab, window_size=2)
    print(f"Number of training pairs: {len(pairs)}")
    
    model = SkipGram(
        vocab_size=len(vocab),
        embedding_dim=50,
        learning_rate=0.025
    )
    print(f"Embedding dimension: {model.embedding_dim}")
    
    print("\nTraining model...")
    print("-" * 60)
    train(model, pairs, epochs=20, verbose=True)
    print("-" * 60)
    
    print("\nTesting learned embeddings:")
    print("=" * 60)
    
    test_words = ['dog', 'cat', 'fox', 'tree']
    
    for word in test_words:
        if word in vocab.word2idx:
            word_idx = vocab.word2idx[word]
            similar = model.most_similar(word_idx, top_k=5)
            
            print(f"\nMost similar words to '{word}':")
            for idx, sim in similar:
                similar_word = vocab.idx2word[idx]
                print(f"  - {similar_word}: {sim:.4f}")
    
    print("\n" + "=" * 60)
    print("Example: Embedding vector for 'dog'")
    print("=" * 60)
    if 'dog' in vocab.word2idx:
        dog_idx = vocab.word2idx['dog']
        dog_embedding = model.get_embedding(dog_idx)
        print(f"Shape: {dog_embedding.shape}")
        print(f"First 10 dimensions: {dog_embedding[:10]}")


if __name__ == "__main__":
    main()
