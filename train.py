import numpy as np
from word2vec import SkipGram, Vocabulary

np.random.seed(42)

corpus_text = """
the king rules the kingdom with wisdom
the queen sits on the throne beside the king
a prince and princess live in the castle
the royal family governs the nation
a king wears a golden crown
the queen wears a beautiful dress
the doctor examines the patient carefully
a nurse helps the doctor in the hospital
hospitals provide medical care to patients
the surgeon performs operations in the hospital
doctors and nurses work together
medical treatment saves many lives
the programmer writes code on the computer
software engineers develop new applications
the computer runs programs quickly
coding requires logical thinking
programmers debug software errors
technology companies hire many engineers
a chef cooks delicious food in the kitchen
the restaurant serves fresh meals daily
people enjoy eating at nice restaurants
the cook prepares ingredients carefully
chefs create new recipes frequently
good food brings people together
students study hard for exams
the teacher explains lessons clearly
schools educate young people
a professor lectures at the university
learning requires dedication and practice
education opens many opportunities
the artist paints beautiful pictures
musicians play instruments skillfully
the painter uses bright colors
art galleries display creative works
music and painting inspire people
creative expression enriches life
the athlete trains every day
football players compete in matches
the team won the championship game
sports require physical fitness
exercise keeps the body healthy
competition builds strong character
rain falls from dark clouds
the sun shines brightly today
winter brings cold weather
snow covers the mountains
weather affects outdoor activities
climate changes with seasons
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
    loss_history = []
    
    for epoch in range(epochs):
        np.random.shuffle(pairs)
        
        total_loss = 0
        for center_idx, context_idx in pairs:
            loss = model.train_step(center_idx, context_idx, num_neg_samples=5)
            total_loss += loss
        
        avg_loss = total_loss / num_pairs
        loss_history.append(avg_loss)
        
        if verbose and (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    return loss_history


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
    loss_history = train(model, pairs, epochs=25, verbose=True)
    print("-" * 60)
    
    import os
    os.makedirs('model', exist_ok=True)
    
    np.savez('model/model_data.npz', 
             W_in=model.W_in, 
             W_out=model.W_out,
             loss_history=loss_history,
             vocab_size=model.vocab_size,
             embedding_dim=model.embedding_dim)
    
    import pickle
    with open('model/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    
    print("\nModel and vocabulary saved to model/model_data.npz and model/vocab.pkl")


if __name__ == "__main__":
    main()
