import numpy as np
from collections import Counter

class SkipGram:
    
    def __init__(self, vocab_size, embedding_dim=100, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        
        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_out = np.random.randn(vocab_size, embedding_dim) * 0.01
        
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def negative_sampling(self, target_idx, num_samples, exclude_indices=None):
        if exclude_indices is None:
            exclude_indices = set()
        else:
            exclude_indices = set(exclude_indices)
        
        negative_samples = []
        while len(negative_samples) < num_samples:
            sample = np.random.randint(0, self.vocab_size)
            if sample not in exclude_indices and sample not in negative_samples:
                negative_samples.append(sample)
        
        return np.array(negative_samples)
    
    def forward_and_loss(self, center_idx, context_idx, negative_samples):
        center_embed = self.W_in[center_idx]
        context_embed = self.W_out[context_idx]
        neg_embeds = self.W_out[negative_samples]
        
        pos_score = np.dot(center_embed, context_embed)
        pos_prob = self.sigmoid(pos_score)
        
        neg_scores = np.dot(neg_embeds, center_embed)
        neg_probs = self.sigmoid(neg_scores)
        
        pos_loss = -np.log(pos_prob + 1e-8)
        neg_loss = -np.sum(np.log(1 - neg_probs + 1e-8))
        
        loss = pos_loss + neg_loss
        
        cache = {
            'center_idx': center_idx,
            'context_idx': context_idx,
            'negative_samples': negative_samples,
            'center_embed': center_embed,
            'context_embed': context_embed,
            'neg_embeds': neg_embeds,
            'pos_prob': pos_prob,
            'neg_probs': neg_probs
        }
        
        return loss, cache
    
    def backward(self, cache):
        center_idx = cache['center_idx']
        context_idx = cache['context_idx']
        negative_samples = cache['negative_samples']
        center_embed = cache['center_embed']
        context_embed = cache['context_embed']
        neg_embeds = cache['neg_embeds']
        pos_prob = cache['pos_prob']
        neg_probs = cache['neg_probs']
        
        grad_W_in = np.zeros_like(self.W_in)
        grad_W_out = np.zeros_like(self.W_out)
        
        pos_grad = -(1 - pos_prob)
        
        grad_center = pos_grad * context_embed
        grad_W_out[context_idx] = pos_grad * center_embed
        
        for i, neg_idx in enumerate(negative_samples):
            neg_grad = neg_probs[i]
            grad_center += neg_grad * neg_embeds[i]
            grad_W_out[neg_idx] += neg_grad * center_embed
        
        grad_W_in[center_idx] = grad_center
        
        return grad_W_in, grad_W_out
    
    def train_step(self, center_idx, context_idx, num_neg_samples=5):
        negative_samples = self.negative_sampling(
            context_idx, 
            num_neg_samples,
            exclude_indices=[context_idx]
        )
        
        loss, cache = self.forward_and_loss(center_idx, context_idx, negative_samples)
        
        grad_W_in, grad_W_out = self.backward(cache)
        
        self.W_in -= self.lr * grad_W_in
        self.W_out -= self.lr * grad_W_out
        
        return loss
    
    def get_embedding(self, word_idx):
        return self.W_in[word_idx]
    
    def most_similar(self, word_idx, top_k=5):
        word_embed = self.W_in[word_idx]
        
        word_norm = np.linalg.norm(word_embed)
        
        similarities = []
        for i in range(self.vocab_size):
            if i == word_idx:
                continue
            other_embed = self.W_in[i]
            other_norm = np.linalg.norm(other_embed)
            
            if word_norm > 0 and other_norm > 0:
                sim = np.dot(word_embed, other_embed) / (word_norm * other_norm)
                similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class Vocabulary:
    
    def __init__(self, min_count=1):
        self.min_count = min_count
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
    def build(self, corpus):
        for sentence in corpus:
            self.word_freq.update(sentence)
        
        idx = 0
        for word, freq in self.word_freq.items():
            if freq >= self.min_count:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def __len__(self):
        return len(self.word2idx)
    
    def encode(self, sentence):
        return [self.word2idx[w] for w in sentence if w in self.word2idx]
