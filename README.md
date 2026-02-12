# Word2Vec Implementation

Skip-gram model with negative sampling implemented in pure NumPy.

## Files

- **word2vec.py**: Contains the SkipGram model class and Vocabulary helper class
- **train.py**: Training script with example corpus and training loop

## Dataset

The model is trained on a small corpus of 15 sentences about animals and nature (54 unique words total). The sentences describe dogs, cats, foxes, rabbits, and their behaviors:

```
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
```

This simple dataset lets the model learn word relationships based on context. Words that appear in similar contexts (like "dog" and "cat") end up with similar embeddings.

## Usage

```bash
python train.py
```

## Results

```
Testing learned embeddings:
============================================================

Most similar words to 'dog':
  - lazy: 0.9973
  - sleeps: 0.9971
  - at: 0.9964
  - on: 0.9958
  - cat: 0.9957

Most similar words to 'cat':
  - lazy: 0.9965
  - dog: 0.9957
  - hunts: 0.9952
  - runs: 0.9952
  - tall: 0.9951

Most similar words to 'fox':
  - rabbit: 0.9974
  - forest: 0.9968
  - hops: 0.9967
  - hunts: 0.9963
  - jumps: 0.9958

Most similar words to 'tree':
  - lazy: 0.9955
  - over: 0.9951
  - forest: 0.9946
  - sleeps: 0.9946
  - fly: 0.9944

============================================================
Example: Embedding vector for 'dog'
============================================================
Shape: (50,)
First 10 dimensions: [ 0.19737093  0.29734164 -0.22623142  0.37096426  0.03926467  0.07644288
  0.50675056  0.45988272 -0.18450127  0.5278143 ]
```

## Results comments and limitations

The model learned meaningful word relationships despite the small dataset: "dog" and "cat" are highly similar (0.9957) because they both appear as subjects doing actions (barking, sleeping, playing). "Fox" and "rabbit" are even more similar (0.9974) as they both appear in outdoor contexts.

However, "Dog" is most similar to "lazy" and "sleeps" because these words frequently appear together in the training sentences (for example, "the lazy dog", "the cat sleeps"). The model seems to capture patterns rather than semantic meaning.

This happens because I used a small dataset with only a few unique words for demonstration purposes.