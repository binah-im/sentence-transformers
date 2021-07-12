"""
loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from numpy import dot
from numpy.linalg import norm


# Load pre-trained Sentence Transformer Model (based on DistilBERT)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Embed a list of sentences
sentences = ['hello how are you? I am fine thank you.',
             'Can you tell me where is Israel? Israel is in the middle east.',
             'what did you do yesterday? I watched tv.']
sentence_embeddings = model.encode(sentences)

def cos_sim (x,y):
    return dot(x, y) / (norm(x) * norm(y))

se0 = sentence_embeddings[0]
se1 = sentence_embeddings[1]
se2 = sentence_embeddings[2]

print('similarity of',sentences[0],sentences[1],'=',cos_sim(se0,se1))
print('similarity of',sentences[0],sentences[2],'=',cos_sim(se0,se2))

# The result is a list of sentence embeddings as numpy arrays
#for sentence, embedding in zip(sentences, sentence_embeddings):
#    print("Sentence:", sentence)
#    print("Embedding:", embedding)
#    print("")
