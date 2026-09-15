from torch import nn
import torch
torch.manual_seed(42)
table=nn.Embedding(num_embeddings=120, embedding_dim=128)
print(table.weight.shape)

identifiant=torch.tensor([55,68,55],dtype=torch.int64)
print(identifiant.shape)
print(identifiant)
vecteur=table(identifiant)
print(vecteur)
print(vecteur.shape)
print(vecteur[:,:4])
