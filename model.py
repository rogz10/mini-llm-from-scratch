import torch
from torch import nn

from tokenizer import batch_x
from visualisation import afficher_embedding, projeter_embedding

torch.manual_seed(42)
#table=nn.Embedding(num_embeddings=120, embedding_dim=128)
print(f"taille du batch x:{len(batch_x)}")

table=nn.Embedding(num_embeddings=120, embedding_dim=128)
points=projeter_embedding(table.weight)
#afficher_embedding(points)
#print(points.shape)
print(table.weight.shape)
print(batch_x.shape)
vecteur=table(batch_x)
print(vecteur.shape)
#print(table.shape)

#identifiant=torch.tensor([55,68,55],dtype=torch.int64)
#identifiant = torch.tensor( [[55, 68, 55], [68, 55, 68]],dtype=torch.int64)
"""
print(identifiant.shape)
print(identifiant)
vecteur=table(identifiant)
print(vecteur)
print(vecteur.shape)
#print(vecteur[:,:4])
ex_1=vecteur[:,:,:4]
print(ex_1.shape)
print(ex_1[:4])
"""
