import torch
import torch.nn.functional as F
from torch import nn

"""
(b,t,v) (3,8,120)
b=batch
t= taille de caractère pat batch
v= nombre de score par caractère
"""

class Bigramme(nn.Module):
    def __init__(self,taille_vocabulaire):
        super().__init__()
        self.table=nn.Embedding(num_embeddings=taille_vocabulaire,embedding_dim=taille_vocabulaire)
    def forward(self,idx,cible=None):

        logits=self.table(idx)
        if cible is None:
            return logits, None
        #print(logits.shape)
        b,t,v=logits.shape
        logits_plat=logits.reshape(-1,v)
        cible_plat=cible.reshape(-1)
        perte=F.cross_entropy(logits_plat,cible_plat)
        #print(b)
        #print(t)
        #print(v)
        #print(logits_plat.shape)
        # print(cible_plat.shape)
        return logits, perte


if __name__=="__main__":
    modele=Bigramme(120)
    print(modele)

    idx=torch.randint(0,120,(3,8))
    cible=torch.randint(0,120,(3,8))
    logits,perte=modele(idx,cible)
    print(perte.item())
    #print(logits.shape)
    #print(logits[0])






