import torch
import torch.nn.functional as F
from torch import nn



class Bigramme(nn.Module):
    def __init__(self,taille_vocabulaire):
        super().__init__()
        self.table=nn.Embedding(num_embeddings=120,embedding_dim=120)
    def forward(self,idx,cible=None):
        logits=self.table(idx)
        print(logits.shape)
        b,t,v=logits.shape
        print(b)
        print(t)
        print(v)
        return logits





if __name__=="__main__":
    modele=Bigramme(120)
    print(modele)

    idx=torch.randint(0,120,(3,8))
    logits=modele(idx)
    print(logits.shape)




