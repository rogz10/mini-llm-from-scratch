import torch
import torch.nn.functional as F
from torch import nn

"""
(b,t,v) (3,8,120)
b=batch
t= taille de caractère pat batch
v= nombre de score par caractère
logit = score pour chaque cactere
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
class Tete(nn.Module):
    """tete d'attention causale : chaque lettre regarde les lettres d'avant et elle-même
    jamais celles d'après et en retire une nouvelle description.

    paramètres:
        n_embd:le nombre de nombres qui décrivent chaque lettre à l'entrée (32)
        taille_tete:le nombre de nombres de chaque requête, clé et valeur (16 ou 32)

        Q = requête : ce que la lettre cherche dans son passé ( ce que je cherche)
        K = clé     : ce que la lettre est, pour être trouvée (qui je suis )
        V = valeur  : ce que la lettre transmet à ceux qui la regardent (ce que je donne quand on me regarde)
    formes:
        entrée x: (b, t, n_embd),b = extraits du batch, t = lettres par extrait
        scores, poids: (b, t, t)ligne i = combien la lettre i regarde chaque lettre
        sortie: (b, t, taille_tete)  chaque lettre, enrichie de son passé
    """

    def __init__(self, n_embd,taille_tete):
        super().__init__()
        # n_embd nombres par lettre , taille_tete nombres
        self.Q=nn.Linear(n_embd,taille_tete,bias=False)    # requete
        self.K=nn.Linear(n_embd,taille_tete,bias=False)    # clé
        self.V=nn.Linear(n_embd,taille_tete,bias=False)    # valeur
        self.taille_tete=taille_tete

    def forward(self,x):
        # (b, t, n_embd) -> (b, t, taille_tete)
        Q=self.Q(x)
        K=self.K(x)
        V=self.V(x)
        taille_tete=self.taille_tete

       # notes pour chaque paire
        scores= Q @ K.transpose(-2,-1)
        # divise par la racine de taille_tete pour éviter un softmax trop grand
        scores=scores *taille_tete**-0.5

        # nombre de lettres de l'extrait
        _,t,_=x.shape
        # triangle (t, t) : 1 = lettre autorisée (passé et soi-même), 0 = lettre d'après
        triangle_matrice=torch.tril(torch.ones(t,t))

        # masque, -inf lettres d'après pour que softmax leur donne 0 %
        scores=scores.masked_fill(triangle_matrice==0,float("-inf"))
        # notes -> parts, chaque ligne fait 100 %
        poids=F.softmax(scores,dim=-1)

        # chaque lettre récupère les valeurs de son passé, selon ses parts;(b,t,taille_tete)
        sortie= poids @ V
        return sortie

if __name__=="__main__":
    modele=Bigramme(120)
    print(modele)

    idx=torch.randint(0,120,(3,8))
    cible=torch.randint(0,120,(3,8))
    logits,perte=modele(idx,cible)
    print(perte.item())
    #print(logits.shape)
    #print(logits[0])






