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
        embd_dim:le nombre de nombres qui décrivent chaque lettre à l'entrée (32)
        taille_tete:le nombre de nombres de chaque requête, clé et valeur (16 ou 32)

        Q = requête : ce que la lettre cherche dans son passé ( ce que je cherche)
        K = clé: ce que la lettre est, pour être trouvée (qui je suis )
        V = valeur : ce que la lettre transmet à ceux qui la regardent (ce que je donne quand on me regarde)

    formes:
        entrée x: (b, t, embd_dim),b = extraits du batch, t = lettres par extrait
        scores, poids: (b, t, t)ligne i = combien la lettre i regarde chaque lettre
        sortie: (b, t, taille_tete)  chaque lettre, enrichie de son passé
    """

    def __init__(self, embd_dim,taille_tete):
        super().__init__()
        # embd_dim nombres par lettre , taille_tete nombres
        self.Q=nn.Linear(embd_dim,taille_tete,bias=False)    # requete
        self.K=nn.Linear(embd_dim,taille_tete,bias=False)    # clé
        self.V=nn.Linear(embd_dim,taille_tete,bias=False)    # valeur
        self.taille_tete=taille_tete

    def forward(self,x):
        # (b, t, embd_dim) -> (b, t, taille_tete)
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
        # triangle : 1 = lettre autorisée (passé et soi-même), 0 = lettre d'après
        triangle_matrice=torch.tril(torch.ones(t,t))

        # masque, -inf lettres d'après pour que softmax leur donne 0 %
        scores=scores.masked_fill(triangle_matrice==0,float("-inf"))
        # notes -> parts, chaque ligne fait 100 %
        poids=F.softmax(scores,dim=-1)

        # chaque lettre récupère les valeurs de son passé, selon ses parts;(b,t,taille_tete)
        sortie= poids @ V
        return sortie


class MiniGPT(nn.Module):
    """
Modèle de langage caractère : prédit le caractère suivant à chaque position

    paramètres:
        taille_vocabulaire: nombre de caractères différents (120)
        embd_dim: nombres qui décrivent chaque caractère (32)
        nb_tetes: têtes d'attention dans chaque bloc (4)
        nb_blocs: blocs Transformer empilés (3)

    chemin des données:
        idx (b, t) -> embedding (b, t, embd_dim) -> nb_blocs blocs (b, t, embd_dim)
        ->LayerNorm finale -> sortie (b, t, taille_vocabulaire)

    renvoie:
        logits: 120 scores par caractère, un par caractère candidat
        perte: entropie croisée si cible est donnée, sinon None

"""
    def __init__(self, taille_vocabulaire,embd_dim,nb_tetes,nb_blocs):
        super().__init__()
        self.embedding=nn.Embedding(taille_vocabulaire,embd_dim)
         # embd_dim dimension embedding(nombre de nombre par lettres), 4 têtes et embd_dim//4(taille de chaque têtes=8)
        #self.tete=MultiTetes(embd_dim,nb_tetes,embd_dim//nb_tetes)
        # feed
        #self.fw=FeedForward(embd_dim)
        #(bloc=attention + feed forward)
        # [Bloc(embd_dim,nb_tetes)= creer un bloc, for _ in range(nb_blocs)=crrer autant de bloc de nb_bloc, 3-> [bloc0,... bloc2] , * deballer->(nn.Sequential(bloc0,...bloc2))
        self.blocs=nn.Sequential(*[Bloc(embd_dim,nb_tetes) for _ in range(nb_blocs)])
        self.ln_finale=nn.LayerNorm(embd_dim)
        self.sortie=nn.Linear(embd_dim,taille_vocabulaire)
        #self.muti_tetes=MultiTetes(embd_dim,4,embd_dim//4)

    def forward(self,idx,cible=None):
        x=self.embedding(idx)
        #x=self.tete(x)
        #x=self.fw(x)
        x=self.blocs(x)
        x=self.ln_finale(x)
        logits=self.sortie(x)

        if cible is None:
            return logits, None

        _,_,v=logits.shape
        logits_plat=logits.reshape(-1,v)
        cible_plat=cible.reshape(-1)

        perte_minigpt=F.cross_entropy(logits_plat,cible_plat)

        return logits,perte_minigpt

class MultiTetes(nn.Module):
    """
Plusieurs têtes d'attention en parallèle

paramètres :
    embd_dim: nombres par lettre à l'entrée
    nb_tete:nombre de têtes
    taille_tete: nombres renvoyés par chaque tête pour chaque lettre

formes :
    entrée x : (b, t, embd_dim)
    sortie: (b, t, nb_tete x taille_tete)

MultiTetes(32, 4, 8) - 4 têtes de 8 - 32 nombres par lettre

"""
    def __init__(self,embd_dim,nb_tete,taille_tete):
        super().__init__()
        self.tetes=nn.ModuleList([Tete(embd_dim,taille_tete) for _ in range(nb_tete)])
    def forward(self,x):
        sorties_tetes=[tete(x) for tete in self.tetes]
        sorties_tetes=torch.cat(sorties_tetes,dim=-1)
        return sorties_tetes
class FeedForward(nn.Module):
    """
    Feed-forward :chaque caractère réfléchit seul à ce qu'il a reçu de l'attention.

    paramètres:
        embd_dim: nombres par caractère
    calcul:
        Linear(embd_dim, 4 * embd_dim) -> ReLU -> Linear(4 * embd_dim, embd_dim)

    formes:
        entrée x: (b, t, embd_dim)
        sortie: (b, t, embd_dim)
"""
    def __init__(self, embd_dim):
        super().__init__()

        self.feed_forward=nn.Sequential(
            nn.Linear(embd_dim,4*embd_dim), # 32 -> 128 (32*4)
            nn.ReLU(),
            nn.Linear(4*embd_dim,embd_dim) # 128 -> 32
        )

    def forward(self,x):
        x=self.feed_forward(x)
        return x

class Bloc(nn.Module):
    """
    Bloc Transformer : écouter le passé (têtes), puis réfléchir (feed-forward).

    paramètres:
        embd_dim: nombres par caractère en entrée comme en sortie
        nb_tetes: nombre de têtes : chacune fait embd_dim // nb_tetes nombres

    calcul:
        x = x + têtes(LayerNorm(x)),  résiduel : on ajoute au lieu de remplacer
        x = x + feed_forward(LayerNorm(x))

    formes:
        entrée x: (b, t, embd_dim)
        sortie: (b, t, embd_dim)
    """
    def __init__(self, embd_dim,nb_tetes):
        super().__init__()
        # normaliser les embed nombre de chaque lettre (moyenne=0, ecart type=1)
        self.ln1=nn.LayerNorm(embd_dim)
        self.tetes=MultiTetes(embd_dim,nb_tetes,embd_dim// nb_tetes)
        # normaliser les embed nombre de chaque lettre (moyenne=0, ecart type=1)
        self.ln2=nn.LayerNorm(embd_dim)
        self.ff=FeedForward(embd_dim)
    def forward(self,x):
        x=x+self.tetes(self.ln1(x)) #pour eviter que chaque bloc remplace entièrement les 32 nbres de chq lettre par sa transformation
        x=x+self.ff(self.ln2(x))# ajout du residuel,l'info de depart est conservée juste ajout de la correction
        return x






if __name__=="__main__":
    modele=Bigramme(120)
    print(modele)

    idx=torch.randint(0,120,(3,8))
    cible=torch.randint(0,120,(3,8))
    logits,perte=modele(idx,cible)
    print(perte.item())
    print("#"*50)

    mt=MultiTetes(32,4,8)
    print(mt(torch.randn(4, 5, 32)).shape)
    print(sum(p.numel() for p in mt.parameters()))
    print(MultiTetes(32, 2, 16)(torch.randn(4, 5, 32)).shape)

    #print(logits.shape)
    #print(logits[0])
    #print(MiniGPT .__doc__)





