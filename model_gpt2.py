import torch
import torch.nn.functional as F
from torch import nn

"""
(b,t,v) (4,64,50257)
b= extraits du batch
t= tokens par extrait (au plus longueur=1024)
v= nombre de scores par token (taille du vocabulaire GPT-2)
embd_dim= nombres qui décrivent chaque token (768)
logit = score pour chaque token candidat
"""

class Tete(nn.Module):
    """tete d'attention causale : chaque token regarde les tokens d'avant et lui-même
    jamais ceux d'après et en retire une nouvelle description.

    plus utilisée depuis la 4b : MultiTetes calcule les 12 têtes en un seul appel.
    gardée comme référence : l'attention écrite à la main, ligne par ligne, est dans le forward
    (entre les triples guillemets), F.scaled_dot_product_attention fait la même chose en un appel.

    paramètres:
        embd_dim:le nombre de nombres qui décrivent chaque token à l'entrée (768)
        taille_tete:le nombre de nombres de chaque requête, clé et valeur (64)
        dropout: taux d'oubli pendant l'entraînement, donné à l'attention (dropout_p)

        Q = requête : ce que le token cherche dans son passé ( ce que je cherche)
        K = clé: ce que le token est, pour être trouvé (qui je suis )
        V = valeur : ce que le token transmet à ceux qui le regardent (ce que je donne quand on me regarde)

    formes:
        entrée x: (b, t, embd_dim),b = extraits du batch, t = tokens par extrait
        scores, poids: (b, t, t)ligne i = combien le token i regarde chaque token
        sortie: (b, t, taille_tete)  chaque token, enrichi de son passé
    """

    def __init__(self, embd_dim,taille_tete,dropout):
        super().__init__()
        # embd_dim nombres par token -> taille_tete nombres
        self.Q=nn.Linear(embd_dim,taille_tete,bias=False)    # requete
        self.K=nn.Linear(embd_dim,taille_tete,bias=False)    # clé
        self.V=nn.Linear(embd_dim,taille_tete,bias=False)    # valeur
        self.taille_tete=taille_tete
        self.dropout=nn.Dropout(dropout) # son taux self.dropout.p est donné à l'attention
    def forward(self,x):
        # (b, t, embd_dim) -> (b, t, taille_tete)
        Q=self.Q(x)
        K=self.K(x)
        V=self.V(x)
        # entraînement (modele.train()) : oubli au hasard pendant l'attention
        if self.training:
            taux=self.dropout.p
            sortie=F.scaled_dot_product_attention(Q,K,V,is_causal=True,dropout_p=taux)
        # mesure (modele.eval()) : pas d'oubli
        else:
            sortie=F.scaled_dot_product_attention(Q,K,V,is_causal=True)

        """
        taille_tete=self.taille_tete

       # notes pour chaque paire
        scores= Q @ K.transpose(-2,-1)
        # divise par la racine de taille_tete pour éviter un softmax trop grand
        scores=scores *taille_tete**-0.5

        # nombre de tokens de l'extrait
        _,t,_=x.shape
        # triangle : 1 = token autorisé (passé et soi-même), 0 = token d'après
        triangle_matrice=torch.tril(torch.ones(t,t,device=x.device))

        # masque, -inf tokens d'après pour que softmax leur donne 0 %
        scores=scores.masked_fill(triangle_matrice==0,float("-inf"))
        # notes -> parts, chaque ligne fait 100 %
        poids=F.softmax(scores,dim=-1)
        poids=self.dropout(poids)

        # chaque token récupère les valeurs de son passé, selon ses parts;(b,t,taille_tete)
        sortie= poids @ V
        """
        #sortie=F.scaled_dot_product_attention(Q,K,V,is_causal=True)

        return sortie


class MiniGPT(nn.Module):
    """
Modèle de langage à tokens GPT-2 (taille GPT-2 small) : prédit le token suivant à chaque position

    paramètres:
        taille_vocabulaire: nombre de tokens différents (50257)
        embd_dim: nombres qui décrivent chaque token (768)
        nb_tetes: têtes d'attention dans chaque bloc (12)
        nb_blocs: blocs Transformer empilés (12)
        longueur: longueur maximale de tokens donnés au modèle pour prédire le suivant (1024)
        dropout: taux de dropout pour la régularisation (0.0 : chaque token n'est lu qu'une fois)
    table partagée:
        embedding et sortie utilisent la même table (50257, 768) : 124 M de paramètres au lieu de 163 M
    initialisation:
        poids des Linear et des Embedding : moyenne 0, écart-type 0.02 ; biais à 0
        modèle neuf : il hésite entre les 50257 tokens, perte = 10.89 (ln(50257) = 10.82)
    chemin des données:
        idx (b, t) -> embedding (b, t, embd_dim)+ position(t, embd_dim) -> nb_blocs blocs (b, t, embd_dim)
        ->LayerNorm finale -> sortie (b, t, taille_vocabulaire)
    attention:
        dans chaque bloc, MultiTetes fait les 12 têtes en un seul appel à F.scaled_dot_product_attention
    calcul en bf16:
        pas dans ce fichier : dans la boucle d'entraînement, with torch.autocast(...) autour de modele(x,y)
        Linear et attention en bf16, softmax, LayerNorm et cross_entropy restent en float32

    renvoie:
        logits: 50257 scores par token, un par token candidat
        perte: entropie croisée si cible est donnée, sinon None

"""
    def __init__(self, taille_vocabulaire,embd_dim,nb_tetes,nb_blocs,longueur,dropout):
        super().__init__()
        self.embedding=nn.Embedding(taille_vocabulaire,embd_dim) # table (50257, 768) : une ligne de 768 nombres par token
        self.position_embedding=nn.Embedding(longueur,embd_dim) # position embedding : chaque position a un vecteur de nombres qui lui est propre
        #(bloc=attention + feed forward)
        # [Bloc(embd_dim,nb_tetes)= creer un bloc, for _ in range(nb_blocs)=crrer autant de bloc de nb_bloc, 3-> [bloc0,... bloc2] , * deballer->(nn.Sequential(bloc0,...bloc2))
        self.blocs=nn.Sequential(*[Bloc(embd_dim,nb_tetes,dropout) for _ in range(nb_blocs)])
        self.ln_finale=nn.LayerNorm(embd_dim) # normalise le vecteur final de chaque token avant la sortie
        self.sortie=nn.Linear(embd_dim,taille_vocabulaire) # 768 -> 50257 : un score par token candidat, weight (50257, 768)
        # table partagée : weight de la sortie = table de l'embedding, même forme (50257, 768)
        # la sortie compare le vecteur final à chaque ligne de la table : 124 M de paramètres au lieu de 163 M
        self.sortie.weight=self.embedding.weight

        # initialisation : self.modules() donne chaque couche du modèle, une par tour
        # (embedding, position_embedding, les couches des 12 blocs, ln_finale, sortie)
        for module in self.modules():
            # isinstance : est-ce que cette couche est une Linear ?
            if isinstance(module,nn.Linear):
                # remplit weight sur place (le _ final) : moyenne 0, écart-type 0.02
                nn.init.normal_(module.weight,mean=0.0,std=0.02)
                # Q, K, V sont créées avec bias=False : elles n'ont pas de biais à régler
                if module.bias is not None:
                    nn.init.zeros_(module.bias) # biais à 0, sur place (un = ne remplirait pas la table)
            # tables des tokens et des positions
            elif isinstance(module,nn.Embedding):
                nn.init.normal_(module.weight,mean=0.0,std=0.02)
        # la table partagée est réglée deux fois (embedding puis sortie) : même réglage, aucun souci
        # les LayerNorm ne sont pas touchées : elles gardent poids 1 et biais 0


    def forward(self,idx,cible=None):
        # idx (b, t) -> embedding (b, t, embd_dim)
        _,t=idx.shape # t=taille de l'extrait
        positions=torch.arange(t,device=idx.device) # vecteur de positions (0,1,...,t-1)
        x=self.embedding(idx) # vecteur de 768 nombres par token, (b, t, 768)
        x=x+self.position_embedding(positions) # ajout du vecteur de position à chaque token, (t, 768) ajouté à chaque extrait
        x=self.blocs(x) # les 12 blocs l'un après l'autre, (b, t, 768) -> (b, t, 768)
        x=self.ln_finale(x) # vecteur final de chaque token
        logits=self.sortie(x) # (b, t, 768) -> (b, t, 50257)

        # génération : pas de cible, donc pas de perte
        if cible is None:
            return logits, None

        _,_,v=logits.shape # v = 50257
        logits_plat=logits.reshape(-1,v) # (b, t, v) -> (b*t, v) : cross_entropy veut une ligne par prédiction
        cible_plat=cible.reshape(-1) # (b, t) -> (b*t) : le bon token de chaque prédiction

        # moyenne des erreurs sur les b*t prédictions (reste en float32 sous autocast)
        perte_minigpt=F.cross_entropy(logits_plat,cible_plat)

        return logits,perte_minigpt

class MultiTetes(nn.Module):
    """
Les têtes d'attention d'un bloc, calculées toutes ensemble, en un seul appel

paramètres :
    embd_dim: nombres par token à l'entrée (768)
    nb_tete:nombre de têtes (12)
    taille_tete: nombres renvoyés par chaque tête pour chaque token (768 // 12 = 64)
    dropout: taux d'oubli, utilisé dans l'attention (dropout_p) et après la projection

formes :
    entrée x : (b, t, embd_dim)
    q, k, v rangés par tête : (b, nb_tete, t, taille_tete)
    sortie: (b, t, embd_dim)

MultiTetes(768, 12, 64, 0.0) - 12 têtes de 64 - 768 nombres par token
Q, K, V : une Linear(768, 768) chacune = les 12 Linear(768, 64) des têtes, empilées

"""
    def __init__(self,embd_dim,nb_tete,taille_tete,dropout):
        super().__init__()
        self.projection=nn.Linear(nb_tete*taille_tete,embd_dim) # 12 têtes de 64 nombres = 768 nombres par token -> 768
        self.dropout=nn.Dropout(dropout) # son taux self.dropout.p sert aussi à l'attention

        self.Q=nn.Linear(embd_dim,nb_tete*taille_tete,bias=False) # requête des 12 têtes d'un coup : 768 -> 768 (12 x 64)
        self.K=nn.Linear(embd_dim,nb_tete*taille_tete,bias=False) # clé
        self.V=nn.Linear(embd_dim,nb_tete*taille_tete,bias=False) # valeur
        # rangés dans self : le forward en a besoin pour couper en têtes
        self.nb_tete=nb_tete
        self.taille_tete=taille_tete
    def forward(self,x):
        nb_extrait,nb_token,_=x.shape # x (b, t, 768) : on garde b et t, le _ ignore 768
        # calcul : (b, t, 768) -> (b, t, 768)
        q=self.Q(x)
        k=self.K(x)
        v=self.V(x)

        # rangement : couper 768 en 12 x 64 (b, t, 12, 64), puis ranger par tête (b, 12, t, 64)
        q=q.reshape(nb_extrait,nb_token,self.nb_tete,self.taille_tete).transpose(1,2)
        k=k.reshape(nb_extrait,nb_token,self.nb_tete,self.taille_tete).transpose(1,2)
        v=v.reshape(nb_extrait,nb_token,self.nb_tete,self.taille_tete).transpose(1,2)
        # F.scaled_dot_product_attention : scores q@k, divisés par racine(taille_tete), masque causal (is_causal=True),
        # softmax, dropout, poids@v ; les 12 têtes en un appel, (b, 12, t, 64) -> (b, 12, t, 64)
        # entraînement (modele.train()) : oubli au hasard d'une part taux des poids d'attention
        if self.training:
            taux=self.dropout.p # p = le taux rangé dans nn.Dropout(dropout)
            sortie=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=taux)
        # mesure (modele.eval()) : pas d'oubli
        else:
            sortie=F.scaled_dot_product_attention(q,k,v,is_causal=True)


        # rangement retour : par token (b, t, 12, 64), puis recoller les 12 têtes côte à côte (b, t, 768)
        sorties_tetes=sortie.transpose(1,2).reshape(nb_extrait,nb_token,self.nb_tete*self.taille_tete)
        sorties_tetes=self.projection(sorties_tetes) # calcul : mélange les 12 têtes, 768 -> 768
        sorties_tetes=self.dropout(sorties_tetes)
        return sorties_tetes
class FeedForward(nn.Module):
    """
    Feed-forward :chaque token réfléchit seul à ce qu'il a reçu de l'attention.

    paramètres:
        embd_dim: nombres par token (768)
        dropout: taux d'oubli après la deuxième Linear
    calcul:
        Linear(embd_dim, 4 * embd_dim) -> ReLU -> Linear(4 * embd_dim, embd_dim)

    formes:
        entrée x: (b, t, embd_dim)
        sortie: (b, t, embd_dim)
"""
    def __init__(self, embd_dim,dropout):
        super().__init__()

        self.feed_forward=nn.Sequential(
            nn.Linear(embd_dim,4*embd_dim), # 768 -> 3072 (768*4)
            nn.ReLU(),
            nn.Linear(4*embd_dim,embd_dim), # 3072 -> 768
            nn.Dropout(dropout)
        )


    def forward(self,x):
        x=self.feed_forward(x)
        return x

class Bloc(nn.Module):
    """
    Bloc Transformer : écouter le passé (têtes), puis réfléchir (feed-forward).

    paramètres:
        embd_dim: nombres par token en entrée comme en sortie (768)
        nb_tetes: nombre de têtes : chacune fait embd_dim // nb_tetes nombres (768 // 12 = 64)
        dropout: taux d'oubli, transmis à MultiTetes et FeedForward

    calcul:
        x = x + têtes(LayerNorm(x)),  résiduel : on ajoute au lieu de remplacer
        x = x + feed_forward(LayerNorm(x))

    formes:
        entrée x: (b, t, embd_dim)
        sortie: (b, t, embd_dim)
    """
    def __init__(self, embd_dim,nb_tetes,dropout):
        super().__init__()
        # normaliser les 768 nombres de chaque token (moyenne=0, ecart type=1)
        self.ln1=nn.LayerNorm(embd_dim)
        self.tetes=MultiTetes(embd_dim,nb_tetes,embd_dim// nb_tetes,dropout) # les 12 têtes du bloc (un seul MultiTetes)
        # normaliser les 768 nombres de chaque token (moyenne=0, ecart type=1)
        self.ln2=nn.LayerNorm(embd_dim)
        self.ff=FeedForward(embd_dim,dropout)
    def forward(self,x):
        x=x+self.tetes(self.ln1(x)) #pour eviter que chaque bloc remplace entièrement les 768 nbres de chq token par sa transformation
        x=x+self.ff(self.ln2(x))# ajout du residuel,l'info de depart est conservée juste ajout de la correction
        return x











