import time
from pathlib import Path

import torch

from dataset import creer_batch
from model import MiniGPT
from tokenizer import charger_texte, construire_vocabulaire, encode

# fixer la graine aléatoire pour reproductibilité
torch.manual_seed(42)
# hyperparamètres
nb_pas=5000
#taille_batch=32
taille_batch=64

#longueur=32
longueur=256
#lr=0.01
lr=1e-3
intervalle_eval=500
nb_batchs_eval=200
dropout=0.2
embd_dim=384
nb_tetes=6
nb_blocs=6


# device
if torch.cuda.is_available():
    device=torch.device("cuda")
elif torch.backends.mps.is_available():
    device=torch.device("mps")
else:
    device=torch.device("cpu")
print(f"device: {device}")
#chemin du fichier texte
chemin=Path(__file__).parent/"data"/"verne_complet.txt"
#chemin_modele=Path(__file__).parent/"model"/"Modele_grand_saut.pt"
chemin_modele=Path(__file__).parent/"model"/"MiniGPT_10M.pt"
#chemin_modele=Path(__file__).parent/"model"/"Bigramme.pt"
texte=charger_texte(chemin)
# construction du vocabulaire
char_to_id,id_to_char,caractere=construire_vocabulaire(texte)
token_id=torch.tensor(encode(texte,char_to_id))
n=len(token_id)
fin_train=int(n*0.9)
fin_val=int(n*0.95)
# split du dataset en train et validation et test

donnees_train=token_id[:fin_train]
donnees_val=token_id[fin_train:fin_val]
donnees_test=token_id[fin_val:]
print(f"taille du train :{len(donnees_train)}| taille val :{len(donnees_val)}| taille test :{len(donnees_test)}")
print(len(donnees_train)+len(donnees_val)+len(donnees_test)==n)

#model
#modele=Bigramme(len(caractere))
#modele= MiniGPT(len(caractere), 32, 4) # 120 lettres , 32 nbres par lettre er 4 têtes
#modele= MiniGPT(len(caractere), 32, 4,3,longueur,dropout) # 3 blocs
modele= MiniGPT(len(caractere), embd_dim, nb_tetes, nb_blocs,longueur,dropout) # 6 blocs, 384 nombres par lettre, 6 têtes
modele.to(device)# deplacer le modèle sur le device
# decorateur pour éviter de suivre le gradient
@torch.no_grad()
# fonction pour estimer la perte sur un certain nombre de batchs
def estimer_perte(modele,donnees,nb_batchs):
    modele.eval() # mode evaluation, pas de dropout
    perte_total=[]

    for _ in range(nb_batchs):
        x_estimer_perte,y_estimer_perte=creer_batch(donnees,taille_batch,longueur)
        # deplacer les batchs sur le device
        x_estimer_perte,y_estimer_perte=x_estimer_perte.to(device),y_estimer_perte.to(device)
        _,loss_estimer_perte=modele(x_estimer_perte,y_estimer_perte)
        perte_total.append(loss_estimer_perte)
    moyenne_perte=(sum(perte_total))/len(perte_total)
    modele.train() # mode entrainement, dropout activé

    return moyenne_perte.item()

# optimiseur
optimiseur=torch.optim.AdamW(modele.parameters(),lr=lr)
print(f"corpus :{len(texte)} caractères | vocabulaire :{len(caractere)} caractères uniques")
#print(f"train :{len(donnees_train)}| val :{len(donnees_val)}")
print(f"paramètres du modèle:{sum(p.numel() for p in modele.parameters())}")

# entraînement
debut=time.time()
for pas in range(nb_pas+1):

    # selection du batch
    x,y=creer_batch(donnees_train,taille_batch,longueur)
    x,y=x.to(device),y.to(device)
    # reset du gradient
    optimiseur.zero_grad()
    #calcul de la perte
    _,loss_train=modele(x,y)
    loss_train.backward()
    optimiseur.step()

    if pas%intervalle_eval==0:
        perte_train=estimer_perte(modele, donnees_train, nb_batchs_eval)
        perte_val=estimer_perte(modele, donnees_val, nb_batchs_eval)

        print(f"pas:{pas:4d} | loss train :{perte_train:.3f}| loss val: {perte_val:.3f} ")

        print(f"temps ecoule depuis le debut: {time.time()-debut:.2f} secondes")
# evaluation finale sur le test set
perte_test=estimer_perte(modele, donnees_test, nb_batchs_eval)
print(f"perte finale sur le test set :{perte_test:.3f}")

torch.save(modele.state_dict(),chemin_modele)






