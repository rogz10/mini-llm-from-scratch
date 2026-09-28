"""
Génération du texte avec le modèle Bigramme entrainé, charger les poids du modèle save par train.py
partir d'un saut de ligne id 0 , ajoute nb_nouveaux caractère tirés au sort un par un puis decoder les ids et afficher le texte

"""
from pathlib import Path

import torch
import torch.nn.functional as F

from model import Bigramme
from tokenizer import charger_texte, construire_vocabulaire, decode

nb_nouveaux=200
# chemin data et model pt
chemin=Path(__file__).parent/"data"/"train.txt"
chemin_modele=Path(__file__).parent/"model"/"bigramme.pt"
texte=charger_texte(chemin)
# construction
char_to_id,id_to_char,caractere=construire_vocabulaire(texte)
modele=Bigramme(len(caractere))
# charger le modèle
modele.load_state_dict(torch.load(chemin_modele))
# texte de départ un saut de ligne id 0 forme (1,1)
idx_depart=torch.zeros((1,1),dtype=torch.long)

@torch.no_grad()
def generer(modele,idx_texte,nb_nouveaux):
    """
paramètres :
modele: le modèle qui donne les scores (Bigramme actu)
idx_texte: les ids du texte de départ, forme (1, t)
nb_nouveaux : le nombre de caractères à ajouter

return  :les ids du texte complet forme (1, t + nb_nouveaux)

A chaque tour (t = longueur actuelle du texte, 120 = taille du vocabulaire)
logits(1, t, 120) : 120 scores pour chaque caractère d'entrée un par candidat
last_score (1, 120): les 120 scores du dernier caractère
proba(1, 120):scores changés en probabilités (entre 0 et 1 somme = 1)
id_tire(1, 1): l'id du caractère tiré au sort (multinomial)
idx_texte  (1, t + 1): le texte allongé de id_tire
t= grandit à chaque tour de boucle (t+=1)
 """

    for _ in range(nb_nouveaux):
        logits,_ =modele(idx_texte)
        last_score=logits[:,-1,:]
        proba=F.softmax(last_score,dim=-1)
        id_tire=torch.multinomial(proba,num_samples=1)
        idx_texte=torch.cat((idx_texte,id_tire), dim=1)
    return idx_texte

resultat=generer(modele,idx_depart,nb_nouveaux)

print(f"résultat:{resultat}")
print(f"taille de resulat avant conversion en liste:{resultat.shape}")
resultat_list=resultat[0].tolist()
print(resultat_list)
texte_generer=decode(resultat_list,id_to_char)
print(f"texte généré: {texte_generer}")



