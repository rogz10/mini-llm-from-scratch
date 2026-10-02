"""
Génération du texte avec le modèle MiniGPT entrainé, charger les poids du modèle save par train.py
partir d'un saut de ligne id 0 , ajoute nb_nouveaux caractère tirés au sort un par un puis decoder les ids et afficher le texte

"""
from pathlib import Path

import torch
import torch.nn.functional as F

from model import MiniGPT
from tokenizer import charger_texte, construire_vocabulaire, decode

nb_nouveaux=200 # nombre de nouveaux caractères à générer
longueur=8 # nombre max de lettre données au modèle pour prédire la suivante
# chemin data et model pt
chemin=Path(__file__).parent/"data"/"train.txt"
#chemin_modele=Path(__file__).parent/"model"/"bigramme.pt"
chemin_modele=Path(__file__).parent/"model"/"Minigpt_avec_tete.pt"
texte=charger_texte(chemin)
# construction
char_to_id,id_to_char,caractere=construire_vocabulaire(texte)
print(f" taille caractère :{len(caractere)}")
#modele=Bigramme(len(caractere))
modele=MiniGPT(len(caractere),32)
# charger le modèle
modele.load_state_dict(torch.load(chemin_modele))
# texte de départ un saut de ligne id 0 forme (1,1)
idx_depart=torch.zeros((1,1),dtype=torch.long)

@torch.no_grad()
def generer(modele,idx_texte,nb_nouveaux):
    """
paramètres :
modele: le modèle qui donne les scores (mini GPT)
idx_texte: les ids du texte de départ, forme (1, t)
nb_nouveaux : le nombre de caractères à ajouter

return  :les ids du texte complet forme (1, t + nb_nouveaux)

description :

A chaque tour (t = longueur actuelle du texte, 120 = taille du vocabulaire)
logits(1, t, 120) : 120 scores pour chaque caractère d'entrée un par candidat
last_score (1, 120): les 120 scores du dernier caractère
proba(1, 120):scores changés en probabilités (entre 0 et 1 somme = 1)
id_tire(1, 1): l'id du caractère tiré au sort (multinomial)
idx_texte  (1, t + 1): le texte allongé de id_tire
t= grandit à chaque tour de boucle (t+=1)
 """

# boucle pour tirer un caractère à la fois et l'ajouter au texte
    for _ in range(nb_nouveaux): # boucle de génération
        idx_contexte=idx_texte[:,-longueur:] # ne garder que les derniers caractères pour la prédiction
        logits,_ =modele(idx_contexte) # calcul des logits

        last_score=logits[:,-1,:]
        proba=F.softmax(last_score,dim=-1) # probabilités pour chaque caractère
        id_tire=torch.multinomial(proba,num_samples=1) # tirer un id au hasard selon les probabilités
        idx_texte=torch.cat((idx_texte,id_tire), dim=1) # concaténer le nouvel id au texte existant
    #print(f"taille de logits :{logits.shape}")
    #print(f"taille de idx_texte :{idx_texte.shape}")
    return idx_texte


resultat=generer(modele,idx_depart,nb_nouveaux)

print(f"résultat:{resultat}")
print(f"taille de resultat avant conversion en liste:{resultat.shape}")
print("#"*50)
resultat_list=resultat[0].tolist()
print(resultat_list)
print("#"*50)
texte_generer=decode(resultat_list,id_to_char) #decode les ids en texte
print(f"texte généré: {texte_generer}")



