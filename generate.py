"""
Génération de mot  avec le modèle MiniGPT entrainé, charger les poids du modèle save par train.py
partir d'un saut de ligne id 0 , ajoute nb_nouveaux caractère tirés au sort un par un puis decoder les ids et afficher le texte

"""
from pathlib import Path

import torch
import torch.nn.functional as F

from model import MiniGPT
from tokenizer import charger_texte, construire_vocabulaire, decode

nb_nouveaux=1000 # nombre de nouveaux caractères à générer
temperature=1.5 # température pour le tirage au sort, 1.0 = pas de changement, <1.0 = plus conservatif, >1.0 = plus aléatoire
top_k=10 # top k pour le tirage au sort, 10 = ne garder que les 10 caractères les plus probables pour tirer au sort
longueur=256 # nombre max de lettre données au modèle pour prédire la suivante
embd_dim=384
nb_tetes=6
nb_blocs=6
dropout=0.2


@torch.no_grad()
def generer(modele,idx_texte,nb_nouveaux,temperature,top_k,longueur):
    """
paramètres :
modele: le modèle qui donne les scores (mini GPT)
idx_texte: les ids du texte de départ, forme (1, t)
nb_nouveaux: le nombre de caractères à ajouter
temperature: divise les scores avant softmax (<1 plus prudent, >1 plus aléatoire)
top_k: on ne tire qu'au sein des top_k caractères les plus probables
longueur: nombre max de caractères donnés au modèle (taille de sa table de positions)

return  :les ids du texte complet forme (1, t + nb_nouveaux)

description :

A chaque tour (t = longueur actuelle du texte, 120 = taille du vocabulaire)
idx_contexte (1, min(t, longueur)): les longueur derniers ids au plus ce que reçoit le modèle
logits(1, min(t, longueur), 120) : 120 scores pour chaque caractère reçu un par candidat
last_score (1, 120): les 120 scores du dernier caractère
proba(1, 120):scores changés en probabilités (entre 0 et 1 somme = 1)
id_tire(1, 1): l'id du caractère tiré au sort (multinomial)
idx_texte  (1, t + 1): le texte allongé de id_tire
t= grandit à chaque tour de boucle (t+=1)
 """

# boucle pour tirer un caractère à la fois et l'ajouter au texte
    for _ in range(nb_nouveaux): # boucle de génération
        idx_contexte=idx_texte[:, -longueur:]  # garder les derniers caractères (longueur) pour prédire le suivant
        logits,_ =modele(idx_contexte) # calcul des logits

        last_score=logits[:,-1,:]/temperature # les scores du dernier caractère divisé par la température
        v,_=torch.topk(last_score,top_k)
        seuil = v[:, [-1]] # le score du dernier des 10
        last_score[last_score < seuil] = float("-inf") # les scores des caractères hors top k sont mis à -inf pour que softmax leur donne 0 %
        proba=F.softmax(last_score,dim=-1) # probabilités pour chaque caractère
        id_tire=torch.multinomial(proba,num_samples=1) # tirer un id au hasard selon les probabilités
        idx_texte=torch.cat((idx_texte,id_tire), dim=1) # concaténer le nouvel id au texte existant
    return idx_texte

def charger_modele(chemin_modele,taille_vocabulaire,embd_dim,nb_tetes,nb_blocs,longueur,dropout,device):
    """
    paramètres :
    chemin_modele: le fichier .pt qui contient les poids sauvegardés par train.py
    taille_vocabulaire, embd_dim, nb_tetes, nb_blocs, longueur, dropout: les réglages du modèle,
    les mêmes que ceux de l'entraînement, sinon load_state_dict refuse les poids
    device: l'appareil où placer le modèle (cpu, mps ou cuda)

    return : le modèle MiniGPT avec ses poids chargés, sur device, en mode eval (pas de dropout)
    """
    modele=MiniGPT(taille_vocabulaire,embd_dim,nb_tetes,nb_blocs,longueur,dropout)
    # charger le modèle
    modele.load_state_dict(torch.load(chemin_modele,map_location=torch.device("cpu")))
    modele.to(device)
    modele.eval() # mode evaluation, pas de dropout
    return modele

def main():
    """
    fonction principale pour générer du texte avec le modèle MiniGPT
    """
    torch.manual_seed(42) # pour reproductibilité
    if torch.cuda.is_available():
        device=torch.device("cuda")
    elif torch.backends.mps.is_available():
        device=torch.device("mps")
    else:
        device=torch.device("cpu")

    # chemin data et model pt
    chemin=Path(__file__).parent/"data"/"verne_complet.txt"
    chemin_modele=Path(__file__).parent/"model"/"MiniGPT_10M.pt"
    texte=charger_texte(chemin)
    # construction
    _,id_to_char,caractere=construire_vocabulaire(texte)
    modele=charger_modele(chemin_modele,len(caractere),embd_dim,nb_tetes,nb_blocs,longueur,dropout,device)
    # texte de départ un saut de ligne id 0 forme (1,1)
    idx_depart=torch.zeros((1,1),device=device,dtype=torch.long)
    resultat=generer(modele,idx_depart,nb_nouveaux,temperature,top_k,longueur)
    print(f"résultat:{resultat}")
    print(f"taille de resultat avant conversion en liste:{resultat.shape}")
    print("#"*50)
    resultat_list=resultat[0].tolist()
    print(resultat_list)
    print("#"*50)
    texte_generer=decode(resultat_list,id_to_char) #decode les ids en texte
    print(f"texte généré: {texte_generer}")
    # sauvegarde du texte généré dans un fichier
    chemin_texte_genere=Path(__file__).parent/"exemples"/"texte_genere_MiniGPT_10M_T_1.5_top_k_10.txt"

    nb_parametres=sum(p.numel() for p in modele.parameters()) # 10 831 224
    with open(chemin_texte_genere,"w",encoding="utf-8") as f:
        f.write(f"Modèle: MiniGPT,{nb_blocs} blocs, {nb_tetes} têtes, embd_dim {embd_dim}, temperature {temperature}\n")
        f.write(f"Contexte: {longueur} caractères | paramètres: {nb_parametres}\n")
        f.write("Perte test: 1.245\n")
        f.write(f"Graine: 42 | caractères générés: {nb_nouveaux}\n")
        f.write("\n")
        f.write(texte_generer)


if __name__=="__main__":
    main()
