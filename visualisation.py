"""
Visualisation des embeddings du modèle MiniGPT 10M entraîné (model/MiniGPT_10M.pt)

les 120 caractères, projetés en 3D avec t-SNE :
    une fenêtre interactive s'ouvre (tourner, zoomer, cliquer sur un point)

lancer: python visualisation.py
"""
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

from model import MiniGPT
from tokenizer import charger_texte, construire_vocabulaire

#hyperparamètres du modèle MiniGPT 10M
embd_dim=384
nb_tetes=6
nb_blocs=6
longueur=256
dropout=0.2
# dossier du projet
dossier=Path(__file__).parent
chemin=dossier/"data"/"verne_complet.txt"
chemin_modele=dossier/"model"/"MiniGPT_10M.pt"

VOYELLES=set("aeiouyAEIOUYàâäéèêëîïôöùûüÀÂÉÈÊËÎÏÔÖÙÛÜœŒæÆ") # voyelles françaises, majuscules et minuscules, y compris les ligatures
COULEURS={
    "voyelle": "#d62728",
    "consonne": "#1f77b4",
    "chiffre": "#2ca02c",
    "ponctuation": "#7f7f7f",
    "espace": "#9467bd",
}

def categorie(caractere):
    """
    caractere: un caractère du vocabulaire
    renvoie: "voyelle", "consonne", "chiffre", "espace" ou "ponctuation"
    """
    if caractere in VOYELLES:
        return "voyelle"
    if caractere.isalpha():
        return "consonne"
    if caractere.isdigit():
        return "chiffre"
    if caractere.isspace():
        return "espace"
    return "ponctuation"


def etiquette(caractere):
    """rend visibles les caractères invisibles : ' ' -> '␣', '\\n' -> '\\n'"""
    return {" ": "␣", "\n": "\\n", "\t": "\\t"}.get(caractere, caractere)


def charger_modele():
    """
    renvoie:
        modele: MiniGPT 10M en mode évaluation, sur le processeur
        texte: le corpus (pour compter les caractères)
        char_to_id, id_to_char: le vocabulaire
    """
    texte=charger_texte(chemin) # chargement du texte
    char_to_id,id_to_char,caractere=construire_vocabulaire(texte) # construction du vocabulaire
    modele=MiniGPT(len(caractere),embd_dim,nb_tetes,nb_blocs,longueur,dropout)
    # map_location="cpu"
    modele.load_state_dict(torch.load(chemin_modele,map_location="cpu"))
    modele.eval()  # dropout éteint
    return modele,texte,char_to_id,id_to_char

def projeter_3d(poids):
    """
    poids: modele.embedding.weight, forme (120, 384)
    renvoie: coordonnées 3D, forme (120, 3)

    t-SNE garde les voisins proches ensemble : deux caractères proches en 384
    dimensions restent proches en 3D
    """
    donnees=poids.detach().cpu().numpy() # (120, 384)
    tsne=TSNE(n_components=3,perplexity=15,metric="cosine",init="pca",random_state=42) # config du t-SNE
    return tsne.fit_transform(donnees)


def plus_proches_voisins(poids,nb_voisins=5):
    """
    poids: (120, 384)
    renvoie: pour chaque caractère, les ids de ses nb_voisins plus proches (similarité cosinus)
    cosine similarity =(a@b) / norme(a) * norme(b)
    """
    normes=F.normalize(poids.detach(),dim=1)  # chaque ligne ramenée à une longueur de 1
    similarite=normes @ normes.T # (120, 120), 1 = même direction
    similarite.fill_diagonal_(-2)# un caractère n'est pas son propre voisin
    return similarite.topk(nb_voisins,dim=1).indices.tolist()


def embeddings_fenetre(points,id_to_char,frequences,voisins):
    """
    ouvre une fenêtre matplotlib interactive :
    tourner le cube molette / clic droit glissé,zoomer
        clic sur un point: infos du caractère, ses 5 plus proches voisins entourés en noir
    """
    figure=plt.figure(figsize=(11,9))
    axe=figure.add_subplot(projection="3d")
    ids_par_nuage={}  # pour retrouver quel caractère a été cliqué
    for nom,couleur in COULEURS.items():
        ids=[i for i in range(len(points)) if categorie(id_to_char[i])==nom]
        if not ids:
            continue
        nuage=axe.scatter(
            points[ids,0],points[ids,1],points[ids,2],
            color=couleur,s=[12+8*math.log10(1+frequences[id_to_char[i]]) for i in ids],
            alpha=0.7,label=f"{nom} ({len(ids)})",picker=True,pickradius=6,depthshade=False,
        )
        ids_par_nuage[nuage]=ids
        for i in ids:
            axe.text(*points[i],etiquette(id_to_char[i]),color=couleur,fontsize=9)
    axe.set_title(f"MiniGPT 10M : les {len(points)} caractères dans l'espace des embeddings (t-SNE 3D)")
    axe.legend(loc="upper left")

    # le cadre d'infos en bas à gauche et les cercles autour des voisins du point cliqué
    infos=figure.text(0.02,0.02,"clique sur un point",fontsize=11,family="monospace",
                      bbox={"facecolor":"white","edgecolor":"gray"})
    entoures=axe.scatter([],[],[],s=220,facecolors="none",edgecolors="black",linewidths=1.5,depthshade=False)

    def clic(evenement):
        ids=ids_par_nuage.get(evenement.artist)
        if ids is None:
            return
        i=ids[evenement.ind[0]]
        caractere=id_to_char[i]
        proches=voisins[i]
        infos.set_text(
            f"caractère : {etiquette(caractere)}   ({categorie(caractere)}, id {i})\n"
            +"apparitions dans le corpus : "+f"{frequences[caractere]:,}".replace(","," ")+"\n"
            +"5 plus proches voisins : "+"  ".join(etiquette(id_to_char[j]) for j in proches)
        )
        entoures._offsets3d=(points[proches,0],points[proches,1],points[proches,2])
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("pick_event",clic)
    plt.show()


if __name__=="__main__":
    modele,texte,char_to_id,id_to_char=charger_modele()

    poids_embedding=modele.embedding.weight
    points=projeter_3d(poids_embedding)
    voisins=plus_proches_voisins(poids_embedding)
    frequences=Counter(texte)
    embeddings_fenetre(points,id_to_char,frequences,voisins)
