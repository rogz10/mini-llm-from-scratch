from pathlib import Path

import torch
import torch.nn.functional as F

from generate import (
    charger_modele,
    dropout,
    embd_dim,
    longueur,
    nb_blocs,
    nb_tetes,
)
from tokenizer import charger_texte, construire_vocabulaire, encode
from tree import dessiner_arbre


def probabilites_suivantes(modele,prefixe,char_to_id,device):
    """
    prefixe: le texte de départ, par exemple "Phileas "
    return : les probabilités du caractère qui suit le préfixe, forme (120,)
    """
    encode_prefixe=encode(prefixe,char_to_id)
    idx=torch.tensor([encode_prefixe],dtype=torch.long,device=device)
    with torch.no_grad():
        logits,_=modele(idx)
        score_suivant=logits[0,-1,:]
        probabilites=F.softmax(score_suivant,dim=-1)
        return probabilites

def scorer_continuation(modele,prefixe,continuation,char_to_id,device):
    """
    Calcule la probabilité de la continuation donnée le préfixe"""
    score=1.0
    contexte=prefixe

    for caractere in continuation:
        probabilites=probabilites_suivantes(modele,contexte,char_to_id,device)
        identifiant=char_to_id[caractere]
        proba=probabilites[identifiant].item()

        score*=proba
        contexte+=caractere
    return score

def choisir_candidat(modele,prefixe,candidats,char_to_id,device):
    """
    Choisit le candidat le plus probable parmi une liste de candidats
    """
    meilleur_candidat=None
    meilleur_score=-1.0
    for candidat in candidats:

        score = scorer_continuation(modele, prefixe, candidat, char_to_id, device)
        print(f"Score de {candidat!r} : {score}")
        if score>meilleur_score:
            meilleur_score=score
            meilleur_candidat=candidat
    #print(f"meilleur candidat : {meilleur_candidat!r} avec score : {meilleur_score}")

    return meilleur_candidat,meilleur_score


def afficher_arbre( modele, prefixe, char_to_id, id_to_char, device,profondeur,noeuds, k=2, niveau=0,proba_chemin=1.0):
    if profondeur == 0:
        return

    probabilites = probabilites_suivantes(modele, prefixe, char_to_id, device)
    valeurs, indices = torch.topk(probabilites, k=k)

    for rang in range(len(valeurs)):
        identifiant = indices[rang].item()
        caractere_suivant = id_to_char[identifiant]
        nouveau_prefixe = prefixe + caractere_suivant
        proba = valeurs[rang].item()
        nouvelle_proba_chemin = proba_chemin * proba

        noeuds.append({
            "texte": nouveau_prefixe,
            "parent": prefixe,
            "caractere": caractere_suivant,
            "proba_locale": proba,
            "proba_chemin": nouvelle_proba_chemin,
            "niveau": niveau + 1,
        })

        #print("  " * niveau, repr(nouveau_prefixe), proba)
        indentation = "  " * niveau # indentation pour l'affichage
        print(f"{indentation}{nouveau_prefixe!r} lettre = {proba:3f} chemin={nouvelle_proba_chemin:.7f}") # le !r dans f-string = repr(nouveau_prefixe) pour afficher les caractères spéciaux


        afficher_arbre(modele, nouveau_prefixe, char_to_id, id_to_char, device,profondeur - 1,noeuds, k=k, niveau=niveau + 1,proba_chemin=nouvelle_proba_chemin)

def main():
    chemin=Path(__file__).parent/"data"/"verne_complet.txt"
    chemin_modele=Path(__file__).parent/"model"/"MiniGPT_10M.pt"
    prefixe="Phileas "
    noeuds = [{
        "texte": prefixe,
        "parent": None,
        "caractere": prefixe,
        "proba_locale": None,
        "proba_chemin": 1.0,
        "niveau": 0,
    }]

    texte=charger_texte(chemin)
    char_to_id,id_to_char,caractere=construire_vocabulaire(texte)
    #print(encode_prefixe.dtype)
    if torch.cuda.is_available():
        device=torch.device("cuda")
    elif torch.backends.mps.is_available():
        device=torch.device("mps")
    else:
        device=torch.device("cpu")
    print(f"device:{device}")

    modele=charger_modele(chemin_modele,len(caractere),embd_dim,nb_tetes,nb_blocs,longueur,dropout,device)
    print(repr(prefixe))

    afficher_arbre(modele,prefixe, char_to_id, id_to_char, device,profondeur=4,noeuds=noeuds, k=2)
    print("nombre de noeuds:",len(noeuds))
    print("#"*50)
    exemples = [
    {
        "prefixe": "Michel ",
        "candidats": ["Grant ", "Smith ", "Ardan "],
        "attendu": "Ardan ",
    },
    {
        "prefixe": "Cyrus ",
        "candidats": ["Ardan ", "Smith ", "Grant "],
        "attendu": "Smith ",
    },
    {
        "prefixe": "Harry ",
        "candidats": ["Grant ", "Ardan ", "Smith "],
        "attendu": "Grant ",
    },
]
    nb_correct = 0
    for exemple in exemples:
        choix, score_choisi = choisir_candidat(modele, exemple["prefixe"], exemple["candidats"], char_to_id, device)
        correct = choix == exemple["attendu"]
        print(f"préfixe: {exemple['prefixe']!r} | candidat choisi: {choix!r} | correct: {correct} | score: {score_choisi}")
        if correct:
            nb_correct += 1
    reusite=nb_correct / len(exemples) * 100
    print(f"bonne reponse: {nb_correct}/{len(exemples)} pourcentage de reussite: {reusite:.2f}%")
    # la fenêtre de l'arbre bloque le programme : on l'ouvre en dernier
    dessiner_arbre(noeuds)

if __name__=="__main__":
    main()
