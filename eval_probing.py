import json
from pathlib import Path

import torch
import torch.nn.functional as F

from generate import charger_modele
from tokenizer import charger_texte, construire_vocabulaire, encode

longueur=256 # nombre max de lettre données au modèle pour prédire la suivante
embd_dim=384
nb_tetes=6
nb_blocs=6
dropout=0.2

def scorer_candidats(modele, prefixe, candidats, char_to_id, device):
    """
    Note chaque candidat en un seul passage du modèle (un candidat par ligne du batch).

    prefixe: le début de texte donné au modèle, par exemple "Phileas "
    candidats: les suites à comparer, toutes de même longueur, par exemple ["Fogg ", "Ford "]
    return : une liste d'un log-score par candidat (somme des log-probabilités de ses lettres) ;
    le plus grand, c'est-à-dire le moins négatif, est le candidat préféré
    """
    if not prefixe:
        raise ValueError('Le préfixe doit être non vide.')
    if not candidats or any(not c for c in candidats):
        raise ValueError('Il faut des candidats non vides.')
    if len({len(c) for c in candidats}) != 1:
        raise ValueError('Cette version attend des candidats de même longueur.')
    if len(prefixe) + len(candidats[0]) - 1 > longueur:
        raise ValueError("Le contexte dépasse la capacité du modèle.")
    sequences = [encode(prefixe + candidat, char_to_id) for candidat in candidats]
    entrees = torch.tensor([ids[:-1] for ids in sequences], dtype=torch.long, device=device)
    cibles = torch.tensor([ids[1:] for ids in sequences], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = modele(entrees)
        log_probabilites = F.log_softmax(logits, dim=-1)
        log_probas_cibles = log_probabilites.gather(dim=-1, index=cibles.unsqueeze(-1)).squeeze(-1)
        debut = len(prefixe) - 1
        scores = log_probas_cibles[:, debut:].sum(dim=-1)
    return scores.tolist()

def evaluer_exemples(modele,exemples,char_to_id,device):
    """
    exemples: liste de dictionnaires avec les clés "prefixe", "candidats" et "attendu"
    return : une fiche de résultat par exemple (l'exemple + ses scores, le choix et correct)
    """
    resultats=[]

    for exemple in exemples:
        scores=scorer_candidats(modele,exemple["prefixe"],exemple["candidats"],char_to_id,device)
        meilleur_score=max(scores)
        #indice meilleur score
        indice_meilleur_score=scores.index(meilleur_score)
        candidat_choisi=exemple["candidats"][indice_meilleur_score]
        correct=candidat_choisi==exemple["attendu"]

        fiche_resultat=exemple.copy()
        fiche_resultat["log_scores"]=scores
        fiche_resultat["candidat_choisi"]=candidat_choisi
        fiche_resultat["correct"]=correct
        resultats.append(fiche_resultat)

    return resultats
def occurences_personnages(fiche):
    nb_occurences=fiche["preuve"]["occurrences_train"]
    return nb_occurences
def occurences_faits(fiche):
    nb_occurences_faits=fiche["preuve"]["confirmations_train"]
    return nb_occurences_faits


def main():

    if torch.cuda.is_available():
        device=torch.device("cuda")
    elif torch.backends.mps.is_available():
        device=torch.device("mps")
    else:
        device=torch.device("cpu")
    print(f"device={device}")

    chemin=Path(__file__).parent/"data"/"verne_complet.txt"
    chemin_modele=Path(__file__).parent/"model"/"MiniGPT_10M.pt"

    texte=charger_texte(chemin)
    char_to_id,_,caractere=construire_vocabulaire(texte)

    modele=charger_modele(chemin_modele,len(caractere),embd_dim,nb_tetes,nb_blocs,longueur,dropout,device)
    chemin_json=Path(__file__).parent/"evals"/"questions_dev.json"
    texte_json=chemin_json.read_text(encoding="utf-8")
    questions=json.loads(texte_json)
    #print(questions[:5])
    resultats=evaluer_exemples(modele,questions,char_to_id,device)
    #print(len(resultats))
    #print(resultats[0])
    correct_answer=0
    for fiche in resultats:
        if fiche["correct"]:
            correct_answer+=1

    #print(correct_answer)
    resultat=correct_answer/len(resultats)*100
    print(f"pourcentage de reussite est de :{resultat:.2f} %")
    par_famille={}
    for fiche in resultats:
        famille=fiche["famille"]
        if famille not in par_famille:
            par_famille[famille]={"juste":0,"total":0}
        par_famille[famille]["total"]+=1
        if fiche["correct"]:
            par_famille[famille]["juste"]+=1
    for famille,compte in par_famille.items():
        print(f"{famille},{compte}| pourcentage:{compte['juste']/compte['total']*100:.2f}")
    nb_candidats=0
    for fiche in resultats:
        if fiche["famille"]=="non_vus":
            nb_candidats=len(fiche["candidats"])
    keeps=["personnages","faits"]
    non_vus="non_vus"
    #family=["personnages","faits","non_vus"]
    total_keep=0
    total_juste=0
    pourcentage=[]
    for keep in keeps:
        total_keep+=par_famille[keep]["total"]
        total_juste+=par_famille[keep]["juste"]
        points=par_famille[keep]["juste"]/par_famille[keep]["total"]*100
        pourcentage.append(points)
    total_juste_non_vu=par_famille[non_vus]["juste"]
    total_questions_non_vu=par_famille[non_vus]["total"]
    #total bonnes repnses/ total questions (chance de reussir sur une question aléatoire pour le moodel)
    micro=(total_juste/total_keep)*100
    # moyenne des pourcentages des familles (moyenne pour chaque type de famille)
    macro=sum(pourcentage)/len(keeps)
    #print(par_famille)
    print("#"*50)
    print(f"micro:{micro:.2f}%")
    print(f"macro:{macro:.2f} %")
    print(f"controle non_vus:{(total_juste_non_vu/total_questions_non_vu)*100:.2f}% (hasard de {(1/nb_candidats)*100:.2f}%)")

    personnages=[]
    for fiche in resultats:
        if fiche["famille"]=="personnages":
            personnages.append(fiche)
    personnages_tries=sorted(personnages,key=occurences_personnages)

    for fiche in personnages_tries:
        valeur_occu=fiche["preuve"]["occurrences_train"]
        attendu=fiche["attendu"]
        correct=fiche["correct"]
        print(f"nombre d'occurrences={valeur_occu} | attendu={attendu} | juste={correct}")
    print("#"*70)

    faits=[]
    for fiche in resultats:
        if fiche["famille"]=="faits":
            faits.append(fiche)
    faits_tries=sorted(faits,key=occurences_faits)

    for fiche in faits_tries:
        valeur_occu=fiche["preuve"]["confirmations_train"]
        attendu=fiche["attendu"]
        correct=fiche["correct"]
        prefixe=fiche["prefixe"]
        print(f"nombre d'occurrences={valeur_occu} | attendu={attendu} | juste={correct} | prefixe={prefixe}")

        #print(results)
    #print(personnages_tries[0]["preuve"]["occurrences_train"])
    #print(personnages_tries[18]["preuve"]["occurrences_train"])
    #print(len(personnages))


if __name__=="__main__":
    main()
