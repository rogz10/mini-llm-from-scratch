"""
Construit le jeu de questions de knowledge probing à partir du corpus Verne.

Lancer depuis la racine du projet :
    python evals/construire_questions.py

Trois familles de questions, chacune en QCM à 4 candidats (hasard = 25 %) :
- personnages : "Phileas " -> Fogg ? Extraites automatiquement du texte d'entraînement.
- faits : phrases à trous (evals/faits_ecrits.json),
  gardées seulement si le corpus confirme la réponse.
- non_vus : personnages présents seulement dans les 10 % de la fin du corpus, que le modèle
  n'a jamais lus. Groupe de contrôle : le modèle doit y être proche du hasard.

Sorties : evals/questions_dev.json, evals/questions_test.json, evals/rapport_construction.json
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

RACINE = Path(__file__).parent.parent
DOSSIER = Path(__file__).parent
GRAINE = 0
NB_DISTRACTEURS = 3      # 1 bonne réponse + 3 mauvaises = 4 candidats
PART_DEV = 0.25          # 25 % des questions en dev, 75 % en test
MIN_OCCURRENCES = 5      # un personnage doit apparaître au moins 5 fois pour être testé
MIN_DOMINANCE = 0.8      # le nom attendu doit suivre le prénom dans au moins 80 % des cas
FENETRE = 400            # distance max (en caractères) pour dire que deux mots « vont ensemble »
DEJA_VUS_EN_DEV = {"Phileas Fogg", "Cyrus Smith"}  # déjà utilisés pendant le développement

# Revue des paires « Majuscule Majuscule » : ce qui n'est pas un prénom de personnage.
PAS_DES_PRENOMS = {
    "Monsieur", "Madame", "Mme", "Mrs", "Mr", "Miss", "Mistress", "Sir", "Lady", "Lord", "Maître",
    "Herr", "Dame", "Uncle", "Master", "Capitaine", "Sergent", "President", "Bégum", "Vve",
    "Mais", "Et", "Le", "La", "Les", "Que", "Cependant", "Aussi", "Ami", "Par", "En", "Mon",
    "Grand", "Sa", "Son", "Votre", "Leurs", "Si", "Ni", "Au", "Puis", "Quant", "Alors", "Ce",
    "Van", "Mac", "De", "New", "San", "Céleste", "Polar", "Ligue", "Asie", "Nouveau", "Nouvelle",
    "Unième", "Quinzième", "Fleet", "Strada", "Société", "Daily", "Grands", "Centrale", "Ancien",
    "Longue", "Hôtel", "Goat", "Gulf", "Far", "Terre", "Niagara", "Fulgurateur", "Flamme",
    "Quatuor", "Portugais", "Martin", "Mathias", "Ile", "Conférence", "Golden",
    "Darwin", "Ponsonby", "Sentry", "Gold", "République", "Jules", "Friedrich",
}


# Noms de famille présents seulement dans les 10 % de la fin du corpus (jamais lus à
# l'entraînement), relus à la main : on a retiré les mots qui ne sont pas des noms de famille
# (Canoe, Bolide, Libéria...), les prénoms (Jenny, Arcadia...) et Raynaud (variante de Reynaud).
NOMS_JAMAIS_LUS = [
    "Halg", "Marx", "Mitz", "Sirk", "Proth", "Vragg",
    "Blaker", "Ceroni", "Dorick", "Kinney", "Norely", "Playne", "Xirdal",
    "Beauval", "Clarish", "Forsyth", "Fuentès", "Gimelli", "Karroly", "Piborne", "Reynaud",
    "Thibaut", "Whaston", "Arvidson", "Hudelson", "Saratoff", "Stanfort", "Haldorsen",
]


def charger_corpus():
    texte = (RACINE / "data" / "verne_complet.txt").read_text(encoding="utf-8")
    fin_train = int(len(texte) * 0.9)  # même découpe que train.py (90 % train)
    return texte, fin_train


def compter_paires(texte, fin_train):
    """Compte les paires « Prénom Nom » dans le train et dans le hors-train."""
    motif = re.compile(r"(?<![\w-])([A-ZÉ][a-zéèëêïîôç]+) ([A-ZÉ][a-zéèëêïîôç]+(?:-[A-Z][a-zé]+)?)(?![\w-])")
    train, hors_train = Counter(), Counter()
    for m in motif.finditer(texte):
        compteur = train if m.start() < fin_train else hors_train
        compteur[(m.group(1), m.group(2))] += 1
    return train, hors_train


def associations_dominantes(paires, autres_paires):
    """Garde les prénoms suivis presque toujours du même nom (sinon la question est ambiguë)."""
    par_prenom = {}
    for (prenom, nom), nb in paires.items():
        par_prenom.setdefault(prenom, Counter())[nom] += nb
    resultat = []
    for prenom, noms in par_prenom.items():
        if prenom in PAS_DES_PRENOMS:
            continue
        nom, nb = noms.most_common(1)[0]
        total = sum(noms.values()) + sum(nb2 for (p, _), nb2 in autres_paires.items() if p == prenom)
        if nb >= MIN_OCCURRENCES and nb / total >= MIN_DOMINANCE:
            resultat.append({"prenom": prenom, "nom": nom, "occurrences": nb, "dominance": round(nb / total, 3)})
    return resultat


def suit_le_prenom(texte, prenom, nom):
    return f"{prenom} {nom}" in texte


def proches(texte, mot_a, mot_b, debut=0, fin=None):
    """Nombre de fois où mot_b apparaît à moins de FENETRE caractères de mot_a."""
    fin = len(texte) if fin is None else fin
    positions_b = [m.start() for m in re.finditer(re.escape(mot_b), texte[debut:fin])]
    if not positions_b:
        return 0
    nb = 0
    for m in re.finditer(re.escape(mot_a), texte[debut:fin]):
        if any(abs(m.start() - p) <= FENETRE for p in positions_b):
            nb += 1
    return nb


def choisir_distracteurs(rng, reponse, reserve, accepte):
    """3 mots de la réserve, même longueur que la réponse, et acceptés par la règle donnée."""
    possibles = sorted({m for m in reserve if len(m) == len(reponse) and m != reponse and accepte(m)})
    if len(possibles) < NB_DISTRACTEURS:
        return None
    return rng.sample(possibles, NB_DISTRACTEURS)


def faire_question(rng, famille, prefixe, reponse, distracteurs, preuve):
    candidats = [reponse + " "] + [d + " " for d in distracteurs]  # l'espace marque la fin du mot
    rng.shuffle(candidats)  # la bonne réponse n'est pas toujours à la même place
    return {"famille": famille, "prefixe": prefixe, "candidats": candidats,
            "attendu": reponse + " ", "preuve": preuve}


def main():
    rng = random.Random(GRAINE)
    texte, fin_train = charger_corpus()
    train_txt = texte[:fin_train]
    vocabulaire = set(texte)
    paires_train, paires_hors = compter_paires(texte, fin_train)
    rejets = []
    questions = []

    # Famille 1 : personnages vus pendant l'entraînement
    persos = associations_dominantes(paires_train, paires_hors)
    reserve_noms = [p["nom"] for p in persos]
    for p in sorted(persos, key=lambda x: -x["occurrences"]):
        distracteurs = choisir_distracteurs(
            rng, p["nom"], reserve_noms, lambda m, p=p: not suit_le_prenom(texte, p["prenom"], m))
        if distracteurs is None:
            rejets.append({"famille": "personnages", "quoi": f"{p['prenom']} {p['nom']}",
                           "raison": "pas 3 noms de même longueur"})
            continue
        preuve = {"occurrences_train": p["occurrences"], "dominance": p["dominance"]}
        questions.append(faire_question(rng, "personnages", p["prenom"] + " ", p["nom"], distracteurs, preuve))

    # Famille 2 : phrases à trous, vérifiées dans le corpus
    faits = json.loads((DOSSIER / "faits_ecrits.json").read_text(encoding="utf-8"))
    reserves = {t: [m for m in mots if train_txt.count(m) >= MIN_OCCURRENCES]
                for t, mots in faits["reserves"].items()}
    for f in faits["faits"]:
        phrase, reponse, cle = f["phrase"], f["reponse"], f["cle"]
        # Version 3 (30/10) : pas de prénom ni de titre juste avant le trou (« ... Phileas » -> Fogg).
        # Sinon le modèle complète le nom par habitude, sans rien savoir du fait : c'est une fuite.
        if phrase.split()[-1][0].isupper():
            rejets.append({"famille": "faits", "quoi": phrase,
                           "raison": f"fuite : le mot juste avant le trou est « {phrase.split()[-1]} »"})
            continue
        if phrase.strip() in texte:
            rejets.append({"famille": "faits", "quoi": phrase, "raison": "phrase recopiée du roman"})
            continue
        confirmations = proches(texte, cle, reponse, 0, fin_train)
        if confirmations < 3:
            rejets.append({"famille": "faits", "quoi": phrase,
                           "raison": f"réponse non confirmée près de « {cle} » ({confirmations} fois)"})
            continue
        distracteurs = choisir_distracteurs(
            rng, reponse, reserves[f["type"]], lambda m, cle=cle: proches(texte, cle, m) == 0)
        if distracteurs is None:
            rejets.append({"famille": "faits", "quoi": phrase, "raison": "pas 3 distracteurs valables"})
            continue
        preuve = {"cle": cle, "confirmations_train": confirmations}
        questions.append(faire_question(rng, "faits", phrase, reponse, distracteurs, preuve))

    # Famille 3 : contrôle, personnages jamais lus pendant l'entraînement.
    # Version 2 (27/10) : les mauvaises réponses sont AUSSI des noms jamais lus. En version 1,
    # c'étaient des noms lus (Hobson, Munbar...) : le modèle évitait le seul nom inconnu et
    # tombait sous le hasard (0/3). Un contrôle juste doit donner 25 % quand on ne sait rien.
    for nom in NOMS_JAMAIS_LUS:
        assert nom not in train_txt, f"{nom} apparaît dans le train"
    rng_non_vus = random.Random(GRAINE + 1)  # hasard séparé : les deux autres familles ne changent pas
    non_vus = [p for p in associations_dominantes(paires_hors, paires_train)
               if p["nom"] not in train_txt]
    for p in non_vus:
        distracteurs = choisir_distracteurs(
            rng_non_vus, p["nom"], NOMS_JAMAIS_LUS, lambda m, p=p: not suit_le_prenom(texte, p["prenom"], m))
        if distracteurs is None:
            rejets.append({"famille": "non_vus", "quoi": f"{p['prenom']} {p['nom']}",
                           "raison": "pas 3 noms de même longueur"})
            continue
        preuve = {"occurrences_hors_train": p["occurrences"], "occurrences_train": 0}
        questions.append(faire_question(rng_non_vus, "non_vus", p["prenom"] + " ", p["nom"], distracteurs, preuve))

    # Contrôles automatiques sur toutes les questions
    for q in questions:
        for c in q["candidats"]:
            assert set(q["prefixe"] + c) <= vocabulaire, f"caractère hors vocabulaire : {q}"
        assert len({len(c) for c in q["candidats"]}) == 1
        assert q["candidats"].count(q["attendu"]) == 1

    # Séparation dev / test, famille par famille
    # On sépare par NOM attendu, pas par question : toutes les questions dont la réponse est
    # « Smith » vont du même côté. Sinon on mettrait au point sur Smith en dev, puis on
    # « découvrirait » Smith en test, et le test ne serait plus une surprise.
    # Le découpage de la version 1 est figé : un nom déjà placé ne change jamais de côté.
    fige = json.loads((DOSSIER / "decoupage_fige.json").read_text(encoding="utf-8"))
    dev, test = [], []
    for famille in ["personnages", "faits", "non_vus"]:
        noms = sorted({q["attendu"] for q in questions if q["famille"] == famille})
        rng.shuffle(noms)
        nb_dev = round(len(noms) * PART_DEV)
        noms_dev = set(noms[:nb_dev])
        for q in questions:
            if q["famille"] != famille:
                continue
            deja_vu = (q["prefixe"] + q["attendu"]).strip() in DEJA_VUS_EN_DEV
            deja_place_en_dev = any(q2["attendu"] == q["attendu"] for q2 in dev)
            deja_place_en_test = any(q2["attendu"] == q["attendu"] for q2 in test)
            if q["attendu"] in fige["dev"]:
                dev.append(q)
            elif q["attendu"] in fige["test"]:
                test.append(q)
            elif deja_vu or deja_place_en_dev or (q["attendu"] in noms_dev and not deja_place_en_test):
                dev.append(q)
            else:
                test.append(q)
    assert not {q["attendu"] for q in dev} & {q["attendu"] for q in test}, "un nom est des deux côtés"
    for nom_ensemble, ensemble in [("dev", dev), ("test", test)]:
        for i, q in enumerate(ensemble):
            q["id"] = f"{nom_ensemble}-{i:03d}"

    # Référence sans modèle : choisir le candidat dont le nom est le plus fréquent dans le train
    def reference_frequence(ensemble):
        bilan = {}
        for famille in ["personnages", "faits", "non_vus"]:
            qs = [q for q in ensemble if q["famille"] == famille]
            justes = sum(max(q["candidats"], key=lambda c: train_txt.count(c)) == q["attendu"] for q in qs)
            bilan[famille] = f"{justes}/{len(qs)}"
        return bilan

    (DOSSIER / "questions_dev.json").write_text(json.dumps(dev, ensure_ascii=False, indent=1), encoding="utf-8")
    (DOSSIER / "questions_test.json").write_text(json.dumps(test, ensure_ascii=False, indent=1), encoding="utf-8")
    rapport = {
        "graine": GRAINE,
        "nb_questions": {"dev": Counter(q["famille"] for q in dev), "test": Counter(q["famille"] for q in test)},
        "reference_hasard": "25 % (4 candidats)",
        "reference_frequence": {"dev": reference_frequence(dev), "test": reference_frequence(test)},
        "rejets": rejets,
    }
    (DOSSIER / "rapport_construction.json").write_text(
        json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rapport.items() if k != "rejets"}, ensure_ascii=False, indent=1))
    print(f"{len(rejets)} rejets (détail dans rapport_construction.json)")


if __name__ == "__main__":
    main()
