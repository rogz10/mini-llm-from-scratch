# Jeu de questions : que sait le modèle des romans de Verne ?

## Ce que ce jeu mesure

Le modèle est un **modèle de base** : il continue du texte, il ne répond pas à des questions. On lui donne donc un début de texte et quatre suites possibles, et on regarde à laquelle il donne la plus grande probabilité. Une seule suite est juste. Si le modèle choisit au hasard, il a 1 chance sur 4 (25 %).

Trois familles de questions :

| Famille | Exemple | D'où elle vient | Ce qu'on attend |
|---|---|---|---|
| `personnages` | `"Phileas "` → `Fogg` / `Zorn` / `Ogul` / `Land` | extraite automatiquement du texte d'entraînement | au-dessus du hasard si le modèle a retenu le nom |
| `faits` | `"Le capitaine du Nautilus se nommait "` → `Nemo` / … | phrases à trous, vérifiées dans le corpus | au-dessus du hasard s'il relie un rôle à un nom |
| `non_vus` | `"Lazare "` → `Ceroni` / … | personnages présents **seulement** dans les 10 % de la fin, jamais lus à l'entraînement | proche du hasard ; sinon, c'est que le test a une fuite |

## Comment il est construit

`python evals/construire_questions.py` refait tout à l'identique (graine fixée).

1. **Extraction** : on cherche dans le texte toutes les paires « Prénom Nom » et on les compte.
2. **Revue** : on retire ce qui n'est pas un personnage (`Monsieur`, `Mais`, `New York`…) et les prénoms ambigus (si un prénom est suivi de plusieurs noms, comme Michel Ardan et Michel Strogoff, la question n'a pas une seule bonne réponse).
3. **Rédaction des faits** : des phrases à trous sur les romans (`faits_ecrits.json`).
4. **Vérification automatique** : une phrase est rejetée si le mot juste avant le trou est un prénom ou un titre (une **fuite**, voir version 3), si elle recopie le roman, ou si sa réponse n'apparaît pas près de l'indice dans le texte d'entraînement. Les rejets sont listés dans `rapport_construction.json`.
5. **Mauvaises réponses** : trois noms du même type et de la même longueur, qui ne vont jamais avec ce prénom ou cet indice dans le corpus.
6. **Séparation dev / test** : 25 % en dev, 75 % en test, par nom (toutes les questions dont la réponse est « Smith » sont du même côté). Ce découpage est figé dans `decoupage_fige.json` : un nom ne change jamais de côté.

## Historique

- **Version 1** (27/10) : premier jeu.
- **Version 2** (27/10) : dans la famille `non_vus`, les mauvaises réponses sont maintenant aussi des noms jamais lus. En version 1, c'étaient des noms connus du modèle : il évitait le seul nom inconnu et tombait sous le hasard. Les familles `personnages` et `faits` sont inchangées.
- **Version 3** (30/10) : dans la famille `faits`, plus de prénom ni de titre juste avant le trou. L'analyse d'erreurs sur dev a montré une **fuite** : avec « … s'appelait Phileas » → `Fogg`, le modèle complète le nom par habitude, sans rien savoir du fait (6/6 sur ces questions). 28 phrases réécrites (prénom retiré), et le script rejette désormais ce cas. Seuls les débuts de phrase changent : candidats, réponses et découpage dev / test sont identiques à la version 2. Sur dev, `faits` passe de 10/14 à 4/14 : les 4 réussites restantes sont des phrases qui n'avaient pas de prénom final.

## Règles d'usage

- **dev** (`questions_dev.json`) : on s'en sert pour mettre au point le code et les choix (format des questions, calcul des scores). On a le droit de le regarder autant qu'on veut.
- **test** (`questions_test.json`) : on le lance **une seule fois**, quand tout est figé. C'est le chiffre qu'on annonce.
- Référence « nom le plus fréquent », calculée sans le modèle : voir `rapport_construction.json`. Le modèle doit faire mieux, sinon il ne fait que choisir le nom qu'il a le plus souvent lu.

## Limites

- Les questions sont peu nombreuses (133 : 36 en dev, 97 en test). Sur 50 questions, une erreur fait bouger le score de 2 points.
- Pour la famille `faits`, la vérification confirme que la réponse est proche de l'indice dans le texte, pas que la phrase est parfaite.
- La famille `non_vus` est petite (8 questions) : elle sert de contrôle, pas de résultat.

## Résultats officiels du test (version 3, 30/10/2026)

Modèle : MiniGPT de la séance 10 (10 831 224 paramètres, 6 blocs × 6 têtes, 384 dimensions, contexte 256), entraîné sur le corpus Verne complet. Le test a été lancé **une seule fois**, après avoir tout mis au point sur le dev.

| famille | dev (v3) | **test** | hasard |
|---|---|---|---|
| `personnages` | 13/19 = 68,4 % | **39/53 = 73,6 %** | 25 % |
| `faits` | 4/14 = 28,6 % | **11/39 = 28,2 %** | 25 % |
| `non_vus` (contrôle) | 0/3 = 0 % | **1/5 = 20 %** | 25 % |
| **micro** (personnages + faits) | 51,5 % | **54,4 %** | |
| **macro** (personnages + faits) | 48,5 % | **50,9 %** | |

Ce que ça montre :

- **Personnages** : le modèle retient l'association « prénom → nom » à partir d'une quinzaine d'occurrences dans le texte d'entraînement. Avec 14 occurrences ou moins : 2/15 (le hasard). Avec 15 ou plus : 37/38.
- **Faits** : au niveau du hasard. Le modèle ne relie pas une description (« le gentleman du tour du monde ») au nom du personnage. Avec 39 questions, l'écart de 3 points avec le hasard n'est pas significatif.
- **Contrôle** : proche du hasard sur les noms jamais lus, donc pas de fuite dans cette famille.
- Les chiffres du dev et du test sont proches : la mise au point sur le dev n'a pas faussé le résultat.

Empreintes SHA-256 des fichiers qui ont produit ces chiffres (le modèle et le corpus ne sont pas dans le dépôt : leur empreinte est la seule trace de la version utilisée) :

```
197b3c3af325b4f9e208a0f53996069d98f127ae67deb4294c38c960eb0d7dce  evals/questions_test.json
0f6d6f330b99e4d8f5434a8c6d4e0f8301c25c8dc64d636e7d847f879030ebe5  model/MiniGPT_10M.pt
fb70d725008e501774a6bb7133aaf049160c96f53aaaa61589ca2d8c4c7676bf  data/verne_complet.txt
11813c76cb6f2c77b5b2cf31d75483d719e45b8129bdf3db8db3c971e272939c  eval_probing.py
```

Pour vérifier qu'on a les mêmes fichiers, depuis la racine du projet :

```
shasum -a 256 evals/questions_test.json model/MiniGPT_10M.pt data/verne_complet.txt eval_probing.py
```
