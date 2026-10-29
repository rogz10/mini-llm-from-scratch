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
4. **Vérification automatique** : une phrase est rejetée si elle recopie le roman, ou si sa réponse n'apparaît pas près de l'indice dans le texte d'entraînement. Les rejets sont listés dans `rapport_construction.json`.
5. **Mauvaises réponses** : trois noms du même type et de la même longueur, qui ne vont jamais avec ce prénom ou cet indice dans le corpus.
6. **Séparation dev / test** : 25 % en dev, 75 % en test, par nom (toutes les questions dont la réponse est « Smith » sont du même côté). Ce découpage est figé dans `decoupage_fige.json` : un nom ne change jamais de côté.

## Historique

- **Version 1** (27/10) : premier jeu.
- **Version 2** (27/10) : dans la famille `non_vus`, les mauvaises réponses sont maintenant aussi des noms jamais lus. En version 1, c'étaient des noms connus du modèle : il évitait le seul nom inconnu et tombait sous le hasard. Les familles `personnages` et `faits` sont inchangées.

## Règles d'usage

- **dev** (`questions_dev.json`) : on s'en sert pour mettre au point le code et les choix (format des questions, calcul des scores). On a le droit de le regarder autant qu'on veut.
- **test** (`questions_test.json`) : on le lance **une seule fois**, quand tout est figé. C'est le chiffre qu'on annonce.
- Référence « nom le plus fréquent », calculée sans le modèle : voir `rapport_construction.json`. Le modèle doit faire mieux, sinon il ne fait que choisir le nom qu'il a le plus souvent lu.

## Limites

- Les questions sont peu nombreuses (133 : 36 en dev, 97 en test). Sur 50 questions, une erreur fait bouger le score de 2 points.
- Pour la famille `faits`, la vérification confirme que la réponse est proche de l'indice dans le texte, pas que la phrase est parfaite.
- La famille `non_vus` est petite (8 questions) : elle sert de contrôle, pas de résultat.
