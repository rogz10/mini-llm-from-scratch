# Mini GPT from Scratch

Implémentation **de zéro** d'un modèle de langage de type GPT, au niveau caractère, en Python et PyTorch, **sans bibliothèque de Transformer** : chaque brique (tokenizer, attention causale, têtes multiples, blocs, génération) est écrite à la main et vérifiée.

Le modèle est entraîné sur **34 romans de Jules Verne** (18,2 millions de caractères, domaine public) et tourne entièrement sur un Mac (GPU Apple, `mps`).

Le projet va au-delà du pré-entraînement : il étudie aussi les **limites** du modèle (ce qu'il « sait »(Knowledge probing), l'abstention « je ne sais pas », l'appel d'outil).

**Statut : en développement**, réimplémentation pas à pas, une brique par étape (voir la feuille de route).

---

## Résultats

### Dans ce dépôt, étape par étape

Corpus Verne complet, découpage **90 % entraînement / 5 % validation / 5 % test**, 5 000 pas, batch 32, contexte de 8 caractères, AdamW (lr 0,01). La perte est l'entropie croisée sur le jeu de **test**.

| modèle | paramètres | perte test |
|---|---|---|
| hasard (120 caractères) | — | 4,787 |
| meilleur bigramme possible (calcul théorique) | — | 2,474 |
| MiniGPT, 1 tête d'attention | 10 872 | 2,505 |
| **MiniGPT, 4 têtes d'attention** | **10 872** | **2,403** |

À nombre de paramètres égal, découper l'attention en 4 têtes fait passer sous le plafond théorique du bigramme.


## Structure

```
mini_gpt/
├── tokenizer.py       vocabulaire, encode / decode (texte ↔ identifiants)
├── dataset.py         extraits et batches (x, y décalé d'un caractère)
├── model.py           Bigramme, Tete (attention causale), MultiTetes, MiniGPT
├── train.py           entraînement, évaluation validation / test, sauvegarde
├── generate.py        génération autorégressive
├── visualisation.py   projection des embeddings (PCA)
├── requirements.txt   bibliothèques et versions
├── ruff.toml          règles de style (ruff)
├── data/              corpus, non versionné
└── model/             poids entraînés, non versionnés
```

---

## Installation et utilisation

```bash
pip install -r requirements.txt
python train.py        # entraîne le modèle et l'enregistre dans model/
python generate.py     # génère 200 caractères avec le modèle entraîné
```

**Le corpus** est constitué des 34 romans français de Jules Verne du [Projet Gutenberg](https://www.gutenberg.org/ebooks/author/60), concaténés dans `data/verne_complet.txt`, sans les en-têtes et licences Gutenberg ni les notes de numérisation, et sans les caractères très rares (moins de 20 occurrences). Un script de préparation automatique est en cours d'écriture.

---

## Feuille de route
**Objectif** : 6 blocs Transformer, 6 têtes, 192 dimensions, contexte de 128 caractères, **2,7 M de paramètres**

- [x] Tokenizer caractère et batches
- [x] Modèle bigramme de référence et boucle d'entraînement
- [x] Génération autorégressive
- [x] Attention causale (une tête), écrite à la main
- [x] Attention multi-têtes
- [ ] Bloc Transformer : couche feed-forward, connexions résiduelles, LayerNorm, blocs empilés
- [ ] Embeddings de position
- [ ] Passage à l'échelle : ~2,7 M de paramètres, GPU Apple, surapprentissage
- [ ] Échantillonnage : température, top-k
- [ ] Visualisation des embeddings appris
- [ ] *Knowledge probing*
- [ ] Abstention par fine-tuning supervisé
- [ ] Appel d'outil

---



Projet personnel d'apprentissage, inspiré de la série *Neural Networks: Zero to Hero* d'Andrej Karpathy.
