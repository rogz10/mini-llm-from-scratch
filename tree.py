"""Exemple de dessin pour la liste de nœuds construite par probe.py.

"""

import matplotlib.pyplot as plt


def calculer_positions(noeuds):
    """Place les feuilles de gauche à droite puis les parents au-dessus."""
    enfants = {}
    profondeur_max = 0

    for noeud in noeuds:
        enfants[noeud["texte"]] = []
        profondeur_max = max(profondeur_max, noeud["niveau"])

    for noeud in noeuds:
        parent = noeud["parent"]
        if parent is not None:
            enfants[parent].append(noeud["texte"])

    positions = {}
    numero_feuille = 0

    for noeud in noeuds:
        texte = noeud["texte"]
        if len(enfants[texte]) == 0:
            positions[texte] = (numero_feuille, -noeud["niveau"])
            numero_feuille = numero_feuille + 1

    for niveau in range(profondeur_max, -1, -1):
        for noeud in noeuds:
            texte = noeud["texte"]
            if noeud["niveau"] == niveau and len(enfants[texte]) > 0:
                somme_x = 0
                for enfant in enfants[texte]:
                    somme_x = somme_x + positions[enfant][0]
                x = somme_x / len(enfants[texte])
                positions[texte] = (x, -niveau)

    return positions


def dessiner_arbre(noeuds, chemin_image=None):
    """Affiche l'arbre ou l'enregistre si chemin_image est fourni

    Chaque case affiche la probabilité locale du caractère après son parent
    Les probabilités de chemin restent présentes dans les données
    """
    positions = calculer_positions(noeuds)
    fig, ax = plt.subplots(figsize=(20, 9))
    fig.patch.set_facecolor("#191919")
    ax.set_facecolor("#191919")

    for noeud in noeuds:
        parent = noeud["parent"]
        if parent is not None:
            x_parent, y_parent = positions[parent]
            x, y = positions[noeud["texte"]]
            ax.plot([x_parent, x], [y_parent, y], color="#777777", zorder=1)

    for noeud in noeuds:
        x, y = positions[noeud["texte"]]

        if noeud["parent"] is None:
            etiquette = repr(noeud["texte"])
            couleur = "#303030"
            couleur_texte = "#eeeeee"
        else:
            caractere = noeud["caractere"]
            if caractere == " ":
                caractere = "espace"
            elif caractere == "\n":
                caractere = "\\n"
            elif caractere == "\t":
                caractere = "\\t"

            pourcentage = 100 * noeud["proba_locale"]
            if pourcentage >= 0.01:
                valeur = f"{pourcentage:.2f} %"
            else:
                valeur = f"{pourcentage:.2g} %"
            etiquette = caractere + "\n" + valeur
            couleur = "#dceaff" if noeud["niveau"] == 1 else "#e8e8e8"
            if noeud["niveau"] >= 3:
                couleur = "#d6eee0"
            couleur_texte = "#202020"

        ax.text(
            x, y, etiquette,
            ha="center", va="center", fontsize=11, color=couleur_texte,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": couleur, "edgecolor": "none"},
            zorder=2,
        )

    ax.set_title("Arbre des prédictions, probabilités locales", color="#eeeeee", pad=20)
    ax.margins(x=0.06, y=0.15)
    ax.axis("off")
    fig.tight_layout()

    if chemin_image is None:
        plt.show()
    else:
        fig.savefig(chemin_image, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
