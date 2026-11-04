from collections import Counter
from pathlib import Path

from tokenizer import charger_texte


def compter_paires(ids):

    couple=zip(ids,ids[1:])

    compteur=Counter(couple)
    return compteur

def fusionner(ids,paire,nouvel_id):
    fusion=[]
    i=0
    while i < len(ids):
        if i +1 < len(ids) and ids[i]==paire[0] and ids[i+1]==paire[1]:
            fusion.append(nouvel_id)
            i+=2
        else:
            fusion.append(ids[i])
            i+=1
    return fusion

def entrainer(texte,nb_fusions):
    ids=list(texte.encode("utf-8"))
    fusions={}

    for tour in range(nb_fusions):
        compteur=compter_paires(ids)
        paire=compteur.most_common(1)[0][0]
        nouvel_id=256+tour
        ids=fusionner(ids,paire,nouvel_id)
        fusions[paire]=nouvel_id
    return fusions

def construire_vocab(fusions):
    vocab={}
    for i in range(256):
        vocab[i]=bytes([i])
    # dans l'ordre d'apprentissage : les deux morceaux existent déjà
    for paire, nouveau_numero in fusions.items():
        vocab[nouveau_numero]=vocab[paire[0]]+vocab[paire[1]]
    return vocab

def encode(texte,fusions):
    encodage=texte.encode("utf-8")
    encodage_liste=list(encodage)

    for paire, nouveau_numero in fusions.items():
        liste=fusionner(encodage_liste,paire,nouveau_numero)
        encodage_liste=liste

    return encodage_liste

def decode(ids,vocab):
    decodage=b""

    for i in ids:
        decodage+=vocab[i]
    return decodage.decode("utf-8",errors="replace")

if __name__ == "__main__":
    chemin=Path(__file__).parent/"data"/"verne_complet.txt"
    texte=charger_texte(chemin)

    fusions=entrainer(texte[:1000000],30)
    vocab=construire_vocab(fusions)
    print(f"taille du vocabulaire:{len(vocab)}")
    for paire, nouveau_numero in fusions.items():
        print(f"{paire} -> {nouveau_numero} {vocab[nouveau_numero]}")

    phrase="Les voyageurs étaient en route."
    print(decode(encode(phrase,fusions),vocab)==phrase)
    print(len(encode(phrase,fusions)))

    # texte jamais vu : les 100 000 caractères juste après le premier million
    texte_inconnu=texte[1000000:1100000]
    octets_inconnu=len(texte_inconnu.encode("utf-8"))
    tokens_inconnu=encode(texte_inconnu,fusions)
    print(decode(tokens_inconnu,vocab)==texte_inconnu)
    print(f"compression:{octets_inconnu/len(tokens_inconnu):.2f}")
