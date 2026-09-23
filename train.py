from pathlib import Path

import torch

from tokenizer import charger_texte, construire_vocabulaire, encode

chemin=Path(__file__).parent/"data"/"train.txt"
texte=charger_texte(chemin)
print(texte[:100])
char_to_id,id_to_char,caractere=construire_vocabulaire(texte)
token_id=torch.tensor(encode(texte,char_to_id))
n=int(0.9 * len(token_id))
print(n)
donnees_train=token_id[:n]
donnees_val=token_id[n:]
print("shape de donne_train et donne_val")
print(donnees_train.shape)
print(donnees_val.shape)
taille_total=donnees_train.shape[0] + donnees_val.shape[0]
print(f"verif du total :{taille_total}")
print("#"*50)

for i ,(clé,valeur) in enumerate(id_to_char.items()):
    if i >=10:
        break
    print(f"{clé}: {valeur}")

if __name__ == "__main__":
    #chemin=Path(__file__).parent/"data"/"train.txt"
    #texte=charger_texte(chemin)
    print(len(texte))
    print("construction")
    #char_to_id,id_to_char,caractere=construire_vocabulaire(texte)
    print(f"char to id :{len(char_to_id)}")
    print(f"id to char :{len(id_to_char)}")
    print(f"caractère :{len(caractere)}")
    print(char_to_id)
    print("#"*50)
    print(id_to_char)
    print("#"*50)
    print(caractere)
    #token_id=torch.tensor(encode(texte,char_to_id))
    print(len(token_id))
    print(token_id.shape)




