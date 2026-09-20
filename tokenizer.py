from pathlib import Path

import torch

from dataset import creer_exemple


def charger_texte(chemin):

    texte=chemin.read_text(encoding="utf-8")

    return texte

def encode(texte,char_to_id):
    mot=texte
    token_ids=[]

    for element in mot:
        token_ids.append(char_to_id[element])
    return token_ids


def decode(token_ids,id_to_char):
    id_tok=[]
    for i,numero in enumerate(token_ids):
        #print(i,numero)
        id_tok.append(id_to_char[numero])
        #print(id_tok)
        #id_to_char[caractere]
    #print(id_tok)

    return "".join(id_tok)

    #print(id_to_char)
#ids_tokens=decode([55, 68, 67], id_to_char)

def construire_vocabulaire(texte):
    char_to_id={}
    id_to_char={}

    #caractère unique (vocabulaire du texte)
    caracteres=sorted(set(texte))
    for numero, caractere in enumerate(caracteres):
        #print(numero, repr(caractere))
        char_to_id[caractere]=numero
        id_to_char[numero]=caractere

    return char_to_id, id_to_char,caracteres

if __name__ == "__main__":
    chemin=Path(__file__).parent/"data"/"train.txt"
    texte=charger_texte(chemin)
    char_to_id,id_to_char,caracteres=construire_vocabulaire(texte)

    token_ids=encode(texte,char_to_id)
    ids_tokens=decode(token_ids, id_to_char)

    debut = 10
    longueur=8
    liste_depart=[0,10,20]
    batch_x=[]
    batch_y=[]
    for nbre in liste_depart:
        x,y=creer_exemple(token_ids,nbre,longueur)
        batch_x.append(x)
        batch_y.append(y)
        print("x et y en fonction de depart")

    batch_x=torch.tensor(batch_x,dtype=torch.int64)
    batch_y=torch.tensor(batch_y,dtype=torch.int64)

    print(batch_x.shape)
    print(batch_x)
    print(batch_y.shape)
    print(f"batch x :{batch_x}")
    print(f"taille du batch:{len(batch_x)}")
    print(batch_x[0])
    print(f"batch y :{batch_y}")
    #token_ids_extrait=token_ids[debut:(longueur+debut+1)]
    #x,y=creer_exemple(token_ids,debut,longueur)
    print(f" longeur des id de token{len(token_ids)}")
    #print(f"nombre de token :{len(token_ids)}")
    #print(f"début du texte:{repr(texte[:20])}")
    #print(f"identifiant correspondant:{token_ids[:20]}")
    ids_tokens_x=decode(x,id_to_char)
    ids_tokens_y=decode(y,id_to_char)

    print(f"nombre de caractère:{len(texte)}")
    print(texte[:300])
    print(char_to_id)

    print(decode(encode(texte,char_to_id),id_to_char)==texte)
    print(caracteres)
    print("#"*50)
    print(f"taille du vocabulaire:{len(caracteres)}")

    print(x,"\n")
    print(len(x))
    print(y)
    print(len(y))
    print("decodage de x et y")

    print(repr(ids_tokens_x))
    print(repr(ids_tokens_y))






