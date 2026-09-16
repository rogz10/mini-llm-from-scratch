from pathlib import Path

import torch

from dataset import creer_exemple

debut = 10
longueur=8
def encode(texte_test,char_to_id):
    mot=texte_test
    token_ids=[]

    for element in mot:
        token_ids.append(char_to_id[element])
    return(token_ids)
chemin=Path(__file__).parent/"data"/"train.txt"
texte=chemin.read_text(encoding="utf-8")
texte_test=texte[:100]
print(f"len du texte_test:{len(texte_test)}")

print(f"nombre de caractère:{len(texte)}")
print(texte[:300])
#caractère unique (vocabulaire du texte)
caracteres=sorted(set(texte))
print(caracteres)
print("#"*50)
print(f"taille du vocabulaire:{len(caracteres)}")

char_to_id={}
id_to_char={}

for numero, caractere in enumerate(caracteres):
    #print(numero, repr(caractere))
    char_to_id[caractere]=numero
    id_to_char[numero]=caractere

print(char_to_id)
print(char_to_id["b"])

token_ids=encode(texte_test,char_to_id)
liste_depart=[0,10,20]
batch_x=[]
batch_y=[]
for nbre in liste_depart:
    x,y=creer_exemple(token_ids,nbre,longueur)
    batch_x.append(x)
    batch_y.append(y)
    print("x et y en fonction de depart")
    #print(x, y)
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
print("decode")

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
ids_tokens=decode(token_ids, id_to_char)
if ids_tokens== texte_test:
    resultat=True
else:
    resultat=False
print(ids_tokens)
print(resultat)
print(x,"\n")
print(len(x))
print(y)
print(len(y))
print("decodage de x et y")
ids_tokens_x=decode(x,id_to_char)
ids_tokens_y=decode(y,id_to_char)
print(repr(ids_tokens_x))
print(repr(ids_tokens_y))



