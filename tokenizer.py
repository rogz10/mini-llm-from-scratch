from pathlib import Path


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
    print(numero, repr(caractere))
    char_to_id[caractere]=numero
    id_to_char[numero]=caractere

print(char_to_id)
print(char_to_id["b"])

token_ids=encode(texte_test,char_to_id)
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
    print(id_tok)

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



