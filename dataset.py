#from pathlib import Path

#debut=10
#longueur=8
#tokens_test = [10, 20, 30, 40, 50]
#chemin=Path("/Users/isidorezongo/Documents/machine learning/mini_gpt/data/train.txt")
#texte=chemin.read_text(encoding="utf-8")
#print(texte[:100])
def creer_exemple(token_ids,debut,longueur):
    token_ids_extrait=token_ids[debut:(longueur+ debut+1)]
    x=token_ids_extrait[:-1]
    y=token_ids_extrait[1:]
    return x , y
#test=creer_exemple(tokens_test,0,4)
#print(test)




#texte=pd.read_text("/Users/isidorezongo/Documents/machine learning/mini_gpt/data/train.txt",encoding="utf-8")

