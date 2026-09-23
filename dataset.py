import torch


def creer_exemple(token_ids,debut,longueur):
    token_ids_extrait=token_ids[debut:(longueur+ debut+1)]
    #print(token_ids_extrait)
    x=token_ids_extrait[:-1]
    y=token_ids_extrait[1:]
    return x , y

def creer_batch(donnees,taille_batch,longueur):
    liste_x=[]
    liste_y=[]
    position =torch.randint(0,(len(donnees)-longueur),(taille_batch,))


    for debut in position:
        x,y=creer_exemple(donnees,debut,longueur)
        liste_x.append(x)
        liste_y.append(y)
    tensor_liste_x=torch.stack(liste_x)
    tensor_liste_y=torch.stack(liste_y)
    #print(tensor_liste_x)

    return tensor_liste_x,tensor_liste_y

if __name__=="__main__":
    test=creer_exemple(torch.arange(10),0,4)
    print(test)
    essai=creer_batch(torch.arange(10), taille_batch=4, longueur=3)
    print(essai[0])












