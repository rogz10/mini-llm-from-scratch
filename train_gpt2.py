import math
import time
from pathlib import Path

import numpy as np
import torch

from model_gpt2 import MiniGPT

"""
entraînement de MiniGPT taille GPT-2 small (124 M de paramètres) sur FineWeb-Edu
réglages RunPod et valeurs du test sur le Mac en commentaire
"""

# graine aléatoire pour la reproductibilité
torch.manual_seed(42)

# hyperparamètres
# données
t=1024# longueur d'un morceau en tokens (test Mac 64)
b=16# nombre de morceaux par micro-batch (test Mac 4)
tokens_par_pas=524288# nombre de tokens par pas (test Mac 512)
nb_micro=tokens_par_pas//(b*t)  # nombre de micro-batchs par pas
# learning rate
lr_max=6e-4# learning rate maximal (GPT-3 125M)
lr_min=0.1*lr_max# learning rate minimal
weight_decay=0.1# weight decay des tables 2D (GPT-3 125M)
warmup_steps=715# nombre de pas de montée du learning rate (test Mac 5)
max_step=5539# nombre total de pas pour une époque du train (test Mac 20)
# sauvegardes et validation
intervalle_sauvegarde=500# nombre de pas entre deux checkpoints (test Mac 10)
intervalle_val=250# nombre de pas entre deux mesures de la perte val (test Mac 5)
nb_batchs_val=20# nombre de micro-batchs par mesure de la perte val (test Mac 5)
# modèle (GPT-2 small)
taille_vocab=50257# nombre de tokens du vocabulaire GPT-2
embd_dim=768# nombre de valeurs par token
nb_tete=12# nombre de têtes d'attention par bloc
nb_bloc=12# nombre de blocs Transformer
longueur=1024# nombre de positions du modèle (t <= longueur)
dropout=0.0# taux de dropout

# device
if torch.cuda.is_available():
    device="cuda"# GPU Nvidia (RunPod)
elif torch.backends.mps.is_available():
    device="mps"# GPU du Mac
else:
    device="cpu"

# dossier des shards
dossier_shards=Path(__file__).parent/"data"/"fineweb_edu"/"shards"

def charger_shard(chemin):
    """charge un shard .npy (uint16) et le renvoie en tenseur int64"""
    shard=np.load(chemin)
    tableau=torch.from_numpy(shard.astype(np.int64))
    return tableau

class ChargeurShards:
    """lecture des shards dans l'ordre batch après batch

    paramètres:
        dossier: dossier des shards
        b: nombre de morceaux par batch
        t: nombre de tokens par morceau
        partie: "train" ou "val"

    attributs:
        num_shard: numéro du shard en cours
        curseur: position du prochain batch dans le shard
        tenseur: tokens du shard num_shard (à recharger si num_shard change)

    formes:
        prochain_batch() renvoie entree et cible (b, t)
        cible: entree décalée d'un token
    """

    def __init__(self,dossier,b,t,partie):
        self.b=b
        self.t=t
        # shards de la partie triés par numéro
        self.fichiers=sorted(dossier.glob(f"fineweb_{partie}_*.npy"))
        self.curseur=0
        self.num_shard=0
        self.tenseur=charger_shard(self.fichiers[self.num_shard])

    def prochain_batch(self):
        # b*t+1 tokens du batch (+1 pour la cible du dernier token)
        morceau=self.tenseur[self.curseur:self.curseur+ self.b*self.t+1]
        entree=morceau[:-1]# tokens sans le dernier
        cible=morceau[1:]# tokens sans le premier

        # découpage en b morceaux de t tokens
        entree=entree.reshape(self.b,self.t)
        cible=cible.reshape(self.b,self.t)

        # position du prochain batch
        self.curseur+=self.b*self.t
        need=self.curseur +(self.b*self.t+1)

        # fin du shard : shard suivant puis retour au premier après le dernier
        if need>len(self.tenseur):
            print(f"shard epuisé, cursur= {self.curseur}")
            self.num_shard=(self.num_shard+1)%len(self.fichiers)
            self.tenseur=charger_shard(self.fichiers[self.num_shard])
            self.curseur=0

        return entree,cible

# learning rate
def get_lr(step):
    """learning rate du pas step

    montée linéaire jusqu'à warmup_steps
    descente en cosinus jusqu'à max_step
    lr_min au-delà
    """
    #montée
    if step < warmup_steps:
        new_lr=lr_max*(step+1) /warmup_steps
        return new_lr
    # min
    if step> max_step:
        new_lr=lr_min
        return new_lr
    # descente
    avancement=(step - warmup_steps)/(max_step-warmup_steps)
    coef=0.5*(1.0 +math.cos(math.pi*avancement))
    new_lr=lr_min+ coef *(lr_max-lr_min)
    return new_lr

# modèle et optimiseur
modele=MiniGPT(taille_vocab,embd_dim,nb_tete,nb_bloc,longueur,dropout)
modele.to(device)# deplacer le modèle sur le device
nb_params=sum(p.numel() for p in modele.parameters())# nombre de paramètres
# paramètres avec et sans weight decay
params_decay=[]# tables 2D des Linear et des Embedding
params_no_decay=[]# biais et LayerNorm
for p in modele.parameters():
    if p.dim()>=2:
        params_decay.append(p)
    else:
        params_no_decay.append(p)
# AdamW à 2 groupes (lr remplacé à chaque pas par get_lr)
optimiseur=torch.optim.AdamW([
    {"params":params_decay,"weight_decay":weight_decay},
    {"params":params_no_decay,"weight_decay":0.0},
],lr=lr_max)

# chargeurs
chargeur_train=ChargeurShards(dossier_shards,b,t,"train")
chargeur_val=ChargeurShards(dossier_shards,b,t,"val")# mesure de la perte val uniquement

# decorateur pour éviter de suivre le gradient
@torch.no_grad()
def estimer_perte(modele,chargeur,nb_batchs):
    """perte moyenne du modèle sur les nb_batchs premiers batchs du chargeur

    paramètres:
        modele: modèle à évaluer
        chargeur: ChargeurShards à lire
        nb_batchs: nombre de batchs lus

    curseur remis à 0 : mêmes tokens à chaque mesure
    """
    modele.eval() # mode evaluation, pas de dropout
    chargeur.curseur=0# lecture depuis le début du shard
    perte_total=[]

    for _ in range(nb_batchs):
        x,y=chargeur.prochain_batch()
        # deplacer les batchs sur le device
        x,y=x.to(device),y.to(device)
        # calcul en bf16
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            _,loss=modele(x,y)
        perte_total.append(loss)
    moyenne_perte=(sum(perte_total))/len(perte_total)
    modele.train() # mode entrainement, dropout activé

    return moyenne_perte.item()

# dossier des checkpoints et du journal
dossier_modele=Path(__file__).parent/"model"/"gpt2"
dossier_modele.mkdir(parents=True,exist_ok=True)
journal=dossier_modele/"journal.txt"# une ligne par mesure (pas partie perte)
#with open (journal,"w",encoding="utf-8") as f:
#    f.write("pas partie perte\n")
#depart=0

# reprise depuis le dernier checkpoint s'il existe
liste_checkpoint=sorted(dossier_modele.glob("checkpoint_*.pt"))# triés par pas (nom sur 5 chiffres)
if len(liste_checkpoint)>0:
    last_checkpoint=liste_checkpoint[-1]# checkpoint le plus récent
    checkpoint_lu=torch.load(last_checkpoint,map_location=device)# tenseurs chargés sur le device
    # poids du modèle et mémoire d'AdamW
    modele.load_state_dict(checkpoint_lu["modele"])
    optimiseur.load_state_dict(checkpoint_lu["optimiseur"])
    # position du chargeur train (shard rechargé puis curseur)
    chargeur_train.num_shard=checkpoint_lu["num_shard"]
    chargeur_train.tenseur=charger_shard(chargeur_train.fichiers[chargeur_train.num_shard])
    chargeur_train.curseur=checkpoint_lu["curseur"]

    depart=checkpoint_lu["pas"]+1# pas suivant le dernier pas sauvegardé
    print(f"après reprise{last_checkpoint.name} depart au pas {depart} | shard:{chargeur_train.num_shard} | curseur:{chargeur_train.curseur}")
else:
    # premier lancement depuis le pas 0 avec un journal neuf
    depart=0
    with open(journal,"w",encoding="utf-8") as f:
        f.write("pas partie perte\n")

# boucle d'entraînement
for step in range(depart,max_step):
    start_time=time.time()# début du chrono du pas
    # learning rate du pas dans chaque groupe de l'optimiseur
    lr=get_lr(step)
    for groupe in optimiseur.param_groups:
        groupe["lr"]=lr
    optimiseur.zero_grad()# gradients du pas précédent remis à zéro
    perte_pas=0.0# perte moyenne du pas
    # accumulation du gradient sur nb_micro micro-batchs
    for micro in range(nb_micro):
        x,y=chargeur_train.prochain_batch()
        x,y=x.to(device),y.to(device)
        # calcul en bf16
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            _,loss=modele(x,y)
        loss=loss/nb_micro# part du micro-batch dans la moyenne du pas
        perte_pas+=loss.item()
        loss.backward()# gradient ajouté dans .grad
    optimiseur.step()# un seul pas après les nb_micro micro-batchs
    step_duration=time.time() - start_time# durée du pas en secondes

    print(f"pas:{step} | lr:{optimiseur.param_groups[0]['lr']:.2e} | perte:{perte_pas:.4f} | durée:{step_duration:.2f}s")
    # ligne train du journal
    with open(journal,"a",encoding="utf-8")as f:
        f.write(f"{step} train {perte_pas:.4f}\n")
    # perte val tous les intervalle_val pas et au dernier pas
    if step%intervalle_val==0 or step==max_step-1:
        perte_val=estimer_perte(modele,chargeur_val,nb_batchs_val)
        print(f"pas {step} perte_val={perte_val:.4f}")
        with open(journal,'a',encoding="utf-8")as f:
            f.write(f"{step} val {perte_val:.4f}\n")
    # checkpoint tous les intervalle_sauvegarde pas et au dernier pas
    if (step >0 and step%intervalle_sauvegarde==0) or step==max_step-1:
        # contenu nécessaire à la reprise
        checkpoint={
            "modele":modele.state_dict(),
            "optimiseur":optimiseur.state_dict(),
            "pas":step,
            "num_shard":chargeur_train.num_shard,
            "curseur":chargeur_train.curseur# position du prochain batch

        }
        chemin_checkpoint=dossier_modele/f"checkpoint_{step:05d}.pt"# pas sur 5 chiffres pour le tri
        torch.save(checkpoint,chemin_checkpoint)
        print(f"checkpoint sauvegardé:{chemin_checkpoint.name}")







