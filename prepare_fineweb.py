from pathlib import Path

import pandas as pd
import tiktoken
from huggingface_hub import hf_hub_download
import numpy as np

#params
capacite=100000000
position=0
compteur=0
dossier_shards=Path(__file__).parent/"data"/"fineweb_edu"/"shards"
tableau=np.zeros(capacite,dtype=np.uint16)

# encodage
enc=tiktoken.get_encoding("gpt2")
def tokenizer(page):

    token_special=enc.eot_token # token spé pour separer une page d'une autre  ("<|endoftext|>")
    texte_encoder=enc.encode_ordinary(page)
    texte_encoder_page=[token_special] + texte_encoder

    return texte_encoder_page
total_token=0
for num in range(4):
    nbre_token_fichier=0
    nom_fichier=f"sample/10BT/{num:03d}_00000.parquet"

    chemin = hf_hub_download(
        repo_id="HuggingFaceFW/fineweb-edu",# nom depot
        repo_type="dataset",# jeu de données
        filename=nom_fichier,  # fichier voulu dans le depot
        local_dir=Path(__file__).parent/"data"/"fineweb_edu",# save dans fineweb_edu
    )
    print(chemin)

    # parquet

    df=pd.read_parquet(chemin,columns=["text"])
    pages=df["text"]
    # parcourir le texte de chaque pages
    for page in pages:
        token_page=tokenizer(page)
        nbre_token_fichier+=len(token_page)

        if position + len(token_page) <= capacite:
            tableau[position:position+len(token_page)]=token_page
            position+=len(token_page)
        else:
            free_case_tableau=capacite - position
            tableau[position:capacite]=token_page[:free_case_tableau]
            if compteur==0:
                chemin_shards=dossier_shards/f"fineweb_val_{compteur:06d}.npy"
            else:
                chemin_shards=dossier_shards/f"fineweb_train_{compteur:06d}.npy"

            np.save(chemin_shards,tableau)
            print(chemin_shards.name)
            compteur+=1
            reste_page_token=token_page[free_case_tableau:]
            tableau[0:len(reste_page_token)]=reste_page_token
            position=len(reste_page_token)
    total_token+=nbre_token_fichier
    print(f"nom du fichier:{nom_fichier} | nombre de page:{len(pages)} | nbre de token: {nbre_token_fichier}")

    # vider la memoire
    del df
    del pages


if position>0:
    chemin_shards=dossier_shards/f"fineweb_train_{compteur:06d}.npy"
    np.save(chemin_shards,tableau[:position])
    print(chemin_shards.name)
print(f"total de token= {total_token}")












