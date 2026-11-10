from pathlib import Path

from huggingface_hub import hf_hub_download

for num in range(4):
    nom_fichier=f"sample/10BT/{num:03d}_00000.parquet"

    chemin = hf_hub_download(
        repo_id="HuggingFaceFW/fineweb-edu",# nom depot
        repo_type="dataset",# jeu de données
        filename=nom_fichier,  # fichier voulu dans le depot
        local_dir=Path(__file__).parent/"data"/"fineweb_edu",# save dans fineweb_edu
    )
    print(chemin)
