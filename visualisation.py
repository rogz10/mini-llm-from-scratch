import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def projeter_embedding(poids):
    donnees=poids.detach().cpu().numpy()

    projection=PCA(n_components=2)
    coordonnees=projection.fit_transform(donnees)

    return coordonnees
def afficher_embedding(points):

    plt.figure(figsize=(8,6))
    plt.scatter(points[:,0],points[:,1])
    plt.xlabel("composante principale 1")
    plt.ylabel("composante principale 2")
    plt.text(points[55,0],points[55,1],"b")
    plt.title("embedding ")
    plt.grid(True)
    plt.show()
