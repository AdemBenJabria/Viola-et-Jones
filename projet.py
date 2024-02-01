import cupy as cp
import pickle
from viola_jones import ViolaJones
from cascade import CascadeClassifier
import time
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython.display import display

# Crée un sprite sheet à partir des images en 19x19
def create_sprite_sheet(training, num_cols=100, image_size=(19, 19)):
    num_images = len(training) # Nombre total d'images dans l'ensemble d'entraînement
    num_rows = int(np.ceil(num_images / num_cols)) # Calcule le nombre de lignes nécessaires

    # Calcule la taille du spritesheet
    sheet_height = num_rows * image_size[0] 
    sheet_width = num_cols * image_size[1]

    sprite_sheet = cp.zeros((sheet_height, sheet_width)) # Crée une matrice vide

    for idx, (image, label) in enumerate(training):
        row = idx // num_cols # Calcule la ligne actuelle pour placer l'image
        col = idx % num_cols 
        # Calcule la position sur la feuille de sprites pour placer l'image
        y = row * image_size[0]
        x = col * image_size[1]

        sprite_sheet[y:y + image_size[0], x:x + image_size[1]] = cp.array(image) # Remplit la zone avec l'image

    return sprite_sheet

# Extrait des images à partir de la spritesheet 
def extract_images(sprite_sheet, num_cols=100, image_size=(19, 19)):
    # Calcule le nombre d'images sur la feuille de sprites
    num_images = sprite_sheet.shape[0] // image_size[0] * sprite_sheet.shape[1] // image_size[1]
    images = [] # Initialise une liste pour stocker les images extraites

    for idx in range(num_images):
        # Calcule la position de la région du sprite sheet contenant l'image
        row = (idx // num_cols) * image_size[0]
        col = (idx % num_cols) * image_size[1]
        # Extrait l'image à partir de la région du sprite sheet
        image = sprite_sheet[row:row + image_size[0], col:col + image_size[1]]
        images.append(image)

    return images


def bench_train():
    with open("training.pkl", 'rb') as f:
        training = pickle.load(f)

    sprite_sheet = create_sprite_sheet(training) #crée la sprite sheet
    extracted_images = extract_images(sprite_sheet) #extrait les images de la sprite sheet
    
    pos_num = sum(label for _, label in training) # Calcule le nombre d'exemples positifs dans les données d'entraînement
    training_data = [(img, label) for img, label in zip(extracted_images, [1 if idx < pos_num else 0 for idx in range(len(extracted_images))])] # Prépare les données d'entraînement pour le classifieur 
    
    clf = ViolaJones(T=200)
    clf.train(training_data, pos_num, len(training) - pos_num) 
    clf.save("200")


# Méthode pour entraîner le classifieur V-J
def train(self, training_data, pos_num, neg_num):
    num_images = len(training_data) # Nombre total d'images dans les données d'entraînement
    weights = np.zeros(num_images)

    print("Calcul des images intégrales")
    for idx, (image, label) in enumerate(training_data):
        # Calcule l'image intégrale à partir de l'image originale
        integral_image_data = self.integral_image(cp.array(image))
        training_data[idx] = (integral_image_data, label)

    sample_image = training_data[0][0] # Prend une image d'exemple pour la taille
    print("Construction des caractéristiques")
    features = self.build_features(sample_image.shape) # Construit les caractéristiques du classifieur
    print("Application des caractéristiques aux exemples d'entraînement")
    X, y = self.apply_features(features, training_data)
    print("Sélection des meilleures caractéristiques")
    indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True)
    X = X[indices]
    features = features[indices]
    print(f"{len(X)} caractéristiques potentielles sélectionnées")

    

def bench_train_fast(subset_size=5, T=3): #5 & 3 pour que ce soit plus rapide
    with open("training.pkl", 'rb') as f:
        training = pickle.load(f)

    # Génère des indices aléatoires pour sélectionner un sous-ensemble des données d'entraînement
    random_indices = cp.random.randint(len(training), size=subset_size).get()
    training_subset = [training[i] for i in random_indices]

    # Extrait les images et les étiquettes du sous-ensemble
    training_data = [(cp.array(img), label) for img, label in training_subset]

    # Initialise Viola-Jones avec moins de classifieurs faibles
    clf = ViolaJones(T=T)
    pos_num = sum(label for _, label in training_subset)
    neg_num = subset_size - pos_num

    # Entraîne le classifieur avec le sous-ensemble
    clf.train(training_data, pos_num, neg_num)

    # Enregistre le classifieur entraîné
    clf.save("200_fast")

    return clf


def evaluate_combined(clf, test_data):
    # Initialisation des métriques d'évaluation
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    total_time = 0
    all_negatives, all_positives = 0, 0

    # Itération sur les données de test pour faire des prédictions
    for image, label in test_data:
        start_time = time.time()
        prediction = clf.classify(image)
        total_time += time.time() - start_time # Calcul du temps total de classification

        # Mise à jour des métriques en fonction des prédictions
        if label == 1:
            all_positives += 1
            if prediction == 1:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            all_negatives += 1
            if prediction == 1:
                false_positives += 1
            else:
                true_negatives += 1

    # Calcul des différentes métriques
    accuracy = (true_positives + true_negatives) / len(test_data)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_classification_time = total_time / len(test_data)
    false_positive_rate = false_positives / all_negatives if all_negatives > 0 else 0
    false_negative_rate = false_negatives / all_positives if all_positives > 0 else 0

    # Affichage des résultats
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Précision: {:.2f}".format(precision))
    print("Rappel: {:.2f}".format(recall))
    print("Score F1: {:.2f}".format(f1_score))
    print("Temps moyen de classification: {:.4f} secondes".format(avg_classification_time))
    print("Taux de faux positifs: {:.2f}".format(false_positive_rate))
    print("Taux de faux négatifs: {:.2f}".format(false_negative_rate))

    
def bench_accuracy():
    with open("test.pkl", "rb") as f:
        test = pickle.load(f)
    
    clf = ViolaJones.load("200")
    print("Évaluation du modèle standard:")
    evaluate_combined(clf, test)

    clf = ViolaJones.load("200_fast")
    print("\nÉvaluation du modèle rapide:")
    evaluate_combined(clf, test)

#import os
from PIL import Image

def find_face(image_source):
    print("Chargement du modèle...")
    # Charger le modèle
    clf = ViolaJones.load("200")

    # Lire l'image originale et la convertir en niveaux de gris
    print("Lecture et conversion de l'image...")
    if isinstance(image_source, np.ndarray):
        original_image = image_source.copy()
        gray_image = cv2.cvtColor(image_source, cv2.COLOR_BGR2GRAY)
    else:
        original_image = cv2.imread(image_source)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Convertir l'image en tableau CuPy
    print("Conversion de l'image en tableau CuPy...")
    image_array = cp.array(gray_image)

    # Classifier l'image et obtenir les coordonnées du visage
    print("Classification de l'image...")
    detected, coordinates = clf.classify_with_coordinates_optimized(image_array)

    # Si un visage est détecté, marquer la zone du visage sur l'image en couleur
    if detected:
        print("DETECTED")
        x, y, w, h = coordinates
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite("src/yes.png", original_image)
        #os.startfile("C:/Users/ademb/Documents/viola jones/yes.jpg")
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.show()
    else:
        print("Aucun visage détecté")
        

def integral_image(image):
    """
    Calcule la représentation de l'image intégrale d'une image. L'image intégrale est définie comme suit :
    1. s(x, y) = s(x, y-1) + i(x, y), s(x, -1) = 0
    2. ii(x, y) = ii(x-1, y) + s(x, y), ii(-1, y) = 0
    Où s(x, y) est la somme cumulative des lignes, ii(x, y) est l'image intégrale, et i(x, y) est l'image originale.
    L'image intégrale est la somme de tous les pixels au-dessus et à gauche du pixel actuel.
    Args:
        image : un tableau numpy avec une forme (m, n)
    """
    ii = cp.zeros(image.shape)
    s = cp.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii

def scan_inclusive(array):
    """
    Effectue une analyse inclusive sur le tableau donné.
    Args : array : Un tableau CuPy de forme (m, n). 
    Returns : Un tableau CuPy de forme (m, n) contenant l'analyse inclusive du tableau d'entrée.
    """

    return cp.cumsum(array, axis=0)

def transpose_matrix(array):
    """
    Transpose la matrice donnée.
    Args : array : Un tableau CuPy de forme (m, n).
    Renvoie : Un tableau CuPy de forme (n, m) contenant la transposée du tableau d'entrée.
    """

    return cp.transpose(array)

if __name__ == "__main__":
    import argparse

    # Crée un analyseur d'arguments de ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", action="store_true", help="Entraîner le classifieur Viola-Jones")
    parser.add_argument("-train-fast", action="store_true", help="Entraîner le classifieur Viola-Jones en utilisant une feuille de sprite")
    parser.add_argument("-accuracy", action="store_true", help="Évaluer la précision du classifieur Viola-Jones")
    parser.add_argument("-find", action="store_true", help="Trouver des visages dans une image")
    parser.add_argument("image_source", nargs="?", help="Chemin de l'image pour trouver des visages")
    args = parser.parse_args()

    # Exécute différentes fonctions en fonction des options fournies
    if args.train:
        bench_train()
    elif args.train_fast:
        bench_train_fast()
    elif args.accuracy:
        bench_accuracy()
    elif args.find:
        find_face(args.image_source)
    else:
        print("Option non valide. Veuillez utiliser -train, -train-fast, -accuracy, ou -find.")