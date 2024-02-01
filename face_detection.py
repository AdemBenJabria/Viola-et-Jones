import cupy as cp
import pickle
from viola_jones import ViolaJones
from cascade import CascadeClassifier
import time

def train_viola(t):
    with open("training.pkl", 'rb') as f:
        training = pickle.load(f)
    
    # Créer un tableau CuPy pour stocker les images d'entraînement
    images_array = cp.array([image for image, _ in training])

    # Entraîner le classifieur Viola-Jones
    clf = ViolaJones(T=t)
    clf.train(images_array, 2429, 4548)
    evaluate(clf, training)
    clf.save(str(t))

def test_viola(filename):
    with open("test.pkl", 'rb') as f:
        test = pickle.load(f)
    
    clf = ViolaJones.load(filename)
    evaluate(clf, test)

def train_cascade(layers, filename="Cascade"):
    with open("training.pkl", 'rb') as f:
        training = pickle.load(f)
    
    # Créer un tableau CuPy pour stocker les images d'entraînement
    images_array = cp.array([image for image, _ in training])

    clf = CascadeClassifier(layers)
    clf.train(images_array)
    evaluate(clf, training)
    clf.save(filename)

def test_cascade(filename="Cascade"):
    with open("test.pkl", "rb") as f:
        test = pickle.load(f)
    
    clf = CascadeClassifier.load(filename)
    evaluate(clf, test)

def evaluate(clf, data):
    correct = 0
    all_negatives, all_positives = 0, 0
    true_negatives, false_negatives = 0, 0
    true_positives, false_positives = 0, 0
    classification_time = 0

    for x, y in data:
        if y == 1:
            all_positives += 1
        else:
            all_negatives += 1

        start = time.time()
        prediction = clf.classify(x)
        classification_time += time.time() - start
        if prediction == 1 and y == 0:
            false_positives += 1
        if prediction == 0 and y == 1:
            false_negatives += 1
        
        correct += 1 if prediction == y else 0
    
    print("Taux de faux positifs: %d/%d (%f)" % (false_positives, all_negatives, false_positives/all_negatives))
    print("Taux de faux négatifs: %d/%d (%f)" % (false_negatives, all_positives, false_negatives/all_positives))
    print("Précision: %d/%d (%f)" % (correct, len(data), correct/len(data)))
    print("Temps de classification moyen: %f" % (classification_time / len(data)))

def integral_image(image):
    """
    Calcule la représentation de l'image intégrale d'une image. L'image intégrale est définie comme suit :
    1. s(x, y) = s(x, y-1) + i(x, y), s(x, -1) = 0
    2. ii(x, y) = ii(x-1, y) + s(x, y), ii(-1, y) = 0
    Où s(x, y) est la somme cumulative des lignes, ii(x, y) est l'image intégrale, et i(x, y) est l'image originale.
    L'image intégrale est la somme de tous les pixels au-dessus et à gauche du pixel actuel.
    Args :
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