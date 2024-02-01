import cupy as cp
import numpy as np
import math
import pickle
import cv2
from sklearn.feature_selection import SelectPercentile, f_classif


class ViolaJones:
    def __init__(self, T = 10):
        """
          Args:
            T: Le nombre de classifieurs faibles à utiliser.
        """
        self.T = T
        self.alphas = []
        self.clfs = []
        
     #Utilisation de cupy pour GPU
    def integral_image(self, image):
        ii = cp.zeros(image.shape)
        s = cp.zeros(image.shape)
        for y in range(len(image)):
            for x in range(len(image[y])):
                s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
                ii[y][x] = ii[y][x-1] + s[y][x] if x-1 >= 0 else s[y][x]
        return ii

    def train(self, training_data, pos_num, neg_num):
        # Initialisation et normalisation des poids pour les données d'entraînement
        weights = cp.zeros(len(training_data))
        for idx, (image, label) in enumerate(training_data):
            integral_image_data = self.integral_image(image)
            training_data[idx] = (integral_image_data, label)
            weights[idx] = 1.0 / (2 * pos_num) if label == 1 else 1.0 / (2 * neg_num)

        sample_image = training_data[0][0]
        features = self.build_features(sample_image.shape)
        X, y = self.apply_features(features, training_data)

        # Boucle d'entraînement pour créer les classificateurs faibles
        for t in range(self.T):
            weights = weights / cp.linalg.norm(weights)
            weak_classifiers = self.train_weak(X, y, features, weights)
            clf, error, accuracy = self.select_best(weak_classifiers, weights, training_data)
            beta = 1e-10 if error == 0 else error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = cp.log(1.0 / beta) if beta != 1e-10 else 0  # Prevent log(inf)
            self.alphas.append(alpha)
            self.clfs.append(clf)


    def gpu_feature_selection(self, X, y):
        # Sélection des caractéristiques basée sur la variance pour les données X
        variances = cp.var(X, axis=1)
        threshold = variances.mean() # Seuil basé sur la moyenne des variances
        selected_indices = cp.where(variances > threshold)[0]
        X_selected = X[selected_indices, :]
        return X_selected, selected_indices

    
    #Trouve les seuils optimaux pour chaque classificateur faible en fonction des poids actuels
    def train_weak(self, X, y, features, weights):
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w

        classifiers = []
        total_features = X.shape[0]
        # Boucle sur chaque caractéristique pour entraîner des classificateurs faibles
        for index, feature in enumerate(X):
            if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
                print(f"Trained {len(classifiers)} classifiers out of {total_features}")

            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])

            # Initialisation des variables pour l'erreur minimale et la meilleure caractéristique
            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1

                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w

            # Vérifier si une caractéristique valide a été trouvée et l'ajouter aux classificateurs
            if best_feature is not None:
                clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
                classifiers.append(clf)
        
        return classifiers
        
    #Construit les caractéristiques possibles données une forme d'image
    def build_features(self, image_shape):
        height, width = image_shape
        features = []
        # Boucle sur toutes les combinaisons possibles de largeur et hauteur pour créer des caractéristiques
        for w in range(1, width+1):
            for h in range(1, height+1):
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        # Caractéristiques rectangulaires
                        immediate = RectangleRegion(i, j, w, h)
                        right = RectangleRegion(i+w, j, w, h)
                        if i + 2 * w < width: # Adjacentes horizontalement
                            features.append(([right], [immediate]))

                        bottom = RectangleRegion(i, j+h, w, h)
                        if j + 2 * h < height: # Adjacentes verticalement
                            features.append(([immediate], [bottom]))

                        right_2 = RectangleRegion(i+2*w, j, w, h)
                        #3 rectangle features
                        if i + 3 * w < width: # Adjacentes horizontalement
                            features.append(([right], [right_2, immediate]))

                        bottom_2 = RectangleRegion(i, j+2*h, w, h)
                        if j + 3 * h < height: # Adjacentes verticalement
                            features.append(([bottom], [bottom_2, immediate]))

                        # Caractéristiques de 4 rectangles
                        bottom_right = RectangleRegion(i+w, j+h, w, h)
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(([right, bottom], [immediate, bottom_right]))

                        j += 1
                    i += 1
        return np.array(features, dtype=object)

    #Sélectionne le meilleur classificateur faible pour les poids donnés
    def select_best(self, classifiers, weights, training_data):
        best_clf, best_error, best_accuracy = None, float('inf'), []
        # Boucle sur chaque classificateur pour trouver le meilleur en fonction de l'erreur
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.classify(data[0]) - data[1])
                accuracy.append(correctness)
                error += w * correctness
            error /= len(training_data)

            if error < best_error and error != 0 and error < 1:  # Gamme d'erreur valide
                best_clf, best_error, best_accuracy = clf, error, accuracy

        # Gestion du cas où aucun classificateur n'est trouvé
        if best_clf is None:
            return None, 1, [0] * len(training_data)  # Classificateur par défaut
        return best_clf, best_error, best_accuracy

    # Application des caractéristiques sur les données d'entraînement
    def apply_features(self, features, training_data): 
        X = cp.zeros((len(features), len(training_data)))
        y = cp.array([data[1] for data in training_data])
        i = 0
        for positive_regions, negative_regions in features:
            # Calcul de la valeur de la caractéristique pour chaque exemple d'entraînement
            feature = lambda ii: cp.sum(cp.array([pos.compute_feature(ii, pos) for pos in positive_regions])) \
                                 - cp.sum(cp.array([neg.compute_feature(ii, neg) for neg in negative_regions]))
            X[i] = cp.array(list(map(lambda data: feature(data[0]), training_data)))
            i += 1
        return X, y

    # Classification optimisée avec détection de coordonnées sur une image
    def classify_with_coordinates_optimized(self, image):
        integral_img = self.integral_image(image)
        window_size = (500, 500)  # Taille de fenêtre
        for scale in np.arange(0.5, 1.5, 0.1):  # Échelle de fenêtre de 0.5x à 1.5x
            scaled_window = (int(window_size[0] * scale), int(window_size[1] * scale))
            for y in range(0, image.shape[0] - scaled_window[1], 4):  # Pas de 4
                for x in range(0, image.shape[1] - scaled_window[0], 4):
                    sub_img = integral_img[y:y+scaled_window[1], x:x+scaled_window[0]]
                    detected = self.classify(sub_img)
                    if detected:
                        return True, (x, y, scaled_window[0], scaled_window[1])
        return False, None

    # Extraction d'une région spécifique de l'image intégrale
    def extract_integral_region(self, integral_img, x, y, window_size):
        x_end, y_end = x + window_size[1], y + window_size[0]
        return integral_img[y_end, x_end] - integral_img[y, x_end] - integral_img[y_end, x] + integral_img[y, x]
    
    # Classification de l'image en utilisant les classificateurs faibles et leurs poids
    def classify(self, image):
        total = 0
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(image)
        return 1 if total >= 0.57 * sum(self.alphas) else 0
    
    # Calcul de l'image intégrale pour une image donnée
    def integral_image(self, image):
        ii = cp.zeros(image.shape)
        s = cp.zeros(image.shape)
        for y in range(len(image)):
            for x in range(len(image[y])):
                s = cp.cumsum(image, axis=0)
                ii = cp.cumsum(s, axis=1)
        return ii

    # Sauvegarde du classificateur dans un fichier pickle
    def save(self, filename):
        """
        Saves the classifier to a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

            
    @staticmethod
    # Chargement d'un classificateur à partir d'un fichier pickle
    def load(filename):
        """
        A static method which loads the classifier from a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)

class WeakClassifier:
    # Initialisation d'un classificateur faible avec des régions positives, négatives, un seuil et une polarité
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        """
        Args:
            positive_regions : Un tableau de régions de rectangle qui contribuent positivement à la caractéristique.
            negative_regions : Un tableau de régions de rectangle qui contribuent négativement à la caractéristique.
            threshold : Le seuil du classifieur faible.
            polarity : La polarité du classifieur faible.
        """
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity
    
    def classify(self, x):
        # Vérifier si les régions positives et négatives sont définies
        if self.positive_regions is None or self.negative_regions is None:
            return 0  # Ou une classification par défaut si les régions ne sont pas définies

        # Calculer la somme des caractéristiques pour les régions positives et négatives
        positive_sum = cp.sum(cp.array([region.compute_feature(x, region) for region in self.positive_regions]))
        negative_sum = cp.sum(cp.array([region.compute_feature(x, region) for region in self.negative_regions]))

        # Calculer la caractéristique finale
        feature_value = positive_sum - negative_sum

        # Classification basée sur la polarité et le seuil
        return 1 if self.polarity * feature_value < self.polarity * self.threshold else 0

    # Représentation en chaîne de caractères du classificateur faible
    def __str__(self):
        return f"Clf faible (threshold={self.threshold}, polarity={self.polarity}, {str(self.positive_regions)}, {str(self.negative_regions)})"

    def __str__(self):
        return "Clf faible (threshold=%d, polarity=%d, %s, %s" % (self.threshold, self.polarity, str(self.positive_regions), str(self.negative_regions))
    
class RectangleRegion:
    # Initialisation d'une région rectangulaire avec des coordonnées et dimensions
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    # Calcul de la caractéristique d'une région dans une image intégrale
    def compute_feature(self, integral_image, region):
        # Extraire les coordonnées de la région
        top_left = (region.x, region.y)
        top_right = (region.x + region.width, region.y)
        bottom_left = (region.x, region.y + region.height)
        bottom_right = (region.x + region.width, region.y + region.height)

        # Vérifier si les coordonnées de la fenêtre sont dans les limites
        if (
            0 <= top_left[0] < integral_image.shape[1]
            and 0 <= top_left[1] < integral_image.shape[0]
            and 0 <= top_right[0] < integral_image.shape[1]
            and 0 <= top_right[1] < integral_image.shape[0]
            and 0 <= bottom_left[0] < integral_image.shape[1]
            and 0 <= bottom_left[1] < integral_image.shape[0]
            and 0 <= bottom_right[0] < integral_image.shape[1]
            and 0 <= bottom_right[1] < integral_image.shape[0]
        ):
            # Toutes les coordonnées de la fenêtre sont dans les limites, on procède au calcul de l'image intégrale
            sum_region = (
                integral_image[bottom_right[1]][bottom_right[0]]
                - integral_image[top_right[1]][top_right[0]]
                - integral_image[bottom_left[1]][bottom_left[0]]
                + integral_image[top_left[1]][top_left[0]]
            )
            return sum_region
        else:
            return 0  

    
    def integral_image(image):
        try:
            height, width = image.shape
        except ValueError:
            # Gère les erreurs de traitement de l'image de manière détaillée
            print("Error with image processing.")
            print(f"Image shape: {image.shape}")
            print(f"Image type: {type(image)}")
            print(f"Image content: {image}")
            raise  # Re-lève l'exception pour une gestion normale des erreurs

         # Calcule l'image intégrale
        ii = cp.zeros((height, width))
        s = cp.zeros((height, width))

        for y in range(height):
            for x in range(width):
                s[y][x] = s[y-1][x] + image[y][x].item() if y-1 >= 0 else image[y][x].item()
                ii[y][x] = ii[y][x-1] + s[y][x] if x-1 >= 0 else s[y][x]
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