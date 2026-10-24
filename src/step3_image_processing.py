import cv2
import numpy as np
import sounddevice as sd
import time

# --- PARAMÈTRES ---
SAMPLE_RATE = 44100
COLUMN_DURATION = 0.08  # Durée par colonne
GRID_WIDTH = 16         # Largeur réduite de l'image (16 colonnes)
GRID_HEIGHT = 16        # Hauteur réduite de l'image (16 lignes)

FREQ_MIN = 200.0        # Hz (Bas)
FREQ_MAX = 4000.0       # Hz (Haut)

def process_image_to_grid(image_path):
    """
    Charge une image, la convertit en niveaux de gris, 
    applique un filtre de contours (Canny) et la redimensionne 
    en une grille 16x16.
    """
    # 1. Charger l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erreur : Impossible de charger l'image {image_path}. Création d'une image de test...")
        # Image de secours : un cercle blanc sur fond noir
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 30, 255, -1)

    # 2. Appliquer le détecteur de contours de Canny
    edges = cv2.Canny(img, 100, 200)

    # 3. Redimensionner l'image de contours à la taille de notre grille audio (16x16)
    grid = cv2.resize(edges, (GRID_WIDTH, GRID_HEIGHT), interpolation=cv2.INTER_AREA)
    
    return grid

def play_grid_audio(grid):
    """
    Lit la grille colonne par colonne en générant le son stéréo associé.
    """
    num_cols = grid.shape[1]
    num_rows = grid.shape[0]
    frequencies = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), num_rows)
    num_samples = int(SAMPLE_RATE * COLUMN_DURATION)
    t = np.linspace(0, COLUMN_DURATION, num_samples, endpoint=False)
    envelope = np.hanning(num_samples)

    print(f"Lancement du balayage sonore de l'image ({num_cols} colonnes)...")

    for x in range(num_cols):
        column_data = grid[:, x]
        mono_signal = np.zeros(num_samples)

        for y, brightness in enumerate(column_data):
            if brightness > 0:
                amplitude = (brightness / 255.0) * 0.05
                freq = frequencies[num_rows - 1 - y]
                mono_signal += amplitude * np.sin(2 * np.pi * freq * t)

        mono_signal *= envelope

        # Panning stéréo
        x_ratio = x / (num_cols - 1) if num_cols > 1 else 0.5
        gain_left = np.cos(x_ratio * np.pi / 2)
        gain_right = np.sin(x_ratio * np.pi / 2)

        stereo_frame = np.column_stack((mono_signal * gain_left, mono_signal * gain_right))
        
        # Lecture de la colonne
        sd.play(stereo_frame, SAMPLE_RATE)
        sd.wait()

if __name__ == "__main__":
    print("--- Étape 3 : Traitement d'image OpenCV & Son ---")
    
    # On traite une image (si elle n'existe pas, la fonction en génère une automatiquement)
    image_path = "assets/test_shape.png"
    
    # Petite astuce : si l'image n'est pas dans assets, on en crée une propre pour tester
    dummy_img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(dummy_img, (50, 50), (150, 150), 255, 3) # Un carré blanc
    cv2.imwrite(image_path, dummy_img)

    # Traitement de l'image
    grid_matrix = process_image_to_grid(image_path)
    
    # Conversion en son
    play_grid_audio(grid_matrix)
    print("Étape 3 terminée avec succès !")