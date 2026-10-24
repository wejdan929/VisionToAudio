import numpy as np
import sounddevice as sd
import time

# --- PARAMÈTRES AUDIO ---
SAMPLE_RATE = 44100     # Fréquence d'échantillonnage (Hz)
COLUMN_DURATION = 0.3  # Durée par colonne (50 millisecondes)
NUM_ROWS = 16           # Hauteur (nombre de fréquences)
NUM_COLS = 5            # Largeur (nombre de colonnes / étapes de balayage)

FREQ_MIN = 200.0        # Hz (Pixel en bas)
FREQ_MAX = 4000.0       # Hz (Pixel en haut)

def generate_stereo_column(column_data, x_position, total_cols):
    """
    Génère un signal stéréo (2 canaux) pour une colonne donnée, 
    avec un panoramique basé sur la position horizontale x_position.
    """
    num_samples = int(SAMPLE_RATE * COLUMN_DURATION)
    t = np.linspace(0, COLUMN_DURATION, num_samples, endpoint=False)
    
    # Signal mono (somme des sinusoïdes des pixels)
    mono_signal = np.zeros(num_samples)
    frequencies = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), NUM_ROWS)

    for y, brightness in enumerate(column_data):
        if brightness > 0:
            amplitude = (brightness / 255.0) * 0.05  # Volume modéré
            freq = frequencies[NUM_ROWS - 1 - y]      # Fréquence selon la hauteur Y
            mono_signal += amplitude * np.sin(2 * np.pi * freq * t)

    # Application d'une enveloppe de fondu (fade in / fade out) pour éviter les clics audio
    envelope = np.hanning(num_samples)
    mono_signal *= envelope

    # Calcul du panoramique (Panning Stéréo de gauche à droite)
    # x_ratio varie de 0.0 (tout à gauche) à 1.0 (tout à droite)
    x_ratio = x_position / (total_cols - 1) if total_cols > 1 else 0.5
    
    # Loi de répartition du volume (Pan constant)
    gain_left = np.cos(x_ratio * np.pi / 2)
    gain_right = np.sin(x_ratio * np.pi / 2)

    # Création du buffer Stéréo (Canal Gauche, Canal Droit)
    stereo_frame = np.column_stack((mono_signal * gain_left, mono_signal * gain_right))
    return stereo_frame

if __name__ == "__main__":
    print("--- Test du Balayage Stéréo Continu ---")
    
    # Création d'une fausse "image" de 16 lignes par 5 colonnes
    # On place un point lumineux qui se déplace de gauche à droite
    fake_image = np.zeros((NUM_ROWS, NUM_COLS), dtype=np.uint8)
    fake_image[4, 0] = 255  # Colonne 0 : Point en haut-gauche
    fake_image[8, 2] = 255  # Colonne 2 : Point au milieu
    fake_image[12, 4] = 255 # Colonne 4 : Point en bas-droite

    print("Démarrage du balayage sonore de gauche à droite...")
    
    # Lecture colonne par colonne en temps réel simulé
    for col_idx in range(NUM_COLS):
        column_data = fake_image[:, col_idx]
        
        # Génération du son pour cette colonne
        frame_audio = generate_stereo_column(column_data, col_idx, NUM_COLS)
        
        print(f"Balayage colonne {col_idx + 1}/{NUM_COLS} (Panning x = {col_idx})")
        sd.play(frame_audio, SAMPLE_RATE)
        sd.wait()  # Attend la fin de la lecture de la colonne avant la suivante

    print("Étape 2 terminée avec succès ! Le son s'est déplacé en stéréo.")