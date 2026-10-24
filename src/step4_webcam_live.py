import cv2
import numpy as np
import sounddevice as sd
import queue
import threading

# --- PARAMÈTRES ---
SAMPLE_RATE = 44100
COLUMN_DURATION = 0.04  # Durée par colonne (40ms)
GRID_WIDTH = 16
GRID_HEIGHT = 16

FREQ_MIN = 200.0
FREQ_MAX = 4000.0

def audio_worker(audio_queue):
    """
    Thread dédié uniquement à la lecture audio pour éviter les blocages de la vidéo.
    """
    frequencies = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), GRID_HEIGHT)
    num_samples = int(SAMPLE_RATE * COLUMN_DURATION)
    t = np.linspace(0, COLUMN_DURATION, num_samples, endpoint=False)
    envelope = np.hanning(num_samples)

    while True:
        grid = audio_queue.get()
        if grid is None:
            break  # Signal d'arrêt

        # Balayage de la grille colonne par colonne
        for x in range(GRID_WIDTH):
            column_data = grid[:, x]
            mono_signal = np.zeros(num_samples)

            for y, brightness in enumerate(column_data):
                if brightness > 0:
                    amplitude = (brightness / 255.0) * 0.08  # Amplitude augmentée pour bien entendre
                    freq = frequencies[GRID_HEIGHT - 1 - y]
                    mono_signal += amplitude * np.sin(2 * np.pi * freq * t)

            mono_signal *= envelope

            # Panning stéréo
            x_ratio = x / (GRID_WIDTH - 1) if GRID_WIDTH > 1 else 0.5
            gain_left = np.cos(x_ratio * np.pi / 2)
            gain_right = np.sin(x_ratio * np.pi / 2)

            stereo_frame = np.column_stack((mono_signal * gain_left, mono_signal * gain_right))
            
            sd.play(stereo_frame, SAMPLE_RATE)
            sd.wait()

        print("🔊 Image convertie et jouée en son !")
        audio_queue.task_done()

if __name__ == "__main__":
    print("--- Étape 4 : Substitution Sensorielle en Direct (Webcam) ---")
    print("Ouverture de la webcam... Appuie sur la touche 'q' dans la fenêtre vidéo pour quitter.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam.")
        exit()

    # File d'attente pour passer les grilles du thread vidéo au thread audio
    audio_queue = queue.Queue(maxsize=2)

    # Démarrage du thread audio
    t_audio = threading.Thread(target=audio_worker, args=(audio_queue,), daemon=True)
    t_audio.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur de lecture du flux vidéo.")
                break

            # 1. Miroir pour un effet plus naturel
            frame = cv2.flip(frame, 1)

            # 2. Conversion en niveaux de gris
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 3. Filtre de Canny pour les contours
            edges = cv2.Canny(gray, 80, 160)

            # 4. Redimensionnement en grille 16x16
            grid = cv2.resize(edges, (GRID_WIDTH, GRID_HEIGHT), interpolation=cv2.INTER_AREA)

            # Affichage de la vidéo de retour
            cv2.imshow("Vision-to-Audio (Webcam Live)", edges)

            # Envoyer la grille au thread audio (si la file n'est pas pleine)
            if audio_queue.empty():
                audio_queue.put(grid)

            # Quitter si l'utilisateur appuie sur 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        audio_queue.put(None)  # Arrêt propre du thread audio
        print("Programme arrêté proprement.")