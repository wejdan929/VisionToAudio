import cv2
import numpy as np
import sounddevice as sd
import queue
import threading
from collections import deque

# --- PARAMÈTRES GLOBAUX ---
SAMPLE_RATE = 44100
DURATION = 0.05  # Courte durée pour un retour ultra-réactif au mouvement
FREQ_MIN = 200.0
FREQ_MAX = 4000.0

def nothing(x):
    pass

def audio_worker(audio_queue):
    """Thread audio dédié aux sons générés par la position du stylo violet"""
    while True:
        item = audio_queue.get()
        if item is None:
            break
        
        y_pos_norm, x_pos_norm, volume = item # y et x normalisés entre 0 et 1

        # Calcul de la fréquence selon la hauteur (Haut = aigu, Bas = grave)
        freq = FREQ_MIN + (1.0 - y_pos_norm) * (FREQ_MAX - FREQ_MIN)
        
        num_samples = int(SAMPLE_RATE * DURATION)
        t = np.linspace(0, DURATION, num_samples, endpoint=False)
        envelope = np.hanning(num_samples)
        
        # Onde sinusoïdale modulée par la position
        mono_signal = volume * np.sin(2 * np.pi * freq * t) * envelope

        # Panning stéréo selon la position gauche/droite
        gain_left = np.cos(x_pos_norm * np.pi / 2)
        gain_right = np.sin(x_pos_norm * np.pi / 2)

        stereo_frame = np.column_stack((mono_signal * gain_left, mono_signal * gain_right))
        
        sd.play(stereo_frame, SAMPLE_RATE)
        sd.wait()
        audio_queue.task_done()

if __name__ == "__main__":
    print("--- Mode Baguette Magique / Air-Writing (Stylo Violet) ---")
    print("Bouge ton objet violet devant la webcam pour dessiner et produire du son en direct !")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam.")
        exit()

    window_name = "Vision-to-Audio : Air-Writing Magic Wand"
    cv2.namedWindow(window_name)

    # Curseur de volume
    cv2.createTrackbar("Volume", window_name, 10, 20, nothing)

    # File d'attente pour le son
    audio_queue = queue.Queue(maxsize=2)
    t_audio = threading.Thread(target=audio_worker, args=(audio_queue,), daemon=True)
    t_audio.start()

    # Historique pour tracer la ligne magique (les 64 derniers points)
    pts = deque(maxlen=64)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # 1. Conversion en espace HSV pour détecter le VIOLET facilement
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Plage de couleur pour le violet/magenta (à ajuster si ton violet est très clair/foncé)
            # Teinte (H) pour le violet tourne autour de 130-160
            lower_purple = np.array([120, 50, 50])
            upper_purple = np.array([165, 255, 255])

            mask = cv2.inRange(hsv, lower_purple, upper_purple)
            # Nettoyage du masque (éliminer le bruit)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # 2. Trouver le centre de l'objet violet
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            center = None

            if len(contours) > 0:
                # Trouver le plus grand contour violet
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    # Si l'objet est assez grand pour être un stylo valide
                    if radius > 5:
                        cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 255), 2)
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)
                        
                        # Ajouter le point à l'historique de dessin
                        pts.append(center)

                        # Normaliser les coordonnées (0.0 à 1.0)
                        x_norm = center[0] / w
                        y_norm = center[1] / h

                        vol_slider = cv2.getTrackbarPos("Volume", window_name)
                        volume_factor = vol_slider / 100.0

                        # Envoyer au thread audio
                        if audio_queue.empty():
                            audio_queue.put((y_norm, x_norm, volume_factor))
            else:
                # Si on ne voit plus le stylo, on ajoute un point vide pour couper la ligne
                pts.append(None)

            # 3. Dessiner la traînée magique (Air-Writing Trail) sur la vidéo
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                # Épaisseur et couleur néon (Violet/Cyan)
                thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (255, 0, 255), thickness)

            # Texte explicatif incrusté dans le style HUD futuriste
            cv2.putText(frame, "MODE: Air-Writing Magic Wand (Stylo Violet)", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "Dessinez dans l'air pour creer du son et des symboles !", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        audio_queue.put(None)
        print("Programme arrêté proprement.")