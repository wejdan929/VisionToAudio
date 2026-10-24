import cv2
import pyttsx3
from ultralytics import YOLO
import threading
import time

# --- DICTIONNAIRE DE TRADUCTION ANGLAIS -> FRANÇAIS ---
TRANSLATIONS = {
    "person": "une personne",
    "bottle": "une bouteille",
    "cup": "une tasse",
    "cell phone": "un téléphone",
    "mobile phone": "un téléphone",
    "chair": "une chaise",
    "laptop": "un ordinateur",
    "book": "un livre",
    "mouse": "une souris",
    "keyboard": "un clavier",
    "scissors": "des ciseaux",
    "apple": "une pomme",
    "backpack": "un sac à dos",
    "handbag": "un sac",
    "umbrella": "un parapluie"
}

def speak_async(text):
    """Fait parler le PC dans un thread séparé"""
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[ERREUR AUDIO] : {e}")

    threading.Thread(target=_speak, daemon=True).start()

def get_spatial_position(x1, y1, x2, y2, frame_w, frame_h):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    if center_x < frame_w / 3:
        h_pos = "à gauche"
    elif center_x > 2 * frame_w / 3:
        h_pos = "à droite"
    else:
        h_pos = "en face"

    if center_y < frame_h / 3:
        v_pos = "en haut"
    elif center_y > 2 * frame_h / 3:
        v_pos = "en bas"
    else:
        v_pos = ""

    if v_pos:
        if h_pos == "en face":
            return v_pos
        return f"{v_pos} et {h_pos}"
    else:
        return h_pos

def estimate_distance(box_h, frame_h):
    ratio = box_h / frame_h
    if ratio > 0.6:
        return "à environ 40 centimètres"
    elif ratio > 0.35:
        return "à environ 70 centimètres"
    elif ratio > 0.15:
        return "à environ 1 mètre et demi"
    else:
        return "à plus de 2 mètres"

if __name__ == "__main__":
    print("--- Assistant Radar Intelligent (Stricte Comparaison de Texte) ---")
    model = YOLO("yolov8n.pt") 

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam.")
        exit()

    window_name = "Vision-to-Audio : Smart Visual Radar"
    cv2.namedWindow(window_name)

    last_spoken_description = ""
    last_speech_time = 0
    speech_debounce = 2.5  # Délai de sécurité minimal entre deux paroles

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            results = model(frame, conf=0.40, verbose=False)
            detected_items = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name_en = model.names[cls_id]

                    class_name_fr = TRANSLATIONS.get(class_name_en, class_name_en)

                    box_height = y2 - y1
                    spatial_pos = get_spatial_position(x1, y1, x2, y2, w, h)
                    distance_str = estimate_distance(box_height, h)

                    detected_items.append({
                        "name": class_name_fr,
                        "pos": spatial_pos,
                        "dist": distance_str,
                        "size": box_height
                    })

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name_fr} | {spatial_pos}"
                    cv2.putText(frame, label, (x1, max(35, y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Construction de la description actuelle
            if detected_items:
                detected_items = sorted(detected_items, key=lambda k: k['size'], reverse=True)
                phrases = [f"{item['name']} {item['pos']} ({item['dist']})" for item in detected_items[:3]]
                current_description = "Attention : " + ", et ".join(phrases)
            else:
                current_description = "Aucun objet détecté."

            # --- LOGIQUE DE COMPARAISON STRICTE (ZÉRO RÉPÉTITION INUTILE) ---
            current_time = time.time()
            if current_description != last_spoken_description:
                # La phrase a changé ! On vérifie juste le délai de sécurité pour ne pas saturer
                if current_time - last_speech_time > speech_debounce:
                    if current_description != "Aucun objet détecté.":
                        print(f"[RADAR UPDATE] : {current_description}")
                        speak_async(current_description)
                    
                    # On met à jour la mémoire avec la nouvelle phrase exacte
                    last_spoken_description = current_description
                    last_speech_time = current_time

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Programme arrêté proprement.")