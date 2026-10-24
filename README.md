# Vision-to-Audio : Smart Visual Radar for Visually Impaired Assistance

> Un système intelligent d'assistance visuelle en temps réel développé en Python, combinant Deep Learning (Computer Vision) et synthèse vocale contextuelle, conçu pour assister les personnes malvoyantes.

---

## 🚀 À propos du projet
Ce projet s'inscrit dans le cadre d'un développement d'ingénierie orienté accessibilité et technologies embarquées. Il transforme une simple webcam de PC en un **radar visuel intelligent** capable de :
1. Détecter les objets du quotidien en temps réel.
2. Analyser leur position spatiale précise (grille 3x3 : haut, bas, gauche, droite, en face).
3. Estimer leur distance empirique (en centimètres).
4. Restituer l'information de manière vocale en français de façon **ergonomique et réactive** (zéro spam auditif, parole conditionnée aux changements de la scène).

---

## 🛠️ Stack Technique
* **Langage :** Python 3.10+
* **Vision par Ordinateur :** OpenCV (`cv2`)
* **Deep Learning (Object Detection) :** YOLOv8 (Ultralytics, modèle nano `yolov8n.pt` optimisé pour l'exécution en local)
* **Synthèse Vocale (Text-to-Speech) :** `pyttsx3` (exécuté en multithreading pour fluidifier le flux vidéo)

---

## 📂 Structure du Projet
```text
VisionToAudio/
│
├── src/
│   └── step5_smart_assistant.py  # Script principal du Radar Intelligent (V2)
│
├── venv/                         # Environnement virtuel Python
├── requirements.txt              # Dépendances du projet
└── README.md                     # Documentation du projet