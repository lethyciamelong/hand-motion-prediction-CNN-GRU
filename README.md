# Hand Motion Prediction — CNN + GRU sur Dataset Vidéo Original


Implémentation d'un pipeline **CNN-GRU** pour la **prédiction du prochain mouvement de la main** lors d'un comptage de 0 à 5 doigts. Ce projet repose sur un dataset **entièrement constitué par nos soins** : nous avons nous-mêmes filmé les mains de volontaires, extrait les frames, et annoté chaque image.

---

## Table des matières

- [Présentation du projet](#présentation-du-projet)
- [Notre dataset — collecte originale](#notre-dataset--collecte-originale)
- [Architecture du pipeline](#architecture-du-pipeline)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation & prérequis](#installation--prérequis)
- [Utilisation](#utilisation)
- [Résultats](#résultats)


---

## Présentation du projet

L'objectif est d'apprendre à un modèle à **anticiper le geste suivant** d'une personne qui compte avec ses doigts, en s'appuyant sur une séquence d'images passées. Le pipeline respecte une architecture imposée en deux étapes distinctes :

1. **Modèle 1 — CNN** : à partir d'une image, produire un vecteur de représentation latente (256 dimensions) qui encode la position et la forme de la main.
2. **Modèle 2 — GRU** : à partir d'une séquence de 4 représentations latentes consécutives produites par le CNN, prédire la représentation latente de l'image suivante.

La qualité de la prédiction est mesurée par la **MSE** entre la représentation prédite par le GRU et la représentation réelle calculée par le CNN. Le GRU apprend donc à anticiper dans l'espace des caractéristiques visuelles, et non directement dans l'espace des pixels.

Le CNN utilisé est **MobileNetV2** (fine-tuning depuis ImageNet) et le module temporel est un **GRU (Gated Recurrent Unit)**.

---

## Notre dataset — collecte originale

> **Ce dataset est 100 % original.** Il n'existe pas dans la littérature et n'a pas été téléchargé depuis une source externe. Chaque vidéo a été filmée par les membres du groupe.

### Protocole de collecte

Nous avons organisé des sessions de filmage au cours desquelles des volontaires ont été invités à compter avec leurs doigts devant une caméra, en suivant un ordre précis et progressif de **0 à 5**.

Les conditions de tournage ont été standardisées pour garantir la cohérence du dataset :
- **Fond blanc uni** pour réduire le bruit visuel et aider le modèle à se concentrer sur la main.
- **Bon éclairage** uniforme, sans ombre parasite.
- **Variété des sujets** : hommes, femmes et enfants, avec une tranche d'âge allant de **8 à 60 ans**, afin que le modèle soit capable de généraliser à des mains de morphologies très différentes.
- Durée de chaque vidéo : **5 à 7 secondes**.

### Extraction des frames

Les vidéos ont été découpées en images à l'aide de **FFmpeg** à un rythme de **5 images par seconde (5 FPS)**, ce qui correspond à la dynamique naturelle d'un comptage réalisé à vitesse normale.

```bash
ffmpeg -i video_sujet.mp4 -vf fps=5 frames/NNN_%02d_C.jpg
```

### Convention de nommage

Chaque image respecte le format suivant :

```
NNN_SS_C.jpg
```

| Champ | Signification | Exemple |
|-------|--------------|---------|
| `NNN` | Numéro de séquence (seconde dans la vidéo) | `001` |
| `SS`  | Numéro de l'image dans la seconde | `03` |
| `C`   | **Classe / nombre de doigts levés** (0 à 5) | `2` |

Exemple : `001_03_2.jpg` → 3ème image de la 1ère seconde, la main montre **2 doigts**.

> Le label utilisé pendant l'entraînement est **uniquement le dernier chiffre** (`C`), extrait automatiquement depuis le nom du fichier.

### Chiffres clés du dataset

| Étape | Valeur |
|-------|--------|
| Vidéos filmées | 136 |
| Vidéos retenues (après contrôle qualité) | 115 |
| **Total images** | **3 689** |
| Images — classe 0 | 504 |
| Images — classe 1 | 673 |
| Images — classe 2 | 686 |
| Images — classe 3 | 678 |
| Images — classe 4 | 566 |
| Images — classe 5 | 582 |

Les 21 vidéos écartées présentaient des défauts : gestes saccadés, flou de mouvement, doigts partiellement hors cadre ou éclairage insuffisant.

### Répartition train / val / test

Le split a été réalisé de façon **stratifiée** pour que chaque sous-ensemble contienne la même proportion de chaque classe (0 à 5). Pour la phase GRU, le split est fait **par sujet** (et non par image), afin d'éviter toute fuite d'information entre ensembles.

| Ensemble | Images | Proportion |
|----------|--------|------------|
| Entraînement | 2 582 | 70 % |
| Validation | 553 | 15 % |
| Test | 554 | 15 % |

---

## Architecture du pipeline

### Vue d'ensemble

```
Image (t)                          Image (t+1)
   │                                    │
   ▼                                    ▼
[CNN — MobileNetV2]             [CNN — MobileNetV2]
   │                                    │
   ▼                                    ▼
Représentation r(t)             Représentation r(t+1)  ← réalité
   │                                    │
   └────────► [GRU] ──► r̂(t+1) ────► MSE Loss
              ▲
  4 représentations passées
  r(t-3), r(t-2), r(t-1), r(t)
```

Le GRU apprend à minimiser l'écart entre la représentation qu'il **prédit** (`r̂(t+1)`) et la représentation **réelle** calculée par le CNN sur l'image suivante (`r(t+1)`).

### Modèle 1 — CNN (MobileNetV2 fine-tuné)

| Composant | Détail |
|-----------|--------|
| Base | MobileNetV2, pré-entraîné ImageNet |
| Couches gelées (phase 1) | Toutes les couches MobileNetV2 |
| Tête personnalisée | GlobalAveragePooling2D → Dense(256, ReLU) → BatchNorm → Dropout(0.4) → Dense(6, Softmax) |
| Couche d'extraction | `feature_layer` — vecteur de 256 dimensions |
| Optimiseur | Adam (lr = 1e-4) |
| Loss | Sparse Categorical Crossentropy |
| Epochs | 20, batch size 32 |

### Modèle 2 — GRU (prédiction temporelle)

| Composant | Détail |
|-----------|--------|
| Entrée | Séquence de 4 vecteurs de 256 dimensions |
| Architecture | GRU(512, return_seq=True) → GRU(256, return_seq=False) → Dense(512, ReLU) → Dense(256, linéaire) |
| Loss | **MSE** (entre représentation prédite et représentation réelle) |
| Optimiseur | Adam |
| Epochs max | 50 |
| Callbacks | EarlyStopping (patience=10) + ReduceLROnPlateau (factor=0.2, patience=5) |
| Batch size | 16 |

### Stratégie de fenêtre glissante

Pour construire les séquences d'entraînement du GRU, nous utilisons une **fenêtre glissante** sur chaque groupe d'images d'un même sujet : pour chaque position `i`, les 4 images `[i, i+1, i+2, i+3]` constituent le contexte d'entrée et l'image `i+4` est la cible à prédire. Cette approche multiplie le nombre d'exemples disponibles tout en respectant l'ordre temporel des gestes.

### Expérimentation sur les niveaux d'abstraction du CNN

L'une des questions centrales de ce projet est : **à quel niveau du CNN faut-il extraire la représentation ?** Nous avons testé 5 configurations correspondant à différents pourcentages de couches MobileNetV2 gelées dans un pipeline CNN-GRU end-to-end :

| Configuration | % Gelé | Couches libres | Val Accuracy | Val Loss | Temps (s) |
|---------------|--------|----------------|--------------|----------|-----------|
| 90% Gelé | 90% | basses | — | — | — |
| 70% Gelé | 70% | intermédiaires basses | — | — | — |
| 50% Gelé | 50% | intermédiaires | — | — | — |
| 30% Gelé | 30% | hautes | — | — | — |
| 0% Gelé (full fine-tune) | 0% | toutes (154) | **meilleur** | **meilleur** | plus long |

> **Conclusion des expérimentations** : le modèle sans aucune couche gelée (fine-tuning complet) donne les meilleures performances de validation. Les configurations avec un fort pourcentage de couches gelées produisent une val_loss élevée et une val_accuracy insuffisante, car les représentations extraites des couches basses (textures, contours) ne capturent pas suffisamment la sémantique gestuelle nécessaire au GRU.

---

## Structure du dépôt

```
hand-motion-prediction/
│
├── Hand_Prediction.ipynb      # Notebook principal — pipeline complet
│
├── README.md                  # Ce fichier
│
└── data/                      # (non inclus — voir section Dataset)
    ├── sujet_AA/
    │   ├── 001_01_0.jpg
    │   ├── 001_02_0.jpg
    │   └── ...
    ├── sujet_AB/
    │   └── ...
    └── ...
```

> Le dossier `data/` n'est pas inclus dans ce dépôt en raison de sa taille (3 689 images). Pour reproduire les expériences, vous devez constituer votre propre dataset en suivant le protocole décrit ci-dessus, ou contacter les auteurs.

---

## Installation & prérequis

Ce projet a été développé et exécuté sur **Google Colab** (GPU T4).

### Dépendances Python

```python
tensorflow >= 2.12
scikit-learn
pandas
numpy
opencv-python (cv2)
matplotlib
seaborn
```

### Installation

```bash
pip install tensorflow scikit-learn pandas numpy opencv-python matplotlib seaborn
```

### FFmpeg (extraction des frames)

```bash
# Linux / macOS
sudo apt install ffmpeg   # ou brew install ffmpeg

# Extraction à 5 FPS
ffmpeg -i ma_video.mp4 -vf fps=5 dossier_sujet/%03d_%02d_C.jpg
```

---

## Utilisation

### 1. Préparer le dataset

Organiser les images dans un dossier `data/` avec un sous-dossier par sujet, en respectant la convention de nommage `NNN_SS_C.jpg`.

### 2. Ouvrir le notebook

```bash
# En local
jupyter notebook Hand_Prediction.ipynb

# Sur Google Colab : importer le notebook et monter Google Drive
```

### 3. Exécuter les sections dans l'ordre

| Section | Description |
|---------|-------------|
| **I — CNN** | Chargement, prétraitement, entraînement MobileNetV2, évaluation |
| **II — GRU** | Construction de l'extracteur, séquences, entraînement GRU, évaluation |
| **III — Expérimentations** | Comparaison des niveaux de CNN (% gelé), tableau récapitulatif |
| **Inférence** | Prédiction sur une nouvelle séquence de 4 images |

### 4. Inférence sur de nouvelles images

```python
nouvelle_sequence = [
    "data/mon_sujet/001_01_0.jpg",
    "data/mon_sujet/001_02_0.jpg",
    "data/mon_sujet/001_03_0.jpg",
    "data/mon_sujet/001_04_0.jpg"
]

classe, confiance = inferer_nouveau_geste(nouvelle_sequence, full_model, model_prediction)
print(f"Geste prédit : {classe} doigt(s) — Confiance : {confiance*100:.1f}%")
```

---

## Résultats

### CNN — MobileNetV2 (classificateur statique)

| Métrique | Train | Validation | Test |
|----------|-------|------------|------|
| Accuracy | 92.18% | 87.34% | **89%** |
| Loss | 0.254 | 0.395 | — |

Les classes aux formes les plus distinctives (0 et 1) obtiennent les meilleurs F1-scores. Les classes intermédiaires (3 et 4) sont légèrement plus difficiles à distinguer en raison de la proximité visuelle des positions de doigts.

### GRU — Prédiction temporelle

| Métrique | Valeur |
|----------|--------|
| Accuracy de classification (test) | 56% |
| Macro F1-score | 0.40 |
| Classe la mieux rappelée | 1 et 5 (recall 95%) |
| Classe la plus problématique | 0 (recall 0%) |

La performance limitée du GRU s'explique principalement par le **déséquilibre entre classes** (504 images pour la classe 0 vs 686 pour la classe 2) et par le **faible volume de séquences temporelles** disponibles après le split par sujet. La classe 0 (poing fermé) n'est presque jamais prédite car elle est sous-représentée.

---



*Université de Yaoundé I — Département d'Informatique — INF4248, 2024–2025*
