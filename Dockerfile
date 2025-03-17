# Utilisation de l'image Python 3.12-slim-bookworm comme base
FROM python:3.12-slim-bookworm

# Définition des variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Définition du répertoire de travail
WORKDIR /app

# Installation des dépendances système
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers requirements
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copie du code source
COPY . .

# Création des répertoires nécessaires
RUN mkdir -p data/extract data/transform data/load data/models data/visualizations logs

# Port pour une éventuelle interface web
EXPOSE 8501

# Commande par défaut au démarrage du conteneur
CMD ["python", "main.py"]

# Pour exécuter l'application avec Streamlit (décommentez si vous ajoutez Streamlit)
# CMD ["streamlit", "run", "app.py"]