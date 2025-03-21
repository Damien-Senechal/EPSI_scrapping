#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de test pour vérifier chaque étape du pipeline de données.
Ce script permet de diagnostiquer où se produisent les erreurs dans le processus.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Création du dossier logs s'il n'existe pas
os.makedirs('logs', exist_ok=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/test_pipeline.log',
    filemode='w'  # 'w' pour écraser le fichier à chaque exécution
)
logger = logging.getLogger('test_pipeline')