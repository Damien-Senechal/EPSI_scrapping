# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Visualization:
    """
    Classe pour la visualisation des données et des résultats d'analyse.
    """
    
    def __init__(self):
        """Initialise la classe de visualisation"""
        # Création du répertoire pour sauvegarder les visualisations
        os.makedirs('data/visualizations', exist_ok=True)
        # Configuration du style
        sns.set(style='whitegrid')
        plt.rcParams.update({'font.size': 12})
    
    def plot_cost_comparison(self, df, items, title="Comparaison des coûts par ville"):
        """
        Crée un graphique à barres comparant les coûts entre villes
        
        Args:
            df: DataFrame avec les villes en index et les coûts en colonnes
            items: liste des éléments de coût à comparer
            title: titre du graphique
        """
        plt.figure(figsize=(14, 10))
        
        # Création du graphique
        ax = df[items].T.plot(kind='bar', figsize=(14, 10))
        
        # Personnalisation
        plt.title(title, fontsize=16)
        plt.xlabel('Éléments', fontsize=14)
        plt.ylabel('Coût (€)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Ville', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Sauvegarde
        plt.savefig(f'data/visualizations/cost_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_health_heatmap(self, df, title="Heatmap des métriques de santé"):
        """
        Crée une heatmap des métriques de santé par ville
        
        Args:
            df: DataFrame avec les données de santé
            title: titre du graphique
        """
        # Sélection des colonnes numériques uniquement
        numeric_df = df.select_dtypes(include=[np.number])
        
        plt.figure(figsize=(16, 12))
        
        # Création de la heatmap
        sns.heatmap(numeric_df, annot=True, cmap='viridis', linewidths=.5, fmt='.1f')
        
        plt.title(title, fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Sauvegarde
        plt.savefig('data/visualizations/health_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_city_radar(self, df, city, metrics, title=None):
        """
        Crée un graphique radar pour une ville spécifique
        
        Args:
            df: DataFrame avec les données
            city: nom de la ville à visualiser
            metrics: liste des métriques à inclure
            title: titre du graphique (optionnel)
        """
        # Nombre de variables
        N = len(metrics)
        
        # Répétition de la première valeur pour fermer le graphique circulaire
        values = df.loc[city, metrics].values.flatten().tolist()
        values += values[:1]
        
        # Calcul de l'angle pour chaque métrique
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Initialisation du graphique
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Tracé du contour des données
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        
        # Remplissage de la zone
        ax.fill(angles, values, alpha=0.25)
        
        # Définition des étiquettes
        plt.xticks(angles[:-1], metrics, size=12)
        
        # Tracé des étiquettes de l'axe y
        ax.set_rlabel_position(0)
        max_val = max(values)
        plt.yticks([max_val/4, max_val/2, 3*max_val/4, max_val], 
                  [f"{max_val/4:.1f}", f"{max_val/2:.1f}", f"{3*max_val/4:.1f}", f"{max_val:.1f}"], 
                  size=10)
        plt.ylim(0, max_val)
        
        # Ajout du titre
        if title is None:
            title = f"Métriques pour {city}"
        plt.title(title, size=16, pad=20)
        
        # Sauvegarde du graphique
        plt.savefig(f'data/visualizations/radar_{city.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, importance_dict, title="Importance des caractéristiques"):
        """
        Trace l'importance des caractéristiques d'un modèle
        
        Args:
            importance_dict: dictionnaire avec les noms des features et leur importance
            title: titre du graphique
        """
        # Tri par importance
        sorted_importance = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
        
        plt.figure(figsize=(12, 10))
        
        # Création d'un graphique à barres horizontales
        plt.barh(list(sorted_importance.keys()), list(sorted_importance.values()))
        
        # Personnalisation
        plt.title(title, fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Caractéristiques', fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Sauvegarde
        plt.savefig('data/visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_regression_results(self, y_test, y_pred, title="Valeurs prédites vs. réelles"):
        """
        Trace les résultats de régression
        
        Args:
            y_test: valeurs réelles
            y_pred: valeurs prédites
            title: titre du graphique
        """
        plt.figure(figsize=(10, 8))
        
        # Nuage de points des valeurs prédites vs. réelles
        plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
        
        # Ligne de prédiction parfaite
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        
        # Personnalisation
        plt.title(title, fontsize=16)
        plt.xlabel('Valeurs réelles', fontsize=14)
        plt.ylabel('Valeurs prédites', fontsize=14)
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Sauvegarde
        plt.savefig('data/visualizations/regression_results.png', dpi=300)
        plt.close()
    
    def plot_classification_results(self, y_test, y_pred, classes=None, title="Résultats de classification"):
        """
        Trace la matrice de confusion pour la classification
        
        Args:
            y_test: valeurs réelles
            y_pred: valeurs prédites
            classes: noms des classes (optionnel)
            title: titre du graphique
        """
        from sklearn.metrics import confusion_matrix
        
        # Calcul de la matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        
        # Tracé de la matrice de confusion
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        
        # Personnalisation
        plt.title(title, fontsize=16)
        plt.xlabel('Prédit', fontsize=14)
        plt.ylabel('Réel', fontsize=14)
        plt.tight_layout()
        
        # Sauvegarde
        plt.savefig('data/visualizations/classification_results.png', dpi=300)
        plt.close()
    
    def plot_pca_analysis(self, df, n_components=2, target_col=None, title="Analyse PCA"):
        """
        Effectue une ACP et visualise les résultats
        
        Args:
            df: DataFrame avec les données
            n_components: nombre de composantes principales (2 ou 3)
            target_col: colonne cible pour la coloration (optionnel)
            title: titre du graphique
        """
        # Préparation des données
        if target_col is not None:
            X = df.drop(target_col, axis=1)
            y = df[target_col]
        else:
            X = df
            y = None
        
        # Standardisation des caractéristiques
        X_scaled = StandardScaler().fit_transform(X)
        
        # Application de l'ACP
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X_scaled)
        
        # Création d'un DataFrame avec les composantes principales
        pca_df = pd.DataFrame(data=principal_components, 
                             columns=[f'PC{i+1}' for i in range(n_components)])
        
        # Ajout de la variable cible si fournie
        if target_col is not None and y is not None:
            pca_df[target_col] = y.values
        
        # Visualisation
        plt.figure(figsize=(12, 10))
        
        if n_components == 2:
            if target_col is not None:
                # Coloration par cible
                sns.scatterplot(x='PC1', y='PC2', hue=target_col, data=pca_df, palette='viridis', s=100)
            else:
                # Pas de coloration par cible
                sns.scatterplot(x='PC1', y='PC2', data=pca_df, s=100)
            
            # Ajout des étiquettes de ville si l'index contient les noms des villes
            if isinstance(df.index, pd.Index) and df.index.name == 'City':
                for i, city in enumerate(df.index):
                    plt.annotate(city, (pca_df.iloc[i, 0], pca_df.iloc[i, 1]), 
                                fontsize=12, alpha=0.8)
        
        elif n_components == 3:
            # Graphique 3D
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            if target_col is not None:
                # Coloration par cible
                scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], 
                                   c=pca_df[target_col], cmap='viridis', s=100)
                plt.colorbar(scatter, label=target_col)
            else:
                # Pas de coloration par cible
                ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], s=100)
            
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
        
        # Variance expliquée
        explained_variance = pca.explained_variance_ratio_
        plt.title(f"{title}\nVariance expliquée: {sum(explained_variance):.2%}", fontsize=16)
        plt.tight_layout()
        
        # Sauvegarde
        plt.savefig('data/visualizations/pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return pca_df, pca