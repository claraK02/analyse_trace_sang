# Guide d'utilisation de l'application Streamlit pour la classification des taches de sang

## Prérequis
- Python 3.7 ou supérieur
- Streamlit
- PyTorch
- OpenCV
- PIL
- Pandas

## Comment lancer l'application

1. **Se placer dans le répertoire 'streamlit'**:
```bash
cd streamlit
```
2. **Run la commande suivante**:
```bash
streamlit run app.py
```
3. **Ouvrir le navigateur**: si tout s'est bien passé, vous devriez voir un message indiquant que l'application est en cours d'exécution et un lien vers lequel vous pouvez cliquer pour ouvrir l'application dans votre navigateur voir le navigateur qui s'ouvre automatiquement.

## Lancement rapide

Il est possible de lancer l'application en lançant le script `run_app.bat` en double cliquant sur ce fichier depuis votre explorateur de fichier (valide uniquement pour Windows). Ce script installe les dépendances nécessaires et lance l'application.

## Comment utiliser l'application


1. **Barre latérale des paramètres**: La barre latérale contient plusieurs options que vous pouvez ajuster selon vos besoins.

    - **Enregistrer les résultats**: Si cette case est cochée, les résultats de la classification seront enregistrés.
    - **Chemin global**: Vous pouvez spécifier le chemin (absolu) où les résultats doivent être enregistrés, si non spécifié, les résultats seront enregistrés dans le répertoire courant.
    - **Plot Saliency Map**: Si cette case est cochée, une carte de saillance sera générée pour chaque image. Elle permet de visualiser les régions de l'image qui ont le plus contribué à la prédiction du modèle. 

2. **Sélection de la configuration**: Vous pouvez sélectionner un fichier de configuration parmi ceux disponibles dans le répertoire 'logs'. Il correspond à un modèle pré-entraîné avec une configuration spécifique. Si vous ne sélectionnez pas de fichier de configuration, le modèle par défaut sera utilisé.

3. **Téléchargement d'image**: Vous pouvez télécharger une ou plusieurs images sur lesquelles l'application effectuera une prédiction en cliquant sur `Browse files` ou en glissant-déposant les images dans la zone prévue à cet effet.

3. **Impression des critères d'interprétabilité**: Si cette case est cochée, l'application affichera les critères d'interprétabilité pour chaque image.

4. **Bouton pour lancer la prédiction**: Une fois que vous avez ajusté tous les paramètres et téléchargé vos images, vous pouvez cliquer sur le bouton `Inférence` pour lancer la prédiction.

## Informations affichées par l'application

Une fois que vous avez lancé la prédiction, l'application affichera les informations suivantes:

- **L'image**: L'image téléchargée.
- **Prédiction**: La prédiction du modèle pour l'image. (ex: 3-Tâche passive)
- **Saliency Map**: Si vous avez choisi de générer des cartes de saillance, elles seront affichées à côté de l'image. Les zones rouges indiquent les régions de l'image qui ont le plus contribué à la prédiction du modèle, les zones bleues indiquent les régions qui ont le moins contribué à la prédiction du modèle.
- **Probaility Histogram**: Un histogramme des probabilités prédites pour chaque classe, correspondant à la certitude du modèle pour chaque classe.

## Où les données sont-elles enregistrées?

Si vous avez choisi d'enregistrer les résultats, ils seront enregistrés dans le chemin que vous avez spécifié dans le champ "Chemin global". Si aucun chemin n'est spécifié, les résultats seront enregistrés dans le répertoire courant. Les résultats sont enregistrés sous forme de fichier CSV nommé 'results.csv'. Ce fichier contient un tableau avec dans chaque ligne le nom du fichier image traité et la classe prédite. De plus, si vous avez choisi de générer des cartes de saillance, elles seront enregistrées dans un sous-répertoire nommé `saliency_maps`.

## Problèmes potentiels

- **Problèmes de mémoire**: Si vous téléchargez un grand nombre d'images ou des images de très grande taille, l'application peut rencontrer des problèmes de mémoire.
- **Problèmes de performance**: L'inférence sur les images peut prendre un certain temps, en particulier si vous utilisez un modèle complexe ou si vous téléchargez un grand nombre d'images.
- **Problèmes de compatibilité**: Cette application a été testée avec Python 3.7. Des problèmes peuvent survenir si vous utilisez une version différente de Python.