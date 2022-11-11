# H1N1-Flu-Shot-Learning
Idées d'amélioration
============= 
En général
------
* Utiliser zeus pour l'entrainnement et les taches longues.
* Créer plusieurs modèles performants de ML et les assembler par méthode d'Ensemble Learning (Soft Voting, Bagging or pasting, stacking). Car les modèles d'Ensemble learning généralisent mieux les prédictions.
* Utiliser la cross validation pour construire les modèles.

Sur les données
-------

* Actuellement on a remplacé les données manquantes par '-1' et 'missing' ce qui marche bien mais ne pourrait on pas améliorer l'imputing ? Par exemple, utiliser '-1' et 'missing' seulement quand les données sont vraiment incertaines pour le KNN imputer
* Possibilité de générer de nouvelles données grâce au feature engineering (autofeat ou featuretools)
* Il faudrait faire une étude poussée des données importantes et sélectionner les plus importantes (SelectKBest, Chi2)
* Possibilité de faire de la réduction de dimension
* Utiliser du OrdinalEncoding pour certaines variables et OneHot pour d'autres

Sur les Algorithmes
---------
**Deep learning**
* Tuner les hyperparamètres des modèles avec KerasTuner (faire ça sur plusieurs heures)
* Changer la structure du réseau
* Layers d'extraction de features
