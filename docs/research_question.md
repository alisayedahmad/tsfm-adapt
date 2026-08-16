# Research question

## Question

Un adapter léger, backbone gelé, entraîné sans labels sur des données cible,
peut-il récupérer une part significative du gap de performance entre un TSFM
zero-shot et le même TSFM fine-tuné entièrement, avec beaucoup moins de données ?

## Scope du MVP

- 1 modèle : Chronos (T5-Small)
- 1 domaine : BDG2 (building energy consumption)
- 1 objectif d'entraînement de l'adapter : reconstruction

Pas de TimesFM, pas de Moirai, pas d'ENTSO-E, pas de NREL, pas d'objectif
contrastif tant que ce scope n'est pas terminé.

## Definition of done

- Baselines (ARIMA/ETS/Theta + au moins un modèle deep) tournent sur BDG2
  et donnent des métriques
- Chronos zero-shot tourne sur BDG2, métriques comparées aux baselines
- Adapter (reconstruction) entraîné sur BDG2, testé sur au moins 1 budget
  de données (1 semaine)
- Comparaison adapter vs zero-shot vs fine-tuning complet, sur les mêmes
  métriques (MASE, CRPS)

## Critère de succès

L'adapter récupère au moins 60% du gap entre zero-shot et fine-tuning complet,
avec 1 semaine de données cible.

Si ce n'est pas le cas : documenter pourquoi, pas insister avant d'avoir compris
la cause. Un résultat négatif expliqué est un résultat valable.

## Ce qu'on fait seulement après le MVP validé

Dans cet ordre, un par un :

1. 2e domaine (ENTSO-E)
2. 2e modèle (TimesFM ou Moirai)
3. objectif contrastif
4. stress tests (missing data, bruit)
5. 3e domaine (NREL)

