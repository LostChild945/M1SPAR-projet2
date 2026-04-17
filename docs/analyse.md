# Analyse du dataset Amazon Reviews

Ce document présente ce qu'on apprend concrètement en explorant le dataset, et ce que ça implique pour la construction du système de recommandation.

---

## Le dataset en quelques chiffres

| | |
|---|---|
| Utilisateurs | 13 664 354 |
| Produits | 1 834 471 |
| Interactions réelles | 34 253 280 |
| Interactions possibles | 25 000 milliards |

On travaille sur **5 catégories Amazon** : Automotive, Arts & Crafts, Amazon Fashion, Appliances, All Beauty.

---

## Q1 — La matrice est presque entièrement vide (sparsité 99,9999 %)

Sur 25 000 milliards de paires user-produit possibles, seulement 34 millions ont été observées. Autrement dit, **un utilisateur donné n'a interagi qu'avec une fraction infime du catalogue** — en moyenne moins d'un produit sur 500 000.

Ce n'est pas une anomalie : c'est la réalité de tout marketplace à grande échelle. Amazon a des millions de références, et chaque acheteur n'en voit qu'une toute petite partie.

**Ce que ça change pour le modèle :** on ne peut pas travailler directement avec cette matrice (elle ne tiendrait pas en mémoire et serait inutilisable). Il faut un algorithme conçu pour les matrices creuses, comme ALS, qui va *déduire* les préférences manquantes à partir des rares interactions connues.

---

## Q2 — 95 % des interactions sont des achats vérifiés

| Type | Volume | % |
|---|---|---|
| Achat vérifié (purchase) | 32 725 041 | 95,5 % |
| Simple review | 1 528 239 | 4,5 % |

Quand quelqu'un laisse un avis sur Amazon après avoir acheté, Amazon le marque `verified_purchase`. C'est un **signal fort** : la personne a réellement sorti sa carte bleue.

Les 4,5 % restants sont des reviews non vérifiées — l'utilisateur a peut-être juste lu la fiche produit, ou reçu le produit gratuitement. C'est un **signal faible**, moins fiable.

Par ailleurs, 7,6 millions de reviews ont reçu des votes "helpful" (utile), avec une moyenne de 0,75 vote par review et un maximum de 11 030. Ces votes sont un signal de qualité supplémentaire qu'on pourrait utiliser pour pondérer l'importance d'une interaction dans l'entraînement.

**Ce que ça change pour le modèle :** on doit distinguer ces deux types. Un achat pèse plus qu'une simple review dans l'apprentissage des préférences.

---

## Q3 — Les notes sont massivement biaisées vers le haut

| Note | Volume | % |
|---|---|---|
| 5 étoiles | 22 893 163 | **66,8 %** |
| 4 étoiles | 3 754 822 | 11,0 % |
| 1 étoile | 3 841 987 | 11,2 % |
| 3 étoiles | 2 198 464 | 6,4 % |
| 2 étoiles | 1 564 843 | 4,6 % |

La moyenne est de **4,18 / 5**, avec un écart-type de 1,38. La distribution est en forme de J : très peu de notes 2-3, un pic massif à 5 et un second pic à 1 (les gens très mécontents se donnent la peine d'écrire).

Pourquoi ce biais ? Deux raisons principales :
1. **Survivorship bias** : on ne laisse un avis que sur les produits qu'on a reçus et utilisés. Les produits vraiment mauvais sont souvent retournés avant toute review.
2. **Self-selection** : les acheteurs satisfaits sont plus enclins à noter.

**Ce que ça change pour le modèle :** si on entraîne ALS directement sur les ratings bruts, un produit noté 3/5 ressemblera à un produit "moyen" alors qu'il est en réalité dans le haut de la distribution réelle. Il vaut mieux utiliser le signal binaire achat/non-achat (implicit feedback) plutôt que la note elle-même.

---

## Q4 — La majorité des produits sont quasi-inconnus du modèle

### Côté produits

| Seuil | Produits concernés | % du catalogue |
|---|---|---|
| ≤ 1 avis | 2 087 968 | **113,8 %** |
| ≤ 5 avis | 3 559 058 | 194,0 % |
| ≤ 10 avis | 3 942 289 | 214,9 % |

> Les pourcentages dépassent 100 % car certains produits du catalogue n'ont aucune interaction — ils sont comptés ici mais absents de la table interactions.

Concrètement : **plus de la moitié des produits ont 1 avis ou moins**. ALS ne peut pas apprendre de facteurs latents fiables avec si peu de données — on appelle ça le **problème du cold start**.

### Côté utilisateurs

| Seuil | Utilisateurs concernés | % des users |
|---|---|---|
| ≤ 1 avis | 7 725 554 | 56,5 % |
| ≤ 5 avis | 12 518 803 | **91,6 %** |
| ≤ 10 avis | 13 273 056 | 97,1 % |

Plus de 9 utilisateurs sur 10 ont moins de 5 interactions. Pour ces utilisateurs, ALS a trop peu d'informations pour construire un profil fiable.

**Ce que ça change pour le modèle :** on ne peut pas se contenter d'ALS seul. Il faut une stratégie pour les entités "cold" :
- Pour les produits cold → utiliser les embeddings de contenu (titre, description) via Sentence Transformers + bridge model
- Pour les users cold → approximer leur profil par la moyenne des embeddings des produits qu'ils ont consultés

La loi de Pareto s'applique clairement : **quelques catégories concentrent l'essentiel des interactions** (Automotive à elle seule représente 58,3 % du volume).

---

## Q5 — Les power users sont une minorité critique

| Segment | Nb users | % users | Achats moy. | % des achats |
|---|---|---|---|---|
| power_user | 3 989 719 | **29,2 %** | 39,1 | **54,2 %** |
| regular_user | 7 660 158 | 56,1 % | 7,4 | 39,7 % |
| casual_user | 2 014 477 | 14,7 % | 1,5 | 6,1 % |

Les power users représentent moins d'un tiers des utilisateurs mais génèrent plus de la moitié des achats. Leur note moyenne (4,26) est aussi légèrement plus élevée que celle des casual users (3,78) — ils sont plus indulgents ou achètent des produits de meilleure qualité.

**Ce que ça change pour le modèle :** optimiser uniquement le CTR moyen sur l'ensemble des utilisateurs reviendrait à sacrifier la qualité des recommandations pour les power users — au profit d'une masse de casual users qui achèteront de toute façon peu. C'est une erreur commerciale. Il faut évaluer le modèle **séparément par segment** et s'assurer que les power users ne régressent pas.

---

## Ce qu'on retient globalement

| Observation | Conséquence architecturale |
|---|---|
| Sparsité 99,9999 % | ALS est l'algorithme adapté |
| 95 % d'achats vérifiés | Implicit feedback > ratings bruts |
| 66,8 % de notes à 5 étoiles | Ne pas utiliser le rating comme signal principal |
| 91,6 % des users ont < 5 interactions | Cold start inévitable → bridge model requis |
| Power users = 29 % mais 54 % des achats | Évaluer par segment, protéger les power users |

Ces cinq constats justifient les choix techniques du pipeline : ALS en mode implicit, Sentence Transformers pour le cold start, MLP bridge pour projeter la sémantique vers l'espace comportemental, et métriques segmentées par type d'utilisateur.
