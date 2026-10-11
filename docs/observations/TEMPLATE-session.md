# Séance GPU NN — observations

*Copier ce fichier dans `docs/observations/AAAA-MM-JJ-sessionNN/README.md`. Remplir uniquement à partir des JSON de la séance, copiés dans le même dossier. Une valeur non mesurée s'écrit « non mesuré », avec la raison.*

| Champ | Valeur |
|---|---|
| Date, durée GPU effective | |
| Pod (fournisseur, GPU, mémoire, pilote, CUDA, `nvcc`) | |
| Commits : alpamayo-recipes / alpamayo2 (épinglé) / alpamayo1.5 / afh / aconcagua | |
| flash-attn : compilé ou repli `sdpa` (venv 2S, venv 1.5) — inconnue 9 | |
| Écarts au plan `docs/gpu_session_01.md` | |

---

## Inconnue 1 — Le modèle livré émet-il `<|cot_start|>` ?

**Question.** Sur le prompt d'inférence de la release (`helper.prepare_model_inputs`, sans composante assistant), 2 Super génère-t-il `<|cot_start|>` en premier jeton, plus loin, ou pas du tout ? Et sous quelle forme le template d'entraînement officiel met-il le CoC dans la cible ? (D-008 hypothèse de travail ; D-011 à écrire si les deux divergent.)

**Valeur observée.**

| Mesure | Valeur |
|---|---|
| Fin du prompt (`prompt.tail_tokens`, 6 derniers) | |
| 10 premiers jetons générés (`runs[0].first_tokens[*].token`) | |
| Verdict (`summary.cot_start_statement`) | |
| Verdicts de toutes les exécutions (`summary.verdicts_all_runs`) | |
| Positions des marqueurs (`runs[0].marker_positions`) | |
| Texte avant `<|traj_future_start|>` (`runs[0].text_before_traj_future_start`) | |
| Ids de jetons conformes à la config (`token_ids.all_match`, `mismatches`) | |
| Tête de la réponse assistant dans le chemin d'entraînement (`attempts[k].steps.sequence.assistant_first_tokens`, bloc 2) | |

**Commande et JSON source.** `runners/probe_tokens_a2.py --out …/tokens_a2_0ea6fd88.json` (bloc 1) ; `runners/probe_train_memory_a2.py --out …/train_memory_a2.json` (bloc 2).

**Conclusion.** (D-008 confirmée / D-011 nécessaire : laquelle des deux options — cibles sans balises, ou inférence en prompt d'entraînement.)

---

## Inconnue 4 — Vocabulaire des méta-actions

**Question.** Quelles valeurs prennent `Longitudinal`, `Lateral` et `Lane` ? Apparaissent-elles sans prompt dédié (tâche trajectoire) ? Sont-elles balisées par `<|meta_action_start|>…<|meta_action_end|>` ?

**Valeur observée.**

| Mesure | Tâche trajectoire | Tâche `meta_action` |
|---|---|---|
| Méta-actions trouvées (`summary.meta_actions_found` / `meta_action_task.meta_actions`) | | |
| Source (`source` : `tagged` / `untagged`) | | |
| Bloc balisé (`meta_action_tagged_block`) | | |
| Extraction de la release (`release_extraction.meta_action`) | | |

**Commande et JSON source.** `runners/probe_tokens_a2.py` (bloc 1), champs `summary` et `meta_action_task` de `tokens_a2_0ea6fd88.json`.

**Conclusion.** (Vocabulaire observé ; un seul échantillon ne donne pas le vocabulaire complet ; suite à prévoir ou non.)

---

## Inconnue 5 — Forward et backward avec gradients partiels

**Question.** Le `forward` de la recette (`TrainableAlpamayo2Super`, étape `vlm`) tourne-t-il sur un GPU avec des gradients limités à `q_proj` des 4 dernières couches et le gradient checkpointing, avec des pertes finies et des gradients non nuls ?

**Valeur observée.**

| Mesure | Valeur |
|---|---|
| Configuration retenue, réduite ou non, et pourquoi (`summary.retained_config`, `retained_reduced`, `reduced_because`) | |
| `forward.loss` / `future_traj_ce` / `others_ce` | |
| Normes de gradient (`backward.grad_norms`) | |
| Erreur hors OOM éventuelle (`attempts[k].error`, `stopped_on_error`) | |

**Commande et JSON source.** `runners/probe_train_memory_a2.py --tokens-json …/tokens_a2_0ea6fd88.json --out …/train_memory_a2.json` (bloc 2).

**Conclusion.** (Le chemin officiel accepte un entraînement partiel, ou ce qui casse.)

---

## Inconnue 6 — Mémoire d'un pas d'entraînement

**Question.** Combien de mémoire coûte un pas d'étape 1 sur un GPU, par configuration d'entrée, et quelle est la plus grande configuration qui passe ? (D-009 ; budget de la roadmap, 4×H100.)

**Valeur observée.**

| Configuration | Statut (`ok`, `oom`, `too_long`) | Étape en échec | Longueur réelle | Pic après forward (GiB) | Pic après backward (GiB) |
|---|---|---|---|---|---|
| 7 caméras × 4 images | | | | | |
| 6 × 4 | | | | | |
| 4 × 4 | | | | | |
| 2 × 4 | | | | | |
| 1 × 4 | | | | | |
| 1 × 2 | | | | | |
| 1 × 1 | | | | | |

| Constats `SequenceTooLong` (`summary.findings`) : configuration, longueur réelle, `max_length` | |
|---|---|
| | |

| Mesure | Valeur |
|---|---|
| GPU et respect de l'exigence H200 (`env.gpu`, `env.gpu_requirement.meets_requirement`, `overridden`) | |
| Poids chargés (`summary.weights_gib`) | |
| Paramètres entraînables (`trainable.n_params`) | |
| CoC de l'exemple : nature, source, longueur (`training_path.cot.kind`, `source`, `tokens`, `length_distribution` si synthétique) | |

**Commande et JSON source.** `runners/probe_train_memory_a2.py` (bloc 2), `summary.attempts` de `train_memory_a2.json` ; `train_memory_a2_sweep.json` si `--sweep` a tourné.

**Conclusion.** (Critère D-014 : une configuration passe-t-elle sur le H200 avec des gradients de taille PEFT ? Si oui, W1 part de 2 Super sur le code de la recette et 1.5 devient le repli ; sinon D-003 tient. Ce que ça implique pour LoRA sur 4×H100 : entrée de D-009.)

---

## Inconnue 8 — Longueur de séquence et part visuelle

**Question.** Combien de jetons fait un exemple d'entraînement réel, dont combien de jetons visuels, par configuration, et reste-t-il sous les 6 144 de la recette ?

**Valeur observée.**

| Configuration | Longueur réelle avant padding (`sequence_before_padding.real_len`) | Longueur paddée | Jetons visuels | Part visuelle | Jetons du CoC | Jetons sous le masque |
|---|---|---|---|---|---|---|
| | | | | | | |

**Commande et JSON source.** `runners/probe_train_memory_a2.py` (bloc 2), `attempts[k].steps.sequence` ; pour comparaison, `prompt.prompt_len` et `prompt.n_image_pad` de `tokens_a2_0ea6fd88.json` (bloc 1, profil 6 caméras).

**Conclusion.** (Marge sous 6 144 ; coût du CoC en jetons ; chaque séquence au-delà de 6 144 est un constat `SequenceTooLong` à rapporter tel quel, avec sa configuration ; la mesure retenue est alors marquée réduite.)

---

## Go / no-go W3

**Question.** Le pipeline d'incertitude (W3) peut-il démarrer, et sur quel modèle ? Décision d'Ousseynou (D-003 réexaminée dans D-014) ; cette rubrique rassemble les faits.

**Valeur observée.**

| Critère | Mesure | Valeur |
|---|---|---|
| 1.5 partage-t-il le défaut de NVlabs/alpamayo2#9 ? | `cf_uncertainty.clear_path_rate` / `no_input_admitted_rate` | |
| Les traces sous blackout (mot pour mot) | `cf_traces` | |
| L'effet dépasse-t-il le bruit d'échantillonnage ? | `exceeds_noise_floor`, `null_control_behavior_change` vs `behavior_change` | |
| Verdict contrefactuel 1.5 vs 2S | `verdict`, `baseline_citation`, `cf_citation` vs `fixtures/cf_0ea6fd88_a2_blackout.json` | |
| Entraîner 2S sur un GPU est-il réaliste ? | `summary.retained_config` du bloc 2 | |
| La chaîne de cibles tourne-t-elle ? | résumés `w3_split/train.jsonl` et `test.jsonl` (bloc 4) | |

**Commande et JSON source.** `runners/probe_blackout_a15.py --k-rollouts 5 --null-control --out …/cf_0ea6fd88_a15_blackout.json` (bloc 3, venv 1.5) ; bloc 2 ; `python -m afh.uncertainty_dataset split|summary` (bloc 4).

**Conclusion.** (GO W3 sur 1.5 / GO W3 sur 2 Super / NO-GO, avec la raison factuelle ; ce qui reste à décider.)
