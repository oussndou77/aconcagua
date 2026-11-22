# Séance GPU 02 — observations (Alpamayo 2 Edge)

*Remplir une copie de ce fichier à partir des JSON de la séance (archive `session02_outputs.tgz`). Une valeur non mesurée s'écrit « non mesuré », avec la raison.*

*Avant de publier une copie remplie : voir « Ce qui est publiable » dans `docs/gpu_session_02.md`. Les lignes marquées **(non publiable)** restent hors de toute publication jusqu'à la réponse de NVIDIA (D-012) ; les JSON bruts et le dossier `eval_base` ne se copient pas dans le dépôt sans ce tri.*

| Champ | Valeur |
|---|---|
| Date, durée GPU effective | |
| Pod (fournisseur, GPU, mémoire, pilote, `nvcc`) ; `env.gpu` d'un des JSON | |
| Commits : alpamayo-recipes / alpamayo2-edge / aconcagua (`env.alpamayo2_edge.commit`, `pins`) | |
| Révisions HF : modèle / PAI ; résultat des trois `cmp` du bloc 0 | |
| Durée de `uv sync`, incidents d'installation | |
| Contrôle d'API du bloc 0 (dernière ligne de `api_check.log` : `9 passed`, aucun sauté) | |
| Écarts au plan `docs/gpu_session_02.md` | |

---

## Bloc 1 — Que génère Edge sur le prompt natif ?

**Question.** Sur les 19 événements annotés, avec le prompt de la release : sous quelle forme sort le raisonnement, quelle est sa longueur (zéro compris), et que vaut la trajectoire ? (Inconnues 4, 8, 11, 13, 15.)

**Valeur observée.**

| Mesure | Valeur |
|---|---|
| Événements avec sortie / échoués ou non chargés (`summary.events_with_output`, `events_failed_or_not_loaded`) | |
| Longueur du prompt, min et max (`summary.prompt_len`) ; jetons d'image mesurés contre 2 892 prédits (`conditions.clean.prompt.n_image_tokens`, `load.predicted_visual_tokens_native`) | |
| Le prompt finit par `<think></think>` (`summary.prompt_ends_with_closed_thinking`) | |
| Candidats, raisons d'arrêt (`summary.reasoning.n_candidates`, `stop_counts`) | |
| Événements sans ancre (`summary.events_without_anchor`) | |
| CoC vides (`summary.reasoning.n_empty_cot`) | |
| Longueur du raisonnement en jetons, min / médiane / max (`summary.reasoning.reasoning_tokens`) | |
| Balises `<think>` générées, méta-actions, jetons de trajectoire générés (`n_think_open_generated`, `n_with_meta_actions`, `n_with_future_tokens_generated`) | |
| 12 premiers jetons d'un candidat (`events[0].conditions.clean.candidates[0].first_tokens[*].token`) | |
| Génération sans les filtres de la release (`events[0].conditions.unconstrained.candidates[0]` : `stop`, `n_future_trajectory_tokens`, `after_anchor`, `eos_positions`) | |
| Dispersion entre candidats, par événement (`summary.per_event[*].within_spread_m`) | |
| Pic mémoire et temps par événement (`conditions.clean.peak_memory_gib`, `seconds`) | |
| Même graine, mêmes jetons et mêmes trajectoires ? (`reproducibility`) | |
| Identifiants de jetons conformes à la config (`token_ids.all_match`, `mismatches`) | |

Trois CoC mot pour mot (`summary.per_event[*].cot_first_candidate`), avec l'instruction de navigation de l'événement :

| Événement | Navigation | CoC |
|---|---|---|
| | | |

**Commande et JSON source.** `runners/probe_tokens_edge.py --repro-check --out …/tokens_edge.json`.

**Conclusion.** (Format réellement généré ; le raisonnement est-il parfois vide, et dans quelles scènes ; ce que la passe sans filtres ajoute ; la sortie est-elle reproductible à graine fixée.)

---

## Bloc 2 — Blackout

**Question.** Toutes caméras noires, puis les deux caméras avant, puis la caméra frontale seule : la CoC décrit-elle encore une scène, et la trajectoire s'écarte-t-elle du run propre au-delà de la dispersion entre candidats ? (Inconnue 13 ; défaut de référence : NVlabs/alpamayo2#9 sur 2 Super.)

**Valeur observée.**

| Mesure | `blackout_all` | `blackout_front_both` | `blackout_front` |
|---|---|---|---|
| Images noircies (`conditions.<condition>.blacked_out.n_images`) | 22 | 10 (caméras 1 et 6) | 5 (caméra 1) |
| Événements avec sortie / manquants (`events_with_output`, `events_missing`) | | | |
| Événements au-delà de la dispersion propre (`n_exceeding_clean_spread` / `n_compared_with_spread`) | | | |
| Écart moyen entre trajectoires moyennes (`mean_ade_between_means_m`) | | | |
| Dispersion propre moyenne, le plancher (`mean_clean_within_spread_m`) | | | |
| Longueur de parcours, propre / noircie (`mean_path_length_clean_m`, `mean_path_length_condition_m`) | | | |
| CoC vides, longueur médiane (`reasoning.n_empty_cot`, `reasoning.reasoning_tokens`) | | | |
| Événements sans ancre (`reasoning.stop_counts`) | | | |

Cinq paires de CoC mot pour mot (`summary.<condition>.per_event[*].cot_clean`, `cot_condition`) :

| Événement | Condition | CoC propre | CoC noircie |
|---|---|---|---|
| | | | |

**Commande et JSON source.** `runners/probe_blackout_edge.py --baseline-json …/tokens_edge.json --out …/blackout_edge.json`.

**Conclusion.** (Sous `blackout_all`, la CoC cite-t-elle des éléments visuels — feux, véhicules, voie libre ? La trajectoire change-t-elle, ralentit-elle ? Différence entre les trois conditions : sans le grand angle seul, le téléobjectif suffit-il ; sans les deux caméras avant, le modèle se comporte-t-il comme sans aucune caméra ? Edge partage-t-il le défaut de 2 Super : oui / non / partiellement, avec les faits.)

---

## Bloc 3 — Contrefactuel de navigation

**Question.** Mêmes images, instruction changée : la trajectoire et la CoC suivent-elles l'instruction ? (Inconnue 14 ; premier point d'extension de W3.)

**Valeur observée.**

| Mesure | `mirror` | `straight` |
|---|---|---|
| Événements lancés / sautés ou échoués (`events_run`, `events_skipped_or_failed`) | | |
| Instructions non validées par la recette (`instructions_not_validated`) | | |
| Trajectoire au-delà de la dispersion propre (`trajectory_beyond_clean_spread`) | | |
| Écart moyen entre trajectoires moyennes (`mean_ade_between_means_m`) | | |
| Latéral d'arrivée moyen, propre / contrefactuel (`mean_abs_end_y_clean_m`, `mean_abs_end_y_counterfactual_m`) | | |
| Signe du latéral inversé (`lateral_sign_flipped`) | | sans objet |
| La CoC nomme la nouvelle direction (`cot_names_new_direction`) | | |
| La CoC nomme encore l'ancienne (`cot_still_names_old_direction`) | | |
| Le texte de la CoC a changé (`cot_text_changed`) | | |

Cinq paires de CoC mot pour mot (`summary.<étiquette>.per_event[*]`) :

| Événement | Instruction d'origine | Instruction contrefactuelle | CoC propre | CoC contrefactuelle |
|---|---|---|---|---|
| | | | | |

**Commande et JSON source.** `runners/probe_nav_edge.py --baseline-json …/tokens_edge.json --out …/nav_edge.json`.

**Conclusion.** (La trajectoire suit-elle l'instruction, la CoC aussi, l'une sans l'autre ? Ce que ça implique pour produire des instructions sur nos événements : une instruction approximative suffit-elle, ou change-t-elle la sortie ?)

---

## Bloc 4 — Mémoire d'un pas d'entraînement conjoint

**Question.** Le plein fine-tuning conjoint de la recette tient-il sur un GPU de 80 Go avec l'entrée native (6 caméras, 22 images), sans DeepSpeed, et à quel coût par pas ? (Inconnues 6 et 8 ; D-004.)

**Valeur observée.**

| Configuration | Images | Statut (`ok`, `oom`, `too_long`) | Étape en échec | Longueur réelle | Jetons visuels mesurés / prédits | Pic mémoire (Gio) | Secondes par pas |
|---|---|---|---|---|---|---|---|
| `6cam_22img_native` | 22 | | | | | | |
| `6cam_12img_last2` | 12 | | | | | | |
| `6cam_6img_last1` | 6 | | | | | | |
| `2cam_2img_front` | 2 | | | | | | |
| `1cam_1img_recipe_debug` | 1 | | | | | | |

| Mesure | Valeur |
|---|---|
| GPU et respect de l'exigence de 75 Gio (`env.gpu`, `env.gpu_requirement.meets_requirement`, `overridden`) | |
| Le plein fine-tuning natif tient (`summary.native_full_finetune_fits`) | |
| Configuration retenue, réduite ou non, et pourquoi (`summary.retained_config`, `retained_reduced`, `reduced_because`) | |
| Constats `SequenceTooLong` (`summary.findings` : configuration, longueur réelle, `max_length`) | |
| Poids chargés en FP32 (`summary.weights_gib`) ; paramètres entraînés et leur type (`model.n_trainable`, `trainable_dtypes`) | |
| Pics du premier pas : après passe avant / arrière / optimiseur (`train_steps[0].peak_after_*_gib`) | |
| Pertes du premier pas (`loss`, `vlm_ce`, `future_traj_ce`, `others_ce`, `expert_diffusion`) et norme du gradient **(non publiable)** | |
| Aucune pose écrite (`example.target_trajectory.values_written` vaut `false`) | |
| Jetons supervisés et début du texte supervisé (`steps.sequence.supervised_tokens`, `supervised_text_head`) | |
| CoC de l'exemple : source et longueur (`example.cot`) | |
| Le prétraitement lu dans la recette est celui attendu (`recipe.preprocess_matches_pinned_copy`) | |

**Commande et JSON source.** `runners/probe_train_memory_edge.py --out …/train_memory_edge.json` ; `…/train_memory_edge_sweep.json` si `--sweep` a tourné.

**Conclusion.** (Règle de D-004 : si la configuration native tient sur 80 Go, pas de LoRA en W1 ; sinon D-004 est rouverte, avec le pic atteint et la configuration qui passe. Marge restante, et ce que le temps par pas implique pour le budget de W3.)

---

## Bloc 5 — Évaluation de la recette sur le checkpoint de base

**Question.** Le chemin d'évaluation de la recette tourne-t-il tel quel sur le checkpoint publié ? (Inconnue 16 ; D-016.)

**Valeur observée.**

| Mesure | Valeur |
|---|---|
| Bloc lancé ? Sinon, pourquoi | |
| Taille du morceau PAI téléchargé (`du -sh $PAI_DIR`) | |
| Événements évalués, couverture (`eval_samples`, `eval_generation_coverage`) | |
| minADE de la recette (`eval_min_ade` : XY, 6,4 s, 6 candidats, prompt trajectoire seule) **(non publiable)** | |
| Format du prompt (`evaluation_protocol.json` : `prompt_format`, `native_release_prompt`) | |
| Pic mémoire et durée | |

**Commande et JSON source.** `python -m alpamayo2edge_sft.evaluate_hf --config-name sft_eval_trajectory …` ; `eval_base/…/trajectory_summary.json`, `evaluation_protocol.json`.

**Conclusion.** (Le chemin est reproduit ou non, et ce qui a bloqué. Rappel : ce minADE ne se compare ni à celui de la fiche ni à celui d'`afh`, et il reste privé jusqu'à D-012.)

---

## Décisions à mettre à jour

**Question.** Que disent les mesures pour D-004, D-015, D-016 et pour le premier point d'extension de W3 ? Décision d'Ousseynou ; cette rubrique rassemble les faits.

**Valeur observée.**

| Décision | Mesure | Valeur |
|---|---|---|
| D-004 : LoRA ou non en W1 | bloc 4, `summary.native_full_finetune_fits`, pic et GPU | |
| D-016 : adopter la recette en config native | blocs 4 et 5 : la passe conjointe et l'évaluation tournent | |
| D-015 : Edge partage-t-il le défaut de 2 Super ? | bloc 2, CoC sous `blackout_all` et écart de trajectoire | |
| W3 : poids de l'instruction de navigation | bloc 3, `lateral_sign_flipped`, `cot_names_new_direction` | |
| Raisonnement adaptatif | bloc 1, CoC vides et distribution des longueurs | |

**Commande et JSON source.** Les quatre JSON de `runners/probe_*_edge.py` et le dossier `eval_base`.

**Conclusion.** (Ce qui est établi, ce qui reste à mesurer, et la séance ou le travail à froid qui suit : runner Edge dans `alpamayo-faithfulness`.)
