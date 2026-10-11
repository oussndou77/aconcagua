# Séance GPU 01 — tokens, mémoire, blackout 1.5

*Préparée le 2026-10-11, à froid. Rien ici n'a encore tourné sur GPU.*

**But.** Lever les inconnues 1, 4, 5, 6 et 8 de `docs/finetune_design.md` §4, donner les entrées de D-014 (recette officielle comme base W1) et du réexamen de D-003, et décider le go/no-go W3. Les observations se notent dans une copie de `docs/observations/TEMPLATE-session.md`.

**Ce qui a changé depuis la note de conception.** NVIDIA a publié `alpamayo-recipes/recipes/alpamayo2_sft` (#39). Le bloc 2 mesure donc la mémoire par le chemin d'entraînement officiel (processeur, collate et `forward` de la recette), plus par un exemple fait main. Revue : `docs/recipe_a2_sft_review.md`.

## Matériel et durée

| Élément | Choix | Pourquoi |
|---|---|---|
| GPU, bloc 2 | **1 × H200 141 GB, obligatoire** | les poids bf16 de 2 Super occupent ~68 GB ; la mesure décide D-014 et doit pouvoir tenir 7 caméras × 4 images. La sonde refuse un GPU de moins de 120 GiB |
| GPU, blocs 1, 3, 4 | 1 × H200, ou 1 × H100 80 GB accepté | inférence 2 Super sur un GPU (chemin par défaut de la release), inférence 1.5 (~24 à 48 GB), CPU pour le bloc 4 |
| Disque | volume persistant ≥ 250 GB monté sur `/workspace` | poids 2S ~68 GB + 1.5 ~22 GB, trois venvs ~10 GB chacun, cache PAI |
| CPU / RAM | ≥ 16 vCPU, ≥ 128 GB | compilation de flash-attn (deux fois), chargement des shards |
| Durée visée | **3 h 30** GPU, clôture comprise | blocs 1 à 3 en série sur le GPU, bloc 4 sur CPU en parallèle |

**Si le pod disponible est un H100 80 GB** : faire les blocs 0, 1, 3, 4 et la clôture, et reporter le bloc 2 à une séance H200 (il ne demande que le JSON du bloc 1 et le venv de la recette). Ne pas lancer le bloc 2 sur H100 avec `--allow-small-gpu` pour la décision : ce drapeau existe pour le débogage, la mesure obtenue ne compte pas pour D-014.

| Bloc | Contenu | Durée visée | Inconnues |
|---|---|---|---|
| 0 | Setup : pod, accès, trois environnements, contrôles | 45 min | 9 |
| 1 | Sonde tokens 2 Super (prompt d'inférence de la release) | 20 min | 1, 4, (11) |
| 2 | Mémoire d'un pas d'entraînement par le chemin officiel | 45 min | 5, 6, 8 |
| 3 | Blackout sur Alpamayo 1.5 | 40 min | go/no-go W3 |
| 4 | Manifeste W3 sur CPU (en parallèle des blocs 1–3) | 15 min | — |
| Clôture | Rapatriement, gabarit rempli, arrêt du pod | 20 min | — |

Versions épinglées pour toute la séance :

| Dépôt | Commit | Rôle |
|---|---|---|
| `NVlabs/alpamayo-recipes` | `3f6504e` | environnement 2 Super des blocs 1 et 2 ; épingle `NVlabs/alpamayo2` `9596749` dans son `uv.lock` |
| `NVlabs/alpamayo1.5` | `36aeb4c` | environnement du bloc 3 |
| `oussndou77/alpamayo-faithfulness` | `7fff9e6` | `afh` (blocs 3 et 4) |
| `oussndou77/aconcagua` | la branche de cette séance | sondes et runner |

`helper.py` et `chat_template/conversation.py` d'`alpamayo2` sont identiques entre `9596749` (épinglé par la recette) et `5977692` (main) : le bloc 1 exerce le même prompt d'inférence que la release actuelle. #18 ne change que l'extraction de texte des tâches VQA et auto-labeling.

---

## Bloc 0 — Setup (45 min)

```bash
# 0.1 Pod et caches (à refaire après chaque reconnexion : utiliser tmux)
tmux new -s s01
export WS=/workspace
export HF_HOME=$WS/.hf_cache UV_CACHE_DIR=$WS/.uv_cache
export OUT=$WS/aconcagua/outputs/session01
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvcc --version | tail -1
hf auth login && hf auth whoami        # accès APPROUVÉ à nvidia/Alpamayo2-Super, nvidia/Alpamayo-1.5-10B, au dataset PAI

# 0.2 Dépôts épinglés
cd $WS
git clone https://github.com/NVlabs/alpamayo-recipes.git && git -C alpamayo-recipes checkout 3f6504e
git clone https://github.com/NVlabs/alpamayo1.5.git      && git -C alpamayo1.5 checkout 36aeb4c
git clone https://github.com/oussndou77/aconcagua.git    && git -C aconcagua checkout <branche de la séance>
mkdir -p $OUT

# 0.3 Environnement 2 Super = celui de la recette (blocs 1 et 2)
cd $WS/alpamayo-recipes/recipes/alpamayo2_sft
export UV_PROJECT_ENVIRONMENT=$WS/venvs/a2sft
time uv sync --locked                                   # compile flash-attn 2.8.3.post1 (5–10 min, MAX_JOBS=8 si la RAM sature)
#   repli si la compilation échoue : uv sync --locked --no-install-package flash-attn
#   (la recette ne sélectionne pas flash-attn : transformers prend sdpa par défaut)
$WS/venvs/a2sft/bin/python -c "import torch, transformers, alpamayo2_super, alpamayo2_sft; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"

# 0.4 Environnement 1.5 (bloc 3), lancé en arrière-plan pendant le bloc 1
cd $WS/alpamayo1.5
( UV_PROJECT_ENVIRONMENT=$WS/venvs/a15 uv sync --locked > $OUT/a15_sync.log 2>&1 \
  && uv pip install --python $WS/venvs/a15/bin/python \
       "alpamayo-faithfulness @ git+https://github.com/oussndou77/alpamayo-faithfulness.git@7fff9e6" \
       >> $OUT/a15_sync.log 2>&1; echo "a15 sync exit $?" >> $OUT/a15_sync.log ) &

# 0.5 Environnement CPU d'afh (bloc 4) : venv standard du dépôt
cd $WS/aconcagua && python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt \
  && PYTHON=.venv/bin/python bash scripts/install_afh.sh && .venv/bin/python -m pytest -q tests
```

**Pourquoi trois environnements.** Les deux verrous épinglent les mêmes `torch 2.8.0` et `transformers 4.57.1`, mais divergent ailleurs (vérifié à sec avec `uv sync --locked --dry-run`) :

| Paquet | `alpamayo1.5` (`uv.lock`) | recette `alpamayo2_sft` (`uv.lock`) | `alpamayo2` (`uv.lock`) |
|---|---|---|---|
| Python | 3.12 | 3.12 | 3.12 |
| torch / torchvision | 2.8.0 / 0.23.0 | 2.8.0 / 0.23.0 | 2.8.0 / 0.23.0 |
| transformers | 4.57.1 | 4.57.1 | 4.57.1 |
| flash-attn | 2.8.3 | 2.8.3.post1 | 2.8.3 |
| physical-ai-av | **== 0.2.0** (épinglage exact du `pyproject.toml`) | 0.2.2 | 0.2.2 |
| numpy | 2.3.5 | 2.5.1 | 2.4.6 |
| pandas | 2.3.3 | 3.0.5 | 3.0.3 |
| accelerate | 1.12.0 | 1.14.0 | 1.13.0 |
| deepspeed | — | 0.19.1 | — |
| Paquets résolus | 111 | 122 | 160 |

`physical-ai-av == 0.2.0` côté 1.5 contre 0.2.2 côté 2 Super, et pandas 2 contre pandas 3, interdisent un venv commun sans sortir des verrous. `afh` (dépendance `numpy>=1.24` seulement) s'installe dans le venv 1.5 sans rien déplacer. Les deux flash-attn sont des versions différentes : deux compilations, d'où le lancement du venv 1.5 en arrière-plan. Les venvs `uv` n'ont pas de `pip` : on installe avec `uv pip install --python <venv>/bin/python`.

**À noter.** Modèle de GPU, mémoire totale, pilote, `nvcc` ; durée de `uv sync` et succès ou repli de flash-attn (inconnue 9) ; commits effectivement extraits.

---

## Bloc 1 — Sonde tokens 2 Super (20 min)

Chemin d'inférence de la release, inchangé, sur `0ea6fd88` à `t0 = 5,1 s`. La sonde ne présuppose aucun format : elle rapporte les 10 premiers jetons générés et dit si `<|cot_start|>` apparaît, où, ou qu'il est absent.

```bash
cd $WS/aconcagua
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$WS/venvs/a2sft/bin/python runners/probe_tokens_a2.py --out $OUT/tokens_a2_0ea6fd88.json
# si le temps le permet (inconnue 11) :
$WS/venvs/a2sft/bin/python runners/probe_tokens_a2.py --n-runs 2 --repro-check \
    --skip-meta-action-task --out $OUT/tokens_a2_repro.json
```

Lecture rapide :

```bash
python3 -c "import json;d=json.load(open('$OUT/tokens_a2_0ea6fd88.json'));print(json.dumps(d['summary'],indent=1));print(d['prompt']['tail_tokens'][-6:]);print([t['token'] for t in d['runs'][0]['first_tokens']])"
```

**À noter.** `summary.cot_start_statement` et `runs[0].first_tokens` (inconnue 1) ; `prompt.tail_tokens` (le prompt se termine-t-il par `assistant\n` sans balise, comme le lit le code) ; `runs[0].marker_positions` ; `summary.meta_actions_found` et `meta_action_task.meta_actions` (inconnue 4) ; `token_ids.all_match` ; `runs[0].ade_vs_logged_future_m` et `peak_memory_gib` ; `reproducibility` si lancé. Le fichier sert d'entrée au bloc 2 (`summary.cot_for_training`).

**Si ça échoue.** Accès refusé au modèle ou au dataset : vérifier l'approbation HF, pas la commande. OOM au chargement sur 80 GB : noter, passer au bloc 3 (le bloc 2 ne passera pas non plus).

---

## Bloc 2 — Mémoire d'un pas d'entraînement, chemin officiel (45 min)

**H200 obligatoire.** La sonde lit la mémoire totale du GPU et s'arrête en dessous de 120 GiB, avec un message qui renvoie à ce paragraphe.

Le modèle est `TrainableAlpamayo2Super` (`training_stage="vlm"`), l'exemple passe par `Alpamayo2SuperProcessor.preprocess` puis `Alpamayo2SuperCollator` (masque sur toute la réponse de l'assistant, padding fixe à 6 144), la perte est le `forward` de la recette. Trois écarts assumés et écrits dans le JSON : un seul GPU sans ZeRO-3 ; gradients sur `q_proj` des 4 dernières couches du LM seulement (proxy de LoRA, D-004) ; `cot` ajouté à `components_order`.

**D'où vient le CoC de l'exemple** (champ `training_path.cot` du JSON) :

1. normalement, du bloc 1 : c'est le CoC que le modèle a lui-même généré sur `0ea6fd88` (`summary.cot_for_training`), donc un texte réel (`kind: real`) ;
2. si le bloc 1 n'en a pas produit, un texte de remplacement (`kind: synthetic`) fait de phrases de CoC réelles, dont la longueur en jetons est le **maximum** d'une distribution réelle : les CoC de toutes les exécutions du bloc 1 s'il y en a, sinon les 15 CoC d'Alpamayo 2 Super des fixtures `afh` (`records_a2.json`, 8 à 30 mots). Le maximum, parce qu'on mesure de la mémoire. La distribution, la statistique et la longueur mesurée sont écrites dans le JSON ;
3. `--cot-text` force un texte (`kind: user`).

**Pas de troncature.** La longueur réelle de la séquence est mesurée avant padding (`steps.sequence_before_padding.real_len`). Au-delà de 6 144 jetons, la sonde lève `SequenceTooLong` et s'arrête : elle ne tronque pas et ne réduit pas l'entrée pour faire tenir une séquence trop longue. Après le collate, elle vérifie que le lot paddé contient exactement les jetons réels d'avant le padding.

```bash
cd $WS/aconcagua
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$WS/venvs/a2sft/bin/python runners/probe_train_memory_a2.py \
    --tokens-json $OUT/tokens_a2_0ea6fd88.json --out $OUT/train_memory_a2.json
```

En cas d'OOM, et seulement d'OOM, la sonde libère la mémoire et réessaie seule, dans cet ordre : 7 caméras × 4 images (défaut de la recette), 6 × 4 (profil trajectoire de la release), 4 × 4 (défaut 1.5), 2 × 4, 1 × 4, 1 × 2, 1 × 1 (config de debug de la recette). Elle s'arrête à la première qui passe et l'indique dans `summary.retained_config`. Toute autre erreur, séquence trop longue comprise, arrête la sonde avec son message. Le JSON est réécrit après chaque étape.

Si le temps le permet, deux mesures complémentaires :

```bash
# la courbe complète mémoire / longueur de séquence
$WS/venvs/a2sft/bin/python runners/probe_train_memory_a2.py --sweep \
    --tokens-json $OUT/tokens_a2_0ea6fd88.json --out $OUT/train_memory_a2_sweep.json
# la cible telle que la recette la livre (trajectoire seule, sans CoC)
$WS/venvs/a2sft/bin/python runners/probe_train_memory_a2.py --no-cot --out $OUT/train_memory_a2_nocot.json
```

Lecture rapide :

```bash
python3 -c "import json;d=json.load(open('$OUT/train_memory_a2.json'));print(json.dumps(d['summary'],indent=1));a=[x for x in d['attempts'] if x['status']=='ok'];print(a and a[0]['steps']['sequence']['assistant_first_tokens'])"
```

**À noter.** `env.gpu` et `env.gpu_requirement` ; `training_path.cot` : `kind`, `source`, `tokens` et, si synthétique, `length_distribution` ; `summary.weights_gib` ; `summary.retained_config` ; pour chaque tentative : statut, étape qui a échoué, `steps.sequence_before_padding.real_len`, part visuelle, pic mémoire après forward et après backward (inconnues 6 et 8) ; `forward.loss`, `future_traj_ce`, `others_ce` finies et normes de gradient non nulles (inconnue 5) ; `sequence.assistant_first_tokens` (format de la cible CoC produite par le template officiel, à comparer au bloc 1 pour D-008 / D-011). **Critère D-014** : si une configuration passe sur le H200 avec ces gradients de taille PEFT, W1 part de 2 Super sur le code de la recette et 1.5 devient le repli ; sinon D-003 tient.

---

## Bloc 3 — Blackout sur Alpamayo 1.5 (40 min)

Libérer le GPU (aucun processus des blocs 1–2 encore vivant : `nvidia-smi`). Vérifier que la synchro du venv 1.5 lancée au bloc 0 a fini : `tail -2 $OUT/a15_sync.log` doit montrer `a15 sync exit 0`. Si flash-attn n'a pas compilé : `UV_PROJECT_ENVIRONMENT=$WS/venvs/a15 uv sync --locked --no-install-package flash-attn` dans `$WS/alpamayo1.5`, puis réinstaller `afh` comme au bloc 0. Le runner passe `--attn sdpa` par défaut, donc flash-attn n'est pas requis.

```bash
cd $WS/aconcagua
$WS/venvs/a15/bin/python runners/probe_blackout_a15.py \
    --k-rollouts 5 --null-control --dump-frames $OUT/frames_a15 \
    --out $OUT/cf_0ea6fd88_a15_blackout.json
```

Contrôle visuel avant de lire les chiffres : `$OUT/frames_a15/blackout_cam*.png` doivent être entièrement noirs, `baseline_cam*.png` non.

**À noter.** `verdict`, `baseline_citation`, `cf_citation` ; `cf_uncertainty.clear_path_rate` et `no_input_admitted_rate` (le défaut de NVlabs/alpamayo2#9 est-il présent sur 1.5 ?) ; `exceeds_noise_floor` ; `cf_traces` mot pour mot ; `baseline_min_ade`. Comparer champ à champ avec `fixtures/cf_0ea6fd88_a2_blackout.json` d'`afh`.

---

## Bloc 4 — Manifeste W3 sur CPU (15 min, en parallèle)

Sur le CPU du pod pendant que les blocs 1–3 occupent le GPU (ou sur un portable). Il valide la chaîne cibles → manifeste avec l'ordre de caméras de la recette officielle (7 caméras, `camera_indices` 0–6, D-010).

```bash
cd $WS/aconcagua
F=.deps/alpamayo-faithfulness/fixtures
.venv/bin/python -m afh.uncertainty_dataset split \
    --records $F/records_a2.json --diag $F/raw_diag_a2.json \
    --camera-indices 0,1,2,3,4,5,6 --test-fraction 0.3 \
    --holdout-family desync --holdout-combo glare+blur --composite-fraction 0.3 \
    --out-dir $OUT/w3_split
.venv/bin/python -m afh.uncertainty_dataset summary $OUT/w3_split/train.jsonl
.venv/bin/python -m afh.uncertainty_dataset summary $OUT/w3_split/test.jsonl
```

Vérifié à froid le 2026-10-11 : sur les trois clips des fixtures, 32 exemples d'entraînement sur 2 clips (47 % propres) et 20 de test sur 1 clip, avec 4 exemples tenus à part ; `split` lance lui-même `check_split` (aucune famille ni combinaison tenue à part ne fuit dans l'entraînement).

**À noter.** Les comptes par famille et la fraction propre des deux résumés ; tout message d'erreur de `check_split`. Le format de manifeste n'est pas encore celui du `PhysicalAIDataset` de la recette : l'adaptateur est l'étape 1 de D-014, pas cette séance.

---

## Clôture (20 min)

```bash
cd $WS/aconcagua
SESSION=docs/observations/$(date +%Y-%m-%d)-session01
mkdir -p $SESSION && cp docs/observations/TEMPLATE-session.md $SESSION/README.md
cp $OUT/*.json $OUT/*.log $SESSION/ 2>/dev/null; cp -r $OUT/w3_split $SESSION/
tar czf $WS/session01_outputs.tgz -C $OUT .            # copie complète, PNG compris
git checkout -b observations/session01 && git add $SESSION && git commit -m "GPU session 01 observations"
git push -u origin observations/session01
nvidia-smi    # puis arrêter le pod ; les poids restent dans $HF_HOME sur le volume
```

Remplir `$SESSION/README.md` rubrique par rubrique, à partir des JSON uniquement. Le tableau des inconnues de `docs/finetune_design.md` et D-008 / D-011 / D-014 se mettent à jour ensuite, dans une PR séparée, à partir de ce dossier.
