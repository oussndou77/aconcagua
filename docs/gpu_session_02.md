# Séance GPU 02 — Alpamayo 2 Edge : sortie brute, blackout, navigation, mémoire

*Préparée le 2026-11-22, à froid. Rien ici n'a encore tourné sur GPU. La séance 01 (2 Super, 1.5) n'a jamais eu lieu ; ses documents et ses sondes sont conservés.*

**But.** Mesurer le modèle de base choisi par D-015 avant d'y toucher : ce qu'Edge génère, ce qu'il fait sans caméras, s'il suit l'instruction de navigation, et si un pas d'entraînement conjoint complet tient sur un GPU. Lever les inconnues 6, 8, 11 et 13 à 16 de `docs/finetune_design.md` §4.1. Les observations se notent dans une copie de `docs/observations/TEMPLATE-session02.md`.

**Ce que la séance décide.**

| Mesure | Décision qu'elle alimente |
|---|---|
| Bloc 4 : le plein fine-tuning conjoint tient-il sur 80 Go avec l'entrée native ? | D-004 : si oui, pas de LoRA en W1 |
| Blocs 4 et 5 : le chemin de la recette tourne-t-il chez nous ? | D-016 : adopter `alpamayo2edge_sft` en config native |
| Blocs 1 et 2 : Edge raconte-t-il une scène qu'il ne voit pas ? | ROADMAP §1, référence d'Edge pour le critère d'incertitude |
| Bloc 3 : la sortie dépend-elle de l'instruction de navigation ? | ROADMAP W3, premier point d'extension |

## Matériel et durée

| Élément | Choix | Pourquoi |
|---|---|---|
| GPU | **1 × H100 80 Go, recommandé** ; A100 80 Go en repli | le H100 80 Go est le seul GPU que NVIDIA déclare testé pour Edge. L'inférence tient dans 12 Gio selon la fiche ; c'est le bloc 4 qui demande 80 Go. La sonde mémoire refuse un GPU de moins de 75 Gio |
| Disque | disque conteneur de 100 Go | environnement de la recette, modèle (5,3 Go), cache des 19 échantillons (environ 3 Go), un morceau de PAI pour le bloc 5 (de l'ordre de 7 Go : la recette annonce 181 Gio pour 26 morceaux) |
| CPU / RAM | ≥ 8 vCPU, ≥ 48 Go | le bloc 4 charge le modèle en FP32 sur CPU (environ 10 Go) avant de le passer au GPU |
| Durée visée | **3 h** GPU, clôture comprise | blocs en série ; le bloc 5 saute si le temps manque |

Sur A100 80 Go, tout fonctionne en principe (BF16 pris en charge), mais ce n'est pas une configuration validée par NVIDIA : une différence de sortie avec un H100 ne serait pas une anomalie de nos sondes. Chaque sonde enregistre le GPU et `bf16_supported` dans `env.gpu`.

| Bloc | Contenu | Durée visée | Inconnues |
|---|---|---|---|
| 0 | Setup : pod, modèle en parallèle, environnement de la recette, contrôles | 30 min | — |
| 1 | Inférence sur les 19 événements : sortie brute, longueur de la CoC, trajectoire | 30 min | 4, 8, 11, 13, 15 |
| 2 | Blackout : toutes caméras, puis caméra frontale seule | 30 min | 13 |
| 3 | Contrefactuel de navigation | 30 min | 14 |
| 4 | Mémoire d'un pas d'entraînement conjoint, un GPU | 20 min | 6, 8 |
| 5 | Si le temps le permet : `evaluate_hf.py` de la recette sur le checkpoint de base | 30 min | 16 |
| Clôture | Rapatriement, gabarit rempli, arrêt du pod | 15 min | — |

Versions épinglées pour toute la séance (D-016) :

| Source | Révision | Rôle |
|---|---|---|
| `NVlabs/alpamayo-recipes` | `622f9d23ff9e4961ba8c6aac71367776cdf6b016` | recette `alpamayo2edge_sft` ; son `uv.lock` fixe l'environnement |
| `NVlabs/alpamayo2-edge` | `02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e` | inférence ; c'est le commit que la recette épingle elle-même |
| `nvidia/Alpamayo2-Edge` (HF) | `f8951d095de3589b3face1dec3f4394c8ecedb38` | poids. La recette de déploiement Thor épingle `aaaa32c` : écart non examiné, sans effet sur cette séance |
| `nvidia/PhysicalAI-Autonomous-Vehicles` (HF) | `33f9bf447ed3bcb7d545ce13f4226f824214fafb` | données ; révision de la recette |
| `oussndou77/aconcagua` | la branche de cette séance | sondes |

Les 19 événements sont ceux de `recipes/alpamayo2edge_sft/assets/nav_demo_samples.json` : 18 clips, une instruction de navigation par événement (16 « Turn left in Xm », 3 « Turn right in Xm ») et une CoC générée par Edge puis relue par NVIDIA. Ce sont les seuls événements publics qui portent une instruction de navigation au format de la release. Nos fixtures (`0ea6fd88`) n'en ont pas et ne servent pas dans cette séance.

**Données.** Les blocs 1 à 4 lisent PAI en flux depuis Hugging Face (`physical_ai_av`, sans téléchargement des archives : les 18 morceaux pèseraient 147 Gio). Chaque échantillon chargé est mis en cache dans `$OUT/samples` et resservi aux blocs suivants.

---

## Bloc 0 — Setup (30 min)

```bash
# 0.1 Pod et caches (à refaire après chaque reconnexion : utiliser tmux)
tmux new -s s02
export WS=/workspace
export HF_HOME=$WS/.hf_cache UV_CACHE_DIR=$WS/.uv_cache UV_HTTP_TIMEOUT=900
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
read -rs HF_TOKEN && export HF_TOKEN      # accès APPROUVÉ à nvidia/Alpamayo2-Edge et au dataset PAI
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvcc --version | tail -1
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"

# 0.2 Modèle, lancé tout de suite en arrière-plan (5,3 Go)
export EDGE_MODEL=$WS/checkpoints/alpamayo2-edge
(uvx --from "huggingface_hub>=1.0" hf download nvidia/Alpamayo2-Edge \
    --revision f8951d095de3589b3face1dec3f4394c8ecedb38 \
    --local-dir "$EDGE_MODEL" > $WS/model_download.log 2>&1 &)

# 0.3 Dépôts épinglés
cd $WS
git clone https://github.com/NVlabs/alpamayo-recipes.git
git -C alpamayo-recipes checkout 622f9d23ff9e4961ba8c6aac71367776cdf6b016
git clone https://github.com/oussndou77/aconcagua.git && git -C aconcagua checkout <branche de la séance>
export REPO_DIR=$WS/alpamayo-recipes ACONCAGUA=$WS/aconcagua
export OUT=$ACONCAGUA/outputs/session02 && mkdir -p $OUT
export EVENTS=$REPO_DIR/recipes/alpamayo2edge_sft/assets/nav_demo_samples.json
export PAI_REVISION=33f9bf447ed3bcb7d545ce13f4226f824214fafb

# 0.4 Environnement = celui de la recette, tel que son README l'installe
cd $REPO_DIR/recipes/alpamayo2edge_sft
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
uv venv $WS/envs/a2e_sft --python 3.12 && source $WS/envs/a2e_sft/bin/activate
uv sync --active --locked 2>&1 | tee $OUT/uv_sync.log
```

C'est le `uv sync --locked` de la recette, appliqué au venv actif. Il installe `alpamayo2-edge` à `02f55a7`, `torch` 2.8.0, `transformers` 5.17.0, `physical-ai-av` 0.2.2 et les utilitaires partagés du dépôt (`alpamayo.data`). flash-attn n'est pas installé : Edge tourne en SDPA par défaut. `afh` n'est pas nécessaire dans cette séance.

Le paquet d'inférence d'Edge épingle `physical-ai-av` 0.2.0 dans son extra `dataset` ; la recette verrouille 0.2.2. Les deux versions ont été comparées à froid : seuls le mode hors-ligne et `download_file` diffèrent ; la lecture vidéo, l'egomotion et l'interpolation, que le chargeur d'Edge appelle, sont identiques.

```bash
# 0.5 Contrôles
python - <<'PY'
import torch, transformers
from importlib.metadata import version
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
print("transformers", transformers.__version__, "physical-ai-av", version("physical-ai-av"))
from alpamayo.data.pai_metadata import validate_navigation
print("validateur de navigation :", validate_navigation("Continue straight"))
import alpamayo2_edge, alpamayo2edge_sft
PY
python -m alpamayo2_edge.inference --help | head -3
python -c "import json; print(len(json.load(open('$EVENTS'))), 'événements')"      # 19

# 0.6 Contrôle d'API contre les sources réellement installées (9 tests, aucun ne doit être sauté)
cd $ACONCAGUA
EDGE_INSTALLED=1 RECIPES_REPO=$REPO_DIR python -m pytest tests/test_edge_pinned_api_cold.py \
    -q -rs --fail-on-skip > $OUT/api_check.log 2>&1 \
  && tail -1 $OUT/api_check.log \
  || { tail -25 $OUT/api_check.log; echo "CONTRÔLE D'API ÉCHOUÉ OU INCOMPLET : ne pas lancer les blocs"; false; }

# 0.7 Modèle : téléchargement fini, fichiers identiques à ceux lus dans la revue
tail -2 $WS/model_download.log && ls $EDGE_MODEL
for f in config.json chat_template.jinja generation_config.json; do
  cmp $EDGE_MODEL/$f $ACONCAGUA/docs/configs/alpamayo2-edge/$f && echo "$f identique à f8951d0"
done
```

**Le contrôle d'API (0.6).** Les sondes n'ont jamais tourné : elles ont été vérifiées à froid contre des clones épinglés. Ce contrôle refait la même vérification contre ce que les sondes vont réellement importer sur le pod : le paquet `alpamayo2_edge` installé dans le venv (commit lu dans son enregistrement d'installation) et la recette de `$REPO_DIR`. Il vérifie chaque fonction appelée et ses arguments, les constantes recopiées, les valeurs des configs, les 38 instructions de navigation contre le validateur, et que les paquets `alpamayo2edge_sft` et `alpamayo` du venv viennent bien de `$REPO_DIR`.

- Résultat attendu : `9 passed`, rien d'autre.
- `--fail-on-skip` fait échouer la commande dès qu'un seul test est sauté : un contrôle qui n'a pas tourné ne doit pas ressembler à un contrôle réussi.
- En cas d'échec, ne pas lancer les blocs : lire `$OUT/api_check.log`, noter le test et le message, et s'arrêter là. Un échec veut dire que le code installé n'est pas celui que les sondes attendent.

**À noter dans le gabarit.** GPU, pilote, version de `nvcc`, durée de `uv sync`, la dernière ligne de `api_check.log`, résultat des trois `cmp`. Un `cmp` qui échoue veut dire que la revue a lu une autre révision que celle téléchargée : s'arrêter et le signaler.

**Si `uv sync` échoue** sur un téléchargement : relancer la même commande (le cache `UV_CACHE_DIR` reprend). S'il échoue à la compilation d'un paquet, noter le paquet et le message ; ne pas remplacer le verrou.

---

## Bloc 1 — Sortie brute sur les 19 événements (30 min)

**Question.** Sur le prompt natif de la release, que génère Edge ? La sonde ne présuppose aucun format : elle enregistre les jetons bruts avant que la release ne cherche l'ancre de trajectoire.

```bash
cd $ACONCAGUA
python runners/probe_tokens_edge.py --events $EVENTS --model-id $EDGE_MODEL \
    --sample-cache $OUT/samples --repro-check \
    --out $OUT/tokens_edge.json 2>&1 | tee $OUT/tokens_edge.log
```

Par événement : 6 candidats (chacun sa CoC), graine 43, 256 jetons au plus, 10 pas de diffusion. Ce sont les réglages par défaut de la CLI d'Edge, sauf le nombre de candidats (1 par défaut, 6 ici comme dans la mesure mémoire de la fiche).

**Ce que la sonde enregistre.**

- La fin du prompt et sa longueur (`conditions.clean.prompt`) : finit-il par `<think></think>` ? Combien de jetons d'image, contre la prédiction à froid de 2 892 (`load.predicted_visual_tokens_native`) ?
- Pour chaque candidat (`conditions.clean.candidates[k]`) : les 12 premiers jetons, la position de `<|traj_future_start|>`, la longueur du raisonnement avant elle (`reasoning_tokens`, **zéro compris**), le texte (`cot`), la présence de balises `<think>`, d'un EOS, de jetons de trajectoire discrets, de lignes de méta-actions.
- L'extraction de la release, pour comparaison (`release_extraction.cot`).
- Les trajectoires (`pred_xyz`, 6 × 64 × 3), leur dispersion entre candidats (`within_spread_m`), le pic mémoire et le temps.
- Sur le premier événement, une génération **sans** les filtres de la release (`conditions.unconstrained`) : la release interdit les jetons de trajectoire et l'EOS avant l'ancre ; cette passe montre ce que le modèle émet quand rien ne le guide.
- Avec `--repro-check` : le premier événement rejoué à la même graine (`reproducibility`).

**Cas prévus.** Un événement qui ne se charge pas (par exemple l'événement 15, à `t0` = 2,0 s exactement, borne basse de la fenêtre) est enregistré avec son erreur et la sonde continue. Une génération qui n'atteint pas l'ancre fait échouer la release ; la sonde l'enregistre avec le statut `no_anchor` et garde les jetons.

**À noter.** `summary.reasoning` (candidats, raisons d'arrêt, CoC vides, longueurs min / médiane / max), `summary.events_without_anchor`, `summary.prompt_len`, `token_ids.all_match`, `reproducibility`, et trois CoC mot pour mot.

---

## Bloc 2 — Blackout (30 min)

**Question.** Privé de caméras, Edge décrit-il encore une scène, et sa trajectoire change-t-elle ? C'est le défaut observé sur 2 Super (NVlabs/alpamayo2#9) ; rien ne dit encore si Edge le partage.

```bash
python runners/probe_blackout_edge.py --events $EVENTS --model-id $EDGE_MODEL \
    --sample-cache $OUT/samples --baseline-json $OUT/tokens_edge.json \
    --out $OUT/blackout_edge.json 2>&1 | tee $OUT/blackout_edge.log
```

Trois conditions, dans cet ordre, sur les mêmes événements, la même graine et le même chemin que le bloc 1 :

| Condition | Images noircies | Ce qui reste |
|---|---|---|
| `blackout_all` | les 22 images, toutes caméras | rien |
| `blackout_front_both` | les 10 images des deux caméras avant : grand angle (identifiant 1) et téléobjectif (identifiant 6) | les 12 images des caméras latérales et arrière |
| `blackout_front` | les 5 images de la caméra frontale grand angle (identifiant 1) | le téléobjectif avant et les quatre autres caméras |

`blackout_front_both` sépare deux lectures de `blackout_front` : si le modèle conduit encore normalement sans le grand angle, est-ce parce qu'il se passe de la vue avant, ou parce que le téléobjectif la remplace ?

Les images sont choisies par identifiant de caméra sur la liste à plat d'Edge. `afh.degradation` n'est pas utilisé : il suppose un tenseur caméras × temps (mise à jour de D-010).

**Ce que la sonde ajoute aux sorties du bloc 1** (`conditions.<condition>.deviation_vs_clean`) : la distance entre les trajectoires moyennes, la distance appariée candidat par candidat (même graine), et la dispersion entre candidats du run propre, qui sert de plancher. `exceeds_baseline_spread` dit seulement si l'écart dépasse ce plancher. Les CoC propres et noircies sont gardées côte à côte, mot pour mot ; la sonde ne les note pas.

**À noter.** Pour chacune des trois conditions : `n_exceeding_clean_spread` sur `n_compared_with_spread`, `mean_ade_between_means_m` contre `mean_clean_within_spread_m`, les longueurs de parcours propre et noircie (le modèle ralentit-il ?), et cinq paires de CoC mot pour mot. Lire les CoC de `blackout_all` et de `blackout_front_both` : parlent-elles de feux, de véhicules, de voie libre devant ?

---

## Bloc 3 — Contrefactuel de navigation (30 min)

**Question.** Même événement, mêmes images, instruction changée : la trajectoire et la CoC suivent-elles l'instruction ? L'instruction est une entrée obligatoire de la tâche native et PAI public ne la fournit pas ; ce bloc dit combien il faudra de soin pour en produire sur nos événements.

```bash
python runners/probe_nav_edge.py --events $EVENTS --model-id $EDGE_MODEL \
    --sample-cache $OUT/samples --baseline-json $OUT/tokens_edge.json \
    --out $OUT/nav_edge.json 2>&1 | tee $OUT/nav_edge.log
```

Deux instructions contrefactuelles par événement, dans la syntaxe de la release :

| Étiquette | Instruction | Exemple |
|---|---|---|
| `mirror` | gauche et droite échangées, distance gardée | « Turn left in 11m » → « Turn right in 11m » |
| `straight` | « Continue straight » | |

Chaque instruction passe par le validateur de la recette (`validate_navigation`) avant usage ; une instruction refusée n'est pas lancée, et un validateur indisponible est enregistré (`instructions_not_validated`).

**Ce que la sonde ajoute** (`conditions.<étiquette>.follow`, `deviation_vs_clean`) : l'écart de trajectoire au run propre, le point d'arrivée latéral des deux, pour `mirror` le changement de signe du latéral quand les deux dépassent 1 m, et les mots de direction (left, right, straight) employés par la CoC propre et contrefactuelle.

**À noter.** Par étiquette : `lateral_sign_flipped`, `trajectory_beyond_clean_spread`, `cot_names_new_direction`, `cot_still_names_old_direction`, et cinq paires de CoC. Une instruction « Turn right » sur une scène sans voie à droite n'a pas de bonne réponse : ce bloc mesure une dépendance, pas une qualité de conduite.

---

## Bloc 4 — Mémoire d'un pas d'entraînement conjoint (20 min)

**Question.** Le plein fine-tuning conjoint de la recette (VLM et expert entraînés, poids maîtres en FP32, calcul en BF16) tient-il sur un GPU de 80 Go avec l'entrée native, 6 caméras et 22 images, sans DeepSpeed ? L'arithmétique donne 39,5 Gio d'états avant toute activation (`docs/edge_review.md` §5).

```bash
python runners/probe_train_memory_edge.py --events $EVENTS --model-dir $EDGE_MODEL \
    --sample-cache $OUT/samples \
    --out $OUT/train_memory_edge.json 2>&1 | tee $OUT/train_memory_edge.log

# si le temps le permet : toutes les configurations, pour la courbe mémoire / longueur
python runners/probe_train_memory_edge.py --events $EVENTS --model-dir $EDGE_MODEL \
    --sample-cache $OUT/samples --sweep \
    --out $OUT/train_memory_edge_sweep.json 2>&1 | tee $OUT/train_memory_edge_sweep.log
```

**Chemin.** Les classes de la recette, sans réécriture : le modèle est instancié depuis `configs/model/release.yaml` (`TrainableAlpamayo2Edge`, étape `joint`), l'exemple passe par `Alpamayo2EdgeProcessor.preprocess` avec `configs/preprocess/nav_cot.yaml`, puis par `Alpamayo2EdgeCollator` (plafond de 8 192 jetons, padding à la plus longue séquence). Le pas : passe avant sous autocast BF16, passe arrière, écrêtage du gradient à 1,0, pas d'AdamW sur les poids FP32, aux réglages de `sft_base.yaml`. Trois pas par configuration.

**Exemple.** L'événement 0 de l'asset : ses 22 images et son histoire par le chargeur d'inférence, sa navigation et sa CoC depuis l'asset, ses poses futures depuis le dataset. La sonde l'assemble elle-même parce que la classe de dataset de la recette exige un téléchargement local de PAI ; le prétraitement, le collate et la passe sont ceux de la recette.

**Repli.** Même logique que la sonde 2 Super. En cas d'OOM ou de séquence au-delà de 8 192 jetons, la sonde enregistre la raison, libère la mémoire et essaie l'entrée suivante : 22 images (native), 12 (les 2 dernières par caméra), 6 (la dernière par caméra), 2 (les deux caméras avant), 1 (config de debug de la recette). Une séquence trop longue est un constat, avec sa longueur mesurée (`summary.findings`). Chaque configuration abandonnée garde sa raison (`summary.abandoned`). Celle qui passe est marquée réduite avec ses raisons (`summary.retained_reduced`, `summary.reduced_because`). Aucune troncature. Toute autre erreur arrête la sonde.

**Ce que la sonde mesure.** La longueur réelle de la séquence et les jetons visuels contre la prédiction à froid ; les jetons supervisés et le début du texte supervisé (`steps.sequence.supervised_text_head`, qui doit commencer par `<think></think>` puis la CoC) ; les pertes (`loss`, `vlm_ce`, `future_traj_ce`, `others_ce`, `expert_diffusion`) et la norme du gradient ; le pic mémoire après la passe avant, la passe arrière et le pas d'optimiseur ; le temps par pas.

**Règle de décision (D-004).** `summary.native_full_finetune_fits` vrai sur un GPU de 80 Go, sans `--allow-small-gpu` : pas de LoRA en W1. Faux : D-004 est rouverte pour Edge, avec le pic atteint et la configuration qui passe comme entrées. `--allow-small-gpu` existe pour déboguer sur un plus petit GPU ; la mesure obtenue ne compte pas pour D-004.

**Vérité terrain.** La trajectoire future de l'exemple sert à calculer la perte et rien d'autre : aucune pose n'est écrite dans le JSON (`example.target_trajectory.values_written` vaut `false`). Les pertes sont des chiffres calculés contre cette trajectoire : elles servent à vérifier que le pas est sain, et ne sont pas publiables (voir « Ce qui est publiable »).

**À noter.** Le tableau des tentatives, `summary.weights_gib`, les trois pics du premier pas de la configuration retenue, `seconds_per_step`, et `recipe.preprocess_matches_pinned_copy`.

---

## Bloc 5 — Évaluation de la recette sur le checkpoint de base (30 min, si le temps le permet)

**Question.** Reproduit-on le chemin d'évaluation de la recette, tel quel, sur le checkpoint publié ? C'est la condition pour comparer plus tard un checkpoint fine-tuné à la base avec leur outil.

```bash
export PAI_DIR=$WS/data/pai
cd $REPO_DIR
python scripts/download_pai.py --revision "$PAI_REVISION" --profile edge --chunk-ids "1134-1135" \
    --camera camera_cross_left_120fov camera_front_wide_120fov camera_cross_right_120fov \
      camera_rear_left_70fov camera_rear_right_70fov camera_front_tele_30fov \
    --labels egomotion --output-dir "$PAI_DIR"
du -sh $PAI_DIR

cd $REPO_DIR/recipes/alpamayo2edge_sft
CUDA_VISIBLE_DEVICES=0 python -m alpamayo2edge_sft.evaluate_hf \
    --config-path pkg://alpamayo2edge_sft/configs --config-name sft_eval_trajectory \
    evaluate.eval_ckpt="$EDGE_MODEL" evaluate.max_eval_steps=-1 seed=42 \
    data.train_dataset.local_dir="$PAI_DIR" data.val_dataset.local_dir="$PAI_DIR" \
    data.val_dataset.chunk_ids="1134-1135" \
    paths.output_dir="$OUT/eval_base" 2>&1 | tee $OUT/eval_base.log
```

Un seul morceau de validation officielle (`1134-1135` est un intervalle semi-ouvert : le morceau 1134), pour tenir sur le disque ; la recette en évalue onze. La config `sft_eval_trajectory` est celle par défaut : prompt trajectoire seule, sans navigation ni CoC générée, 6 candidats, minADE en XY sur 6,4 s. **Ce n'est pas le prompt natif** : la variante native (`sft_eval_trajectory_nav_cot`) demande des annotations de navigation que nous n'avons pas pour ces clips. Le chiffre obtenu ne se compare ni au 0,52 m de la fiche (horizon 5 s, 1 392 échantillons) ni à un minADE d'`afh`.

**Non publiable.** Ce minADE est un chiffre de benchmark contre la vérité terrain de PAI : il reste privé jusqu'à la réponse de NVIDIA (D-012), comme tout le dossier `eval_base`.

**À noter.** Dans le sous-dossier créé sous `$OUT/eval_base` : `trajectory_summary.json` (`eval_min_ade`, `eval_samples`, `eval_generation_coverage`), `evaluation_protocol.json` (`prompt_format`, `native_release_prompt`, `candidate_count`), le pic mémoire journalisé, et la durée. Si le téléchargement demande une confirmation de taille, l'accepter et noter la taille.

---

## Clôture (15 min)

La séance ne publie rien : pas de commit, pas de branche, pas de `git push` depuis le pod. Elle produit une archive, à rapatrier **avant** d'arrêter le pod (le disque conteneur disparaît avec lui).

```bash
cd $WS
tar czf $WS/session02_outputs.tgz -C $OUT --exclude=samples .
tar tzf $WS/session02_outputs.tgz | grep -c "samples/" ; ls -lh $WS/session02_outputs.tgz    # 0, puis la taille
runpodctl send $WS/session02_outputs.tgz      # affiche un code ; `runpodctl receive <code>` sur ta machine
nvidia-smi                                    # arrêter le pod seulement une fois l'archive reçue et ouverte
```

Sans `runpodctl`, télécharger `session02_outputs.tgz` par l'explorateur de fichiers du pod ou par `scp`. Vérifier sur ta machine que l'archive s'ouvre et contient les quatre JSON des sondes avant d'arrêter le pod.

**Contenu de l'archive.** Les JSON et journaux des blocs 0 à 4, et le dossier `eval_base` du bloc 5. Le cache `$OUT/samples` en est exclu : il contient les images de PAI et les poses futures de l'exemple du bloc 4, qui ne quittent pas le pod.

**Ensuite, hors du pod.** Remplir une copie de `docs/observations/TEMPLATE-session02.md` à partir des JSON uniquement. Ce qui entre dans le dépôt se décide avec la section suivante. D-004, D-016 et le tableau des inconnues se mettent à jour dans une PR séparée.

---

## Ce qui est publiable

L'archive est privée dans son ensemble. Avant de publier quoi que ce soit (dépôt, issue, message à NVIDIA), trier ainsi :

| Publiable : sorties du modèle et mesures de la séance | Non publiable jusqu'à la réponse de NVIDIA (D-012) |
|---|---|
| Les CoC générées, mot pour mot, propres et sous condition | Tout chiffre de benchmark contre la vérité terrain de PAI, dont le minADE du bloc 5 et tout le dossier `eval_base` |
| Les trajectoires prédites (`pred_xyz`) et leurs résumés | Toute vérité terrain du dataset : poses d'ego passées ou futures, trajectoires enregistrées |
| Les écarts entre conditions (blackout, navigation) et la dispersion entre candidats : ce sont des distances entre sorties du modèle | Les images de PAI et le cache `samples/` |
| Les longueurs de raisonnement, les jetons générés, les longueurs de séquence | Les pertes du bloc 4 (`loss`, `vlm_ce`, `future_traj_ce`, `others_ce`, `expert_diffusion`) : elles sont calculées contre la trajectoire vraie |
| La mémoire et les temps (blocs 1 à 4), l'environnement, les commits | |

Les pertes du bloc 4 sont classées non publiables par prudence : ce ne sont pas des chiffres de benchmark, mais elles dépendent de la vérité terrain. Classement confirmé par Ousseynou le 2026-11-22 ; elles restent dans le JSON.

La règle est D-012 (`docs/DECISIONS.md`) : tant que NVIDIA n'a pas répondu par écrit, aucune donnée du dataset, aucun chiffre contre sa vérité terrain et aucun poids dérivé ne sont publiés ; les sorties du modèle le sont.

**Ce que les sondes garantissent.**

- Elles n'écrivent leur JSON que par `write_probe_json`, qui refuse toute clé nommant une vérité terrain (pose d'ego, futur enregistré, score contre le futur enregistré).
- Aucune pose n'est écrite : ni l'histoire d'ego, ni le futur. Les jetons d'histoire que contiendrait la fin du prompt rapportée sont retirés (`history_tokens_withheld_from_tail`).
- Les blocs 1 à 3 ne chargent aucun futur : leurs écarts comparent des sorties du modèle entre elles.
- Le bloc 4 charge le futur de son exemple pour la perte, et n'en écrit que le nombre de points.
- Ces garanties sont couvertes par les tests à froid (`tests/test_probes_edge_cold.py`).

Ce que les JSON contiennent du dataset sans être une vérité terrain : les identifiants de clip, les instants `t0`, les décalages de capture des images. Les instructions de navigation et la CoC de l'exemple viennent de l'asset public de la recette, pas de PAI.

---

## Ce que la séance ne fait pas

- **Pas d'occlusion contrefactuelle.** Elle demande la calibration des caméras, que le chargeur d'Edge ne charge pas, et le harnais `afh`, qui n'a pas de runner Edge.
- **Pas de notation des CoC par `afh`.** Les textes sont gardés mot pour mot ; les axes du harnais s'appliqueront quand le runner existera.
- **Pas de LoRA**, et pas d'entraînement au-delà de trois pas de mesure.
- **Pas d'évaluation au prompt natif** dans le bloc 5, faute d'annotations de navigation sur les clips de validation.
- **Pas de publication.** Ni commit ni `git push` depuis le pod ; l'archive est rapatriée, puis triée selon « Ce qui est publiable ».

## Tâche suivante : un runner Edge dans `alpamayo-faithfulness`

À écrire dans `oussndou77/alpamayo-faithfulness`, pas ici (D-010, mise à jour). Ce qu'il doit couvrir, d'après `docs/edge_review.md` §7 :

1. **Entrées à plat.** 22 images, un identifiant de caméra et un horodatage par image, 5 ou 3 images selon la caméra. `check_camera_indices` refuse aujourd'hui les identifiants répétés, et `apply_degradation` attend `(n_cam, n_t, 3, H, W)`.
2. **Dégradations par image.** `freeze` et `desync` raisonnent en nombre d'images sur un pas de temps régulier ; ceux d'Edge ne le sont pas (−2 ; −1 ; −0,5 ; −0,1 ; 0 s).
3. **Instruction de navigation** comme entrée de chaque événement, enregistrée dans le manifeste.
4. **Occlusion.** `occlude_frames` a besoin des intrinsèques et extrinsèques, à charger à part (`camera_intrinsics`, `sensor_extrinsics`), et d'horodatages par image.
5. **Environnement.** Celui d'Edge (`transformers` 5.17.0), distinct de ceux de 2 Super et de 1.5.
6. **Références à refaire.** Les fixtures `cf_0ea6fd88*.json` viennent de 2 Super, sans navigation, sur 6 caméras × 4 images : elles ne sont pas une référence pour Edge.

Les sorties des blocs 1 à 3 (CoC mot pour mot, trajectoires, écarts) donnent les premières références d'Edge pour ce runner.
