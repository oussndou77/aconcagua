# Revue d'Alpamayo 2 Edge : dépôt d'inférence et recette d'entraînement officielle

*Lecture du code, 2026-11-22. Aucune exécution de modèle, aucun GPU. Sert de source à D-015, à la réécriture de ROADMAP W1/W6 et au tableau des inconnues de `finetune_design.md` §4. Les contradictions avec ROADMAP, DECISIONS et `finetune_design.md` sont listées en §13, sans être tranchées.*

| Source | Révision lue | Objet |
|---|---|---|
| `NVlabs/alpamayo2-edge` | `02f55a7` (main, 2026-11-09) | inférence PyTorch ; `EDGE/` = racine du dépôt, `E/` = `src/alpamayo2_edge/` |
| `NVlabs/alpamayo-recipes` | `622f9d2` (main, 2026-11-10) ; recette ajoutée par `9c56cbd` (#50) | `recipes/alpamayo2edge_sft/`, abrégé `R/` |
| `nvidia/Alpamayo2-Edge` (HF) | `f8951d0`, fichiers déposés dans `docs/configs/alpamayo2-edge/` (commit `a214352`), abrégé `C/` | `config.json`, `chat_template.jinja`, fiche, licence |
| `transformers` | 5.17.0 (roue PyPI, lue sans l'installer), abrégé `T/` = `models/cosmos3_edge/` | backbone Cosmos3-Edge |
| `oussndou77/alpamayo-faithfulness` | `7fff9e6` (épinglé par ce dépôt), abrégé `AFH/` | `afh/cameras.py`, `afh/degradation.py`, fixtures |

La recette épingle exactement le commit d'inférence lu ici : `R/pyproject.toml:12` (`alpamayo2-edge @ …@02f55a7…`).

**Ce qui a été lu en entier** : tout `E/` hors `accel/`, `action_space/` (survolé) et `geometry/` ; `R/models/sft.py`, `R/processor.py`, `R/conversation.py`, `R/dataset.py`, `R/models/utils.py`, `R/evaluate_hf.py`, `R/models/public_inference.py`, `R/label_mask.py`, `R/train_hf.py`, `R/run_utils.py`, toutes les configs de `R/configs/`, le README de la recette. **Lu en partie** : `R/trainer.py` (optimiseur, précision), `R/compare_evaluations.py` (docstring et structure), les tests (noms et un test cité). **Non lu** : `R/driving_datasets.py`, `R/driving_splits.py`, `R/deepspeed_*.py`, `R/global_norm.py`, `E/accel/`. Les recettes voisines (§9) ne sont lues qu'en README et configs.

**Deux vérifications à froid faites ici** : le rendu du `chat_template.jinja` du checkpoint avec jinja2 3.1.6 (§2, §3), et le calcul du nombre de jetons visuels (§5). `tokenizer.json` n'est pas dans `docs/configs/` : aucun comptage de jetons de texte n'a pu être fait.

---

## 1. Architecture

**Classes.** `Alpamayo2Edge(PreTrainedModel)` (`E/models/alpamayo2_edge.py:58`) contient `self.vlm` = `Cosmos3EdgeForConditionalGeneration` construit depuis la config (`:79-80` ; `C/config.json:205`) et `self.expert` = `ExpertModel` (`:95-98`, `E/models/expert.py:46`). Le chargement refuse toute clé manquante ou inattendue (`E/checkpoint.py:124-135`).

| Bloc | Module | Dimensions (`C/config.json`) |
|---|---|---|
| Modèle de langue | `vlm.model.language_model` (`T/modeling_cosmos3_edge.py:779`), `Cosmos3EdgeTextModel` | 28 couches, largeur 2 048, MLP 9 216 (`relu2`), 16 têtes × 128, 8 têtes KV, vocabulaire 135 168, RoPE multimodal `[24, 20, 20]` (`:228-275`) |
| Encodeur vision | `vlm.model.visual` (`T/:777`), `Cosmos3EdgeVisionModel` (SigLIP2 selon la fiche, `C/README.md:57`) | 27 couches, largeur 1 152, patch 16, fusion spatiale 2 (`:278-307`) |
| Projecteur | `vlm.model.projector` (`T/:778`), `Cosmos3EdgePatchMerger` : `LayerNorm` → `linear_fc1` → GELU → `linear_fc2` (`T/:748-758`) | 4 × 1 152 → 11 520 → 2 048 (`:226`) |
| Tête de sortie | `vlm.lm_head` (`T/:1136`), non liée aux embeddings (`tie_word_embeddings: false`, `:276`) | 2 048 → 135 168 |
| Expert d'action | `expert.expert` = même classe de texte que le LM, sans table d'embeddings (`E/models/expert.py:63-64`) | 28 couches, largeur 512, MLP 3 840, 16 têtes × 128, 8 têtes KV (`:61-68`, `:78-125`) |
| Entrée/sortie d'action | `expert.action_in_proj` = `PerWaypointActionInProjV2` (Fourier, 2 couches) ; `expert.action_out_proj` = `nn.Linear` | `:12-21` |

L'expert et le LM doivent avoir la même géométrie de cache KV (couches, têtes KV, dimension de tête), vérifié au constructeur (`E/models/alpamayo2_edge.py:99-103`). L'attention de l'expert est non causale et obligatoire (`E/models/expert.py:61-62`, `C/config.json:60`).

**Nombres de paramètres** (`C/README.md:57-58`) : backbone 2,45 Md (LM 1,96 Md, vision 0,41 Md, projecteur 0,08 Md), expert 0,2 Md. Recalcul à froid depuis la config : LM avec `lm_head` = 1,963 Md, tronc de l'expert = 0,198 Md, cohérent avec la fiche.

**Tokenizers de trajectoire.** Ce ne sont pas des modules entraînables.
- Histoire : `DeltaTrajectoryTokenizer`, 1 000 bins, 20 poses → 19 deltas × 3 axes = 57 jetons (`C/config.json:173-177`, `:189` ; `E/config.py:117-125`).
- Futur : `DiscreteTrajectoryTokenizer`, 3 000 bins sur [−10, 10] par dimension, 64 points × (accélération, courbure) = 128 jetons (`:133-171`, `:188` ; `E/action_space/discrete_action_space.py:24-78`).
- Espace d'action : `UnicycleAccelCurvatureCourseActionSpace`, 64 points à 0,1 s (`:22-47`).

**Identifiants de jetons** (`C/config.json:190-199`) : `history_id0` 131 072, `future_id0` 132 072, `history_start` 135 077, `history_end` 135 079, `future_start` 135 084, `future_end` 135 086, `history_pad` 135 087, `future_pad` 135 088. Image : `image_token_id` 19, `vision_start` 20, `vision_end` 21 (`:216`, `:308-309`). EOS = `<|im_end|>` = 11 (`C/tokenizer_config.json`, `C/generation_config.json`).

**Aucun jeton n'est ajouté au chargement.** `build_alpamayo2_edge_tokenizer` charge le tokenizer du checkpoint tel quel et vérifie seulement la présence de `<i0>`, `<i1000>`, `<i3999>` (`E/config.py:56-69` : « Load saved tokens without adding tokens or resizing trained embeddings »). La liste `SPECIAL_TOKENS_KEYS` (`E/models/utils.py:26-56`) contient encore `cot_start`, `cot_end`, `meta_action_start`, `meta_action_end`, mais le seul qui serve à l'inférence est `traj_future_start` (`E/models/token_utils.py:148`).

---

## 2. Inférence : prompt, chemin, sortie

**Prompt.** `prepare_model_inputs` (`E/helper.py:138-160`) appelle `build_conversation` (`E/chat_template.py:64-91`) puis `_tokenize_messages` (`E/helper.py:87-135`). Contenu du tour utilisateur, dans l'ordre :

1. Pour chaque image triée par caméra puis par horodatage : le nom de la caméra au changement de caméra (`"Front camera: "`, etc., `E/chat_template.py:10-17`, `:43-44`), l'étiquette de temps relatif `<t=-1.0s>` (`:27-32`), puis l'image.
2. `\n`, puis `<|traj_history_start|>` + 57 × `<|traj_history|>` + `<|traj_history_end|>` (`:79-84`).
3. `\n`, puis `"Navigation input is: " + nav_text` (`:86`).
4. `\n`, puis la consigne `"Your task: output the chain-of-thought reasoning of the driving process, then output the future trajectory."` (`:19-22`).

Le message système est `"You are a driving assistant that generates safe and accurate actions."` (`:18`).

**Où `<think></think>` est construit.** Il vient du template du checkpoint, pas du code Python. `_tokenize_messages` appelle `apply_chat_template(add_generation_prompt=True, enable_thinking=False)` (`E/helper.py:91-98`) ; la branche correspondante du template émet `<|im_start|>assistant\n<think></think>` (`C/chat_template.jinja:320-322`). Le code vérifie que le texte se termine bien ainsi et lève une erreur sinon (`E/helper.py:99-100`). Le template active la réflexion par défaut (`:15`) : sans `enable_thinking=False`, le prompt finirait par `<think>\n` (`:316-319`). La fiche le dit aussi (`C/README.md:78`). Rendu à froid sur un message réduit :

```
…Navigation input is: Turn left in 25m\nYour task: output the chain-of-thought reasoning of the driving process, then output the future trajectory.<|im_end|>\n<|im_start|>assistant\n<think></think>
```

**Chemin de bout en bout** (`Alpamayo2Edge.sample_trajectories_from_data`, `E/models/alpamayo2_edge.py:203-377`) :

1. `fuse_traj_tokens` remplace les 57 jetons factices par les jetons d'histoire (`:234-240`).
2. `_generate_with_shared_prefill` encode le prompt une fois, duplique le cache par candidat, puis appelle `vlm.generate` (`:115-161`). Échantillonnage : `top_p` 0,98, température 0,6, 256 nouveaux jetons au plus par défaut (`:206-208`, `:261-263`).
3. Pendant la génération, les jetons discrets d'histoire et de futur sont interdits (`:275-282`), et l'EOS de texte est interdit tant que l'ancre n'est pas sortie (`:283-285`). L'arrêt se fait un pas après `<|traj_future_start|>` (`:291`, `E/models/expert_utils.py:23-48`).
4. Si une ligne n'a pas produit l'ancre, `find_eos_offset` lève `ValueError("No <traj_future_start> token found …")` (`E/models/expert_utils.py:70-87`). Il n'y a pas de repli.
5. Le cache est coupé juste après l'ancre (`:331`) et l'expert débruite 64 jetons d'action conditionnés par ce cache (`_sample_actions`, `:163-200`).
6. `action_to_traj` intègre le modèle unicycle (`:359-363`).

**Flow matching.** Intégration d'Euler, 10 pas par défaut (`E/diffusion/flow_matching.py:32`, `:108-152` ; CLI `E/inference.py:285`). Pas de guidage sans classifieur (`E/models/expert.py:69-70`).

**Ce que renvoie l'API trajectoire** : `pred_xyz` de forme `[lot, sets, échantillons, 64, 3]`, `pred_rot` `[…, 64, 3, 3]`, et un troisième tenseur de zéros qui n'est pas une log-probabilité (`:367` ; `EDGE/README.md:318-319`). Avec `return_extra=True`, un dictionnaire avec `cot`, `meta_action`, `answer`, `raw_outputs` et `generated_ids` (`:370-376`). Chaque candidat a sa propre CoC, tirée indépendamment.

**Format de la CoC décodée.** `extract_text_tokens` (`E/models/token_utils.py:135-181`) prend le texte après le dernier `<|im_start|>assistant\n`, le coupe à `<|traj_future_start|>` ou `<|im_end|>`, retire le préfixe `<think></think>` et les espaces (`:151-155`). La CoC est donc du texte libre, sans balise. Le dépôt n'utilise ni `<|cot_start|>` ni `<|cot_end|>`.

**Méta-actions.** La consigne native ne les demande pas. Le décodeur les sépare quand même si le texte contient `Longitudinal:`, `Lateral:` ou `Lane:` (`:122-132`, `:167-174`). Aucun test ni exemple du dépôt ne montre Edge en émettre ; c'est à observer sur GPU.

**VQA.** Un second chemin (`prepare_vqa_inputs`, `E/helper.py:163-182` ; `generate_vqa_answer`, `E/text_tasks.py:17-57`) : 6 caméras × 4 images, pas de navigation, pas d'histoire, pas de bloc `<think></think>`, décodage glouton, pas de trajectoire.

---

## 3. Cibles d'entraînement (question 1)

**La CoC est placée après le bloc `<think></think>` vide, pas dedans, et sans balise.** Chaîne de construction pour la config native `nav_cot` (`R/configs/preprocess/nav_cot.yaml:2-4`) :

```yaml
components_order: [image, traj_history, nav_instruction, prompt, cot, traj_future]
components_prompt: [cot, traj_future]
label_components: [assistant]
```

- `R/conversation.py:147-153` met les composantes jusqu'à `prompt` dans le tour utilisateur et le reste dans un tour assistant. Les composantes sont jointes par `\n` (`:134-142`).
- Le tour assistant vaut donc `{cot}\n<|traj_future_start|>` + 128 × `<|traj_future|>` + `<|traj_future_end|>` (`:96-108`, `:129-131`). Le test de la recette l'écrit en toutes lettres (`R/tests/test_public_conversation.py:72-75`).
- `R/processor.py:189-196` appelle `apply_chat_template(add_generation_prompt=False, enable_thinking=False)`. Le template préfixe `<think></think>` à toute réponse assistant qui n'en contient pas (`C/chat_template.jinja:186-190`) et ferme par `<|im_end|>\n` (`:247-250`).

Rendu à froid de la séquence d'entraînement :

```
…<|im_end|>\n<|im_start|>assistant\n<think></think>Yield to oncoming traffic before turning left.\n<|traj_future_start|><|traj_future|>…<|traj_future_end|><|im_end|>\n
```

**Le format d'entraînement prolonge exactement le prompt d'inférence.** Dans le rendu à froid, la séquence d'entraînement commence par le prompt d'inférence complet, `<think></think>` compris. La recette le teste elle-même : `test_nav_cot_sft_prefix_matches_actual_inference_conversation` compare les deux tours système et utilisateur à ceux du paquet d'inférence (`R/tests/test_public_conversation.py:51-76`). La consigne est la même chaîne (`R/conversation.py:15-20`, `:109-113` contre `E/chat_template.py:19-22`), ainsi que la ligne de navigation (`R/conversation.py:120` contre `E/chat_template.py:86`).

Deux écarts entre entraînement et inférence, relevés dans le code :

1. **Ancre des poses.** Le dataset de la recette ancre l'histoire et le futur sur l'horodatage de la dernière image réellement capturée, pas sur le `t0` demandé (`R/dataset.py:309-312`), et interpole les poses linéairement (`:51-96`, `:322-325`). Le chargeur d'inférence utilise le `t0` demandé et l'interpolateur du SDK (`E/load_physical_aiavdataset.py:165-167`).
2. **Jetons discrets du futur.** Ils sont une cible à l'entraînement mais ne sont jamais générés à l'inférence : la génération s'arrête à l'ancre et la trajectoire vient de l'expert (§2).

**Rôle et format de `nav_instruction`.** C'est une entrée, jamais une cible (`R/README.md:19` : « Navigation is an input »). Texte : `"Navigation input is: " + nav_text[0]`, chaîne non vide obligatoire (`R/conversation.py:114-120`). Le format est validé par expression régulière à la lecture des annotations, sauf `validate_navigation_format=false` (`R/dataset.py:430-431` ; `src/alpamayo/data/pai_metadata.py:69-100`). Formes acceptées : `Continue straight`, `Turn left in 25m`, `Keep right in 80m`, `Exit roundabout`, etc., distance entière de 0 à 250 m.

**Ce que couvre le masque `label_components: [assistant]`.** `get_assistant_mask` (`R/processor.py:25-45`) marque tout ce qui suit l'en-tête `<|im_start|>assistant\n` jusqu'à `<|im_end|>` inclus (`:40-42`). Sont donc supervisés : les jetons de `<think></think>`, la CoC, le `\n`, `<|traj_future_start|>`, les 128 jetons de trajectoire, `<|traj_future_end|>` et `<|im_end|>`. Le `\n` final n'est pas supervisé. L'autre valeur possible, `[traj_future]`, ne supervise que le bloc trajectoire et l'EOS (`R/label_mask.py:9-26`) ; le processeur la refuse dès que la réponse contient autre chose que la trajectoire (`R/processor.py:132-136`).

**Config par défaut.** `sft_base.yaml` charge le prétraitement `trajectory` (`R/configs/sft_base.yaml:3-4`) : sans navigation ni CoC. Le README prévient que `sft_joint`, `sft_joint_nav` et `sft_joint_cot` « change the prompt format » et ne sont pas des références du prompt natif (`R/README.md:8-12`). Seule `sft_joint_nav_cot` conserve la tâche native.

---

## 4. Pertes (question 2)

**Étapes d'entraînement.** `set_training_stage` (`R/models/sft.py:232-260`) accepte `joint`, `vlm`, `expert`. La recette livre `joint` (`R/configs/model/release.yaml:5`).

| Étape | `vlm` (LM + vision + projecteur + `lm_head`) | `expert` | Référence |
|---|---|---|---|
| `joint` | entraîné | entraîné | `sft.py:245-247` |
| `vlm` | entraîné | gelé, en mode `eval` | `:245-247`, `:267-268` |
| `expert` | gelé, sous `no_grad` | entraîné | `:245-247`, `:684-686` |

En `joint`, rien n'est gelé. Les tokenizers de trajectoire n'ont pas de paramètres. Les taux d'apprentissage diffèrent par préfixe : base 3,5e-6 pour le LM, ×0,1 pour `vlm.model.visual` et `vlm.model.projector`, ×4 pour `expert` (`R/configs/sft_base.yaml:93-97` ; `R/trainer.py:319-328`).

**Formule en `joint`** (`_forward_joint`, `sft.py:605-667`, et `_vlm_loss(joint=True)`, `:421-478`) :

```
L = w_others × (1/N) × Σ_i  p_i × CE_i   +   w_future_traj × MSE(v_pred, x − bruit)
```

- La somme porte sur les N positions supervisées par le masque (§3). `p_i` = 1,0 si la cible est un jeton de trajectoire future, un de ses deux délimiteurs, ou l'EOS qui suit immédiatement `<|traj_future_end|>` ; `p_i` = 0,2 sinon, donc pour la CoC et le bloc `<think></think>` (`sft.py:448-457` ; `R/models/utils.py:70-88` ; `R/configs/model/release.yaml:13-15`).
- Le dénominateur est N, le nombre total de jetons supervisés : les poids changent le numérateur, pas le dénominateur (`sft.py:446-447`, `:457`).
- `w_others` = 1,0 multiplie toute l'entropie croisée, trajectoire discrète comprise (`:460`). `w_future_traj` = 1,0 multiplie la perte de l'expert, pas l'entropie croisée de trajectoire (`:658`). Valeurs : `release.yaml:9-11`.
- Le second terme est le flow matching : `x` = action (accélération, courbure) calculée en float32 depuis la trajectoire vraie, `t` tiré d'une Beta(1,5 ; 1) ramenée à [0 ; 0,999], entrée bruitée `t·x + (1−t)·bruit`, cible `x − bruit`, erreur quadratique moyenne en float32 (`:722-735`, `:786-790`, `:792-813`).
- Les logits ne sont calculés que sur les positions supervisées (`:430-432`), ce qui évite de matérialiser les logits du prompt et des images.
- `future_traj_ce` et `others_ce` journalisés sont des moyennes par catégorie calculées sous `no_grad` ; elles ne font pas partie de la perte (`:461-465`).

**« Copie détachée du cache du VLM ».** Après la passe du VLM avec `use_cache=True` (`:636`), la recette construit un second cache pour l'expert en copiant les clés et valeurs de chaque couche, coupées après l'ancre, avec `.detach()` (`:641-648`). Conséquences :

- Le gradient de la perte de l'expert ne remonte pas dans le VLM. Le VLM ne reçoit de gradient que de l'entropie croisée (`R/README.md:17-18`).
- L'expert reçoit son gradient par ses propres poids (28 couches, projections d'entrée et de sortie).
- Le cache du VLM utilisé par l'entropie croisée n'est ni détaché ni coupé (`:641-642`).
- L'expert voit le prompt et la CoC de référence jusqu'à `<|traj_future_start|>` inclus, pas les jetons discrets du futur (`:649-651`).

Le checkpoint publié porte les mêmes drapeaux : `cotrain_expert_vlm: true`, `cotrain_detach_expert_kv: true` (`C/config.json:5-6`).

**Étape `vlm` seule.** Formule différente : somme de deux moyennes par catégorie, `w_future_traj × CE_traj + w_others × CE_autres`, sans les poids par jeton 1,0 / 0,2 (`sft.py:466-471`). La recette l'appelle « legacy » (`:350`, `:389`).

**Lots mixtes.** `_forward_joint_mixed` accepte des lignes sans trajectoire (VQA) ; un lot sans aucune trajectoire fait passer un segment d'expert à poids nul pour garder les rangs distribués synchronisés (`:480-603`). La recette ne fournit pas de dataset VQA (`R/README.md:25-26`).

---

## 5. Mémoire (question 3)

**Ce que dit la recette.**

| Réglage | Valeur | Référence |
|---|---|---|
| GPU | 8, lot de 1 par GPU, accumulation 8 (lot global 64) | `R/README.md:411`, `:454` ; `R/configs/sft_base.yaml:88-90` |
| Poids maîtres et état de l'optimiseur | FP32 | `R/README.md:415` ; `R/configs/model/release.yaml:3` (`dtype: float32`) |
| Calcul | BF16 (`trainer.bf16: true`) | `R/configs/sft_base.yaml:109` |
| Parallélisme | DeepSpeed ZeRO-3, accumulation des gradients en FP32 | `R/configs/sft_base.yaml:114` ; `R/configs/deepspeed/zero3.json:9`, `:25-28` |
| Gradient checkpointing | désactivé, et interdit en `joint` | `R/configs/sft_base.yaml:110-111` ; `sft.py:344-355`, `:609-610` |
| Longueur maximale | 8 192 jetons, padding à la plus longue séquence du lot | `R/configs/sft_base.yaml:60-63` ; `R/README.md:418` |
| Dépassement | erreur, pas de troncature | `R/processor.py:301-305` |
| État d'entraînement sauvegardé | 35 à 40 Go par sauvegarde | `R/README.md:440-441` |
| Attention | SDPA | `R/configs/model/release.yaml:4` |

Le README ne nomme aucun modèle de GPU et ne donne aucun chiffre de mémoire GPU pour l'entraînement. Une variante `zero3_optimizer_offload.json` déporte l'optimiseur sur CPU.

**Pourquoi pas de gradient checkpointing.** La passe conjointe a besoin du cache KV du VLM, que le VLM ne peut pas exporter sous checkpointing (`sft.py:351-355`). L'étape `vlm` seule l'autorise (`:356-360`) et tourne avec `use_cache=False` (`:411`).

**Sur un seul GPU, sans DeepSpeed.** L'optimiseur exige des paramètres entraînables en FP32 (`R/trainer.py:249-261`). La seule config mono-GPU livrée est `sft_joint_single_node_debug.yaml` : 1 caméra, 1 image, 2 pas, `deepspeed: null`.

**Ce que la fiche mesure à l'inférence** (`C/README.md:187-190`) : un H100 80 Go, un échantillon de 22 images, **3 236 jetons de prompt**, BF16, 6 candidats : pic de 12,06 Gio de mémoire processus, 8,40 Gio alloués par PyTorch. Seul le H100 est déclaré testé (`C/README.md:109-110`).

**Longueur d'une séquence native, calcul à froid.** Le nombre de jetons d'une image est `H × W / 1 024` après redimensionnement (`T/processing_cosmos3_edge.py:69-73`, patch 16, fusion 2). `smart_resize` avec un facteur 32 et les budgets par caméra (`E/sample.py:13-17`, `E/helper.py:111-116`) donne, pour des images sources de 1 080 × 1 920 :

| Caméras | Images | Budget de pixels | Taille après redimensionnement | Jetons par image | Total |
|---|---|---|---|---|---|
| 1 et 6 (avant, téléobjectif) | 2 × 5 | 163 840 à 196 608 | 320 × 576 | 180 | 1 800 |
| 0, 2, 3, 5 | 4 × 3 | 81 920 à 98 304 | 224 × 416 | 91 | 1 092 |
| **Total visuel** | **22** | | | | **2 892** |

- Les 2 892 jetons visuels sont calculés. Les 3 236 jetons de prompt viennent de la fiche ; la différence, 344 jetons de texte, d'histoire et de délimiteurs, n'est pas recalculée faute de `tokenizer.json`.
- Séquence d'entraînement = prompt + CoC + `\n` + 130 jetons de bloc trajectoire + `<|im_end|>`, soit environ **3 370 jetons plus la longueur de la CoC**. Les CoC de démonstration font une phrase.
- La marge sous 8 192 est d'un facteur 2,4 environ. Elle dépend de la résolution source : le calcul suppose 1 080 × 1 920, la résolution de l'exemple public (`EDGE/README.md:251`).

**Ordre de grandeur des états, arithmétique seule.** 2,65 Md de paramètres en FP32 = 9,9 Gio. Poids, gradients et deux moments d'AdamW = 4 × 9,9 = **39,5 Gio avant toute activation**, si tout est entraîné sur un seul GPU sans déport. C'est un calcul, pas une mesure : il ne dit rien des activations d'une séquence de 3 400 jetons sans checkpointing.

---

## 6. Données (question 4)

**Schéma du JSON d'annotations** (`R/README.md:121-126` ; lecture dans `PhysicalAIAnnotatedDataset`, `R/dataset.py:370-493`) : une liste d'objets.

| Champ | Type | Règle |
|---|---|---|
| `clip_id` | chaîne | obligatoire, UUID d'un clip PAI présent dans l'index local (`:408-410`, `:464-465`) |
| `t0_relative` | entier, microsecondes depuis le début du clip | obligatoire, entre 2 000 000 et 13 600 000 (`:411-415`, `:451-463`) |
| `nav_text` | chaîne ou liste d'une chaîne | optionnel sauf `require_navigation` ; format validé (`:417-434`) |
| `cot` | chaîne non vide | optionnel sauf `require_cot` (`:435-441`) |

Les champs supplémentaires sont ignorés ; l'exemple livré en contient (`nav_maneuver`, `distance_m`). Deux annotations du même clip restent deux exemples. Le README exige que navigation, raisonnement et trajectoire décrivent le même événement (`R/README.md:128-129`).

**CoC de démonstration.** `R/assets/nav_demo_samples.json` : 19 événements de 18 clips. Navigation et horodatages viennent de la démo d'Alpamayo 1.5. Phrase exacte (`R/README.md:144-146`) : « Reasoning was generated by Alpamayo2 Edge, reviewed, and edited to remove flagged unsupported wording. These are synthetic demonstration annotations, not human ground truth. » Aucun script produisant ces CoC n'a été trouvé dans `R/` ni dans `scripts/` ; l'évaluation génère des CoC mais ne les conserve pas (§8). Le README qualifie le découpage qui en résulte de « pipeline smoke test only » (`:170`).

**Phrase exacte sur les CoC publiques de PAI** (`R/README.md:181-182`) : « PAI's public CoC labels describe selected events and differ from route-conditioned reasoning and nominal labels used by the release. Review their suitability for your task before use. »

Autres faits sur ces données :
- Toutes les tranches CoC éligibles avec six caméras pèsent environ 7,7 Tio (`:183-184`).
- Le découpage de validation CoC officiel fait 226 clips et 266 événements dans la fenêtre 2,0 à 13,6 s (`:217-219`).
- `scripts/prepare_pai_annotations.py` joint les CoC publiques à une navigation fournie par l'utilisateur, à l'horodatage exact ; il ne génère pas d'instruction de navigation.
- Le jeu d'entraînement de la release, selon la fiche : environ 115 000 heures, environ 3,7 millions de traces CoC dont des traces synthétiques, des instructions de navigation et des données de question-réponse (`C/README.md:144-148`).

**Ce qu'il faudrait pour entraîner avec nos textes (cibles d'incertitude `afh`).** Constat sur le code, sans proposition de conception :

1. **Texte.** Le champ `cot` accepte n'importe quelle chaîne non vide. Rien ne contraint son vocabulaire ni sa longueur à l'entraînement, hors le plafond de 8 192 jetons.
2. **Navigation.** La tâche native exige un `nav_text` par événement. PAI public n'en fournit pas (`E/load_physical_aiavdataset.py:147-148`) ; nos manifestes et fixtures non plus.
3. **Images dégradées.** Le dataset décode les images directement depuis PAI (`R/dataset.py:292-298`) et les passe au prétraitement (`:365-366`) sans point d'entrée pour une dégradation. Deux points d'extension existent dans le code : `dataset_interface_factory` (`:151`, `:200-204`) et la surcharge de `_load_sample`.
4. **Trajectoire cible amortie.** Le futur est recalculé depuis l'egomotion (`:322-346`). Une cible différente de la trajectoire enregistrée demande de remplacer `ego_future_xyz` et `ego_future_rot`, qui alimentent à la fois les jetons discrets et l'expert (`sft.py:725-735`).
5. **Fenêtre temporelle.** `t0` entre 2,0 s et 13,6 s. Notre `t0` de 5,1 s sur `0ea6fd88` y entre.
6. **Données locales.** Le clip doit être dans un téléchargement PAI local avec `clip_index.parquet` (`R/README.md:104-106`).
7. **Mise en forme des images.** 22 images à plat, étiquetées par caméra, pas un tenseur caméras × temps (§7).

---

## 7. Entrées (question 5)

**Caméras.** Six caméras, identifiants `0, 1, 2, 3, 5, 6` ; la caméra 4 (téléobjectif arrière) est absente (`E/load_physical_aiavdataset.py:21-28` ; `R/dataset.py:22-31`). Correspondance identique à `AFH/afh/cameras.py:20-25`. Les noms affichés dans le prompt (`E/chat_template.py:10-17`) correspondent caméra par caméra à ceux d'`AFH/afh/cameras.py:30-35`.

**Ordre.** Tri par identifiant de caméra, puis par horodatage de capture (`E/sample.py:123-129` ; `R/processor.py:158-160`).

**Images.** 22 au total (`E/sample.py:13` ; `E/load_physical_aiavdataset.py:29-30`) :

| Caméras | Images | Décalages demandés (s) | Budget de pixels |
|---|---|---|---|
| 1 et 6 | 5 chacune | −2 ; −1 ; −0,5 ; −0,1 ; 0 | 163 840 à 196 608 |
| 0, 2, 3, 5 | 3 chacune | −1 ; −0,1 ; 0 | 81 920 à 98 304 |

La cadence n'est pas uniforme et la fenêtre fait 2 s à l'avant, 1 s ailleurs. Les horodatages réels de capture sont conservés et rendus dans le prompt, relatifs à la dernière image de la dernière caméra (`E/chat_template.py:27-32`). L'histoire d'ego est de 20 poses, de −1,9 s à 0 par pas de 0,1 s (`E/load_physical_aiavdataset.py:165`). `t0` doit valoir au moins 2,0 s (`:157-158`).

**Comparaison avec `afh` et nos fixtures.**

| Point | `afh` à `7fff9e6` | Edge |
|---|---|---|
| Forme des images | tableau `(n_cam, n_t, 3, H, W)` (`AFH/afh/degradation.py:28`) | 22 images à plat `[22, 1, 3, H, W]` (`E/sample.py:53-70`) |
| `camera_indices` | un identifiant par caméra ; les doublons sont refusés (`AFH/afh/cameras.py:51-52`) | un identifiant par image, donc avec doublons (`E/sample.py:71-80`) |
| Images par caméra | même `n_t` pour toutes | 5 ou 3 selon la caméra |
| Dégradations temporelles | `freeze` et `desync` raisonnent en nombre d'images sur `n_t` (`AFH/afh/degradation.py:301-322`) | pas de temps irréguliers |
| Occlusion contrefactuelle | `occlude_frames` attend des horodatages `(n_cam, n_t)` et la calibration (`AFH/runners/run_counterfactual_a2.py:159-184`) | horodatages par image fournis ; calibration non demandée par le chargeur d'inférence (`E/load_physical_aiavdataset.py:147-148`), optionnelle dans la recette (`R/dataset.py:357-359`) |
| Navigation | absente des fixtures et des manifestes | obligatoire |
| Runner | `run_counterfactual_a2.py` importe `alpamayo2_super` | aucun runner Edge |

Les fixtures `cf_0ea6fd88*.json` portent le clip `0ea6fd88` ; ce clip n'apparaît dans aucun exemple d'Edge ni de la recette.

**Inférence sans instruction de navigation.** Elle échoue à trois niveaux : la CLI (`E/inference.py:159-160`), le chargeur (`E/load_physical_aiavdataset.py:153-154`), la validation de l'échantillon (`E/sample.py:116-120`). `build_conversation` lit `data["nav_text"][0]` sans condition (`E/chat_template.py:86`). Le paquet d'inférence n'a pas de prompt de conduite sans navigation. Le seul chemin sans navigation est la VQA, qui ne produit pas de trajectoire. La recette sait construire des prompts sans navigation (`cot`, `trajectory`), en prévenant qu'ils sortent du format natif (`R/README.md:8-12`).

---

## 8. Évaluation (question 6)

**`evaluate_hf.py`**, deux modes (`R/evaluate_hf.py:397-433`).

- **`loss`** : perte de validation avec la CoC de référence en entrée forcée, aux poids enregistrés dans le checkpoint (`R/README.md:683-686`).
- **`trajectory`** : minADE. Un seul processus, sans DeepSpeed (`:163-164`). Pour chaque événement, 6 candidats tirés par groupes de 2 (`R/configs/sft_base.yaml:72-74`).
  - Définition (`compute_min_ade`, `:39-58`) : distance en XY, moyenne sur les 64 points (6,4 s), minimum sur les candidats, puis moyenne sur les événements (`:363-364`).
  - Avec `generate_cot: true` (config `sft_eval_trajectory_nav_cot.yaml`), le modèle génère sa propre CoC puis la trajectoire, sans CoC de référence (`:180-187`).
  - **Le texte généré n'est ni conservé ni évalué** : l'appel ne garde que `pred_xyz` (`:251-254`), et `trajectory_samples.jsonl` ne contient que l'identité de l'événement, un statut et le minADE (`:232-235`, `:268-278`).
  - Si l'ancre manque, tout le lot est exclu et `eval_min_ade` n'est publié qu'à 100 % de couverture (`:200-204`, `:261-270`).
  - Sans `generate_cot`, la trajectoire est tirée depuis un prompt qui s'arrête à l'ancre, sans CoC (`R/models/public_inference.py:14-35`).

**`compare_evaluations.py`** : comparaison appariée entre un checkpoint de base et des checkpoints fine-tunés, sur exactement les mêmes événements et les mêmes graines (docstring, `:4-37`). Trois modes exclusifs : moyennes seules, bootstrap par session de conduite, bootstrap par clip sous hypothèse d'indépendance. Il rapporte « SFT moins base » en minADE ; négatif = mieux (`:594`). Les métadonnées publiques de PAI n'ont pas d'identifiant de session (`R/README.md:643-646`).

**Comparabilité avec `afh`.**

- Le minADE est le seul point commun, et il n'est pas défini de la même façon partout : la recette moyenne sur 6,4 s avec 6 candidats ; la fiche publie un minADE à 5 s sur 1 392 échantillons, 0,52 m (`C/README.md:157`) ; `afh` tronque à l'horizon le plus court et prend le meilleur des rollouts fournis (`AFH/afh/eval_uncertainty.py:136-154`).
- Rien dans la recette ne lit la CoC générée. Les axes d'`afh` (sensibilité contrefactuelle, citation, cohérence texte-action, calibration de l'incertitude) n'ont pas d'équivalent.
- Le protocole d'appariement par événement et par graine de `compare_evaluations.py` est indépendant de la métrique, mais l'outil n'accepte que les fichiers produits par `evaluate_hf` (`:14-19`).

---

## 9. Recettes voisines (question 7, README et configs seulement)

**`recipes/alpamayo_distillation/`** : quatre sous-recettes (`README.md:16-19`).

- **Professeurs acceptés** : Alpamayo 1 et 1.5 seulement. « The recipe collection supports Alpamayo 1 and Alpamayo 1.5 teacher models. » (`README.md:9`). Configs : `teacher_name_or_path: nvidia/Alpamayo-1.5-10B` (`alpamayo_1_5_distill/configs/baseline.yaml:75`). **2 Super n'est accepté nulle part** : aucune occurrence de `alpamayo2`, `Super` ou `34B` dans les README, configs et `pyproject.toml`.
- **Élève** : fixé à `nvidia/Cosmos-Reason2-2B` plus un expert compact. « Only one student architecture is supported. » (`alpamayo_1_distill/README.md:5-6`, `:26`). **Edge n'est accepté nulle part** : aucune occurrence de `edge`, `Cosmos3` ou `Nemotron`.
- **Signaux distillés** : entropie croisée, KL professeur-élève sur les logits, projection d'états cachés, flow matching sur la trajectoire vraie, perte auxiliaire de trajectoire discrète, phase optionnelle sur la politique de l'élève (`alpamayo_1_5_distill/README.md:222-230`). Les réponses du professeur sont générées en ligne.
- **Matériel** : au moins 128 H100 80 Go recommandés pour l'étape 1 (`alpamayo_1_distill/README.md:57` ; `alpamayo_1_5_distill/README.md:19`), 8 H100 pour les étapes sur l'expert, 100 To de stockage (`README.md:48`).
- **Versions** : `transformers==4.57.1`, `torch==2.8.0`.

**`recipes/alpamayo2edge_edgellm/`** : déploiement.

- **Cible matérielle** : DRIVE AGX Thor uniquement, DRIVE OS 7.2.7.0, CUDA 13.4, TensorRT 11.3.1.9 ; « Other versions are untested. » (`README.md:19`, `:23`). Hôte x86-64 avec un GPU d'au moins 16 Go pour la quantification (`:25-26`).
- **Format d'export** : ONNX sur l'hôte, puis moteurs TensorRT construits sur la carte, via TensorRT Edge-LLM `v0.11.1` (`:28`, `:68-70`, `:96-98`). Quatre graphes : vision, LLM, expert d'action, modèle brouillon.
- **Quantification** : FP8 pour les poids, les activations et le cache KV, calibrée sur les images d'un clip ; quelques couches restent en FP16 (`:14-15`, `:57-58`). Variante FP16 possible (`:79`).
- **Expert** : exporté en un seul graphe à 5 pas d'Euler, contre 10 par défaut en PyTorch (`:68-70`).
- **Vocabulaire de sortie réduit** : la tête du LM est réduite aux **2 211 identifiants de jetons que la CoC utilise** (`host/coc_vocab.json`), complétés à 8 192 lignes (`:63-64`, `:82-83`).
- **Décodage spéculatif** : avec un modèle brouillon « DFlash2 » qui « is not part of the model release » (`:77-78`) ; sans lui, décodage glouton.
- **Latence indiquée** : de l'ordre de 200 ms par requête en FP8 avec 6 candidats, « not a benchmark measurement » (`:5`, `:114-116`).
- **Checkpoint fine-tuné, LoRA, poids fusionnés** : non mentionnés. Le README ne traite que le checkpoint publié.
- **Révision épinglée** : `nvidia/Alpamayo2-Edge` à `aaaa32c` (`:27`, `:39`), différente de la `f8951d0` du README d'inférence (`EDGE/README.md:109`). La recette SFT n'en fixe aucune (`R/README.md:73`, `$MODEL_REVISION`).

---

## 10. Raisonnement adaptatif (question 8)

**Dans le code, ce n'est rien de configurable.** Le mot « adaptive » n'apparaît qu'en prose : README d'inférence (`EDGE/README.md:14`), fiche (`C/README.md:25`), README d'alpamayo-recipes (`README.md:118`, `:125` : « it reasons adaptively, conserving compute resources during routine driving »). Recherche dans `E/` et `R/` : aucune occurrence dans le code, aucun seuil, aucun mode, aucun second prompt de conduite.

Ce que le code fait, et qui borne ce que « adaptatif » peut vouloir dire :

- Le prompt de conduite est toujours le même et finit toujours par `<think></think>` fermé. `enable_thinking=False` est une constante du code (`E/helper.py:97`), pas une option. Le champ `"enable_thinking": false` de la config (`C/config.json:9`) n'est lu nulle part dans `E/`.
- La longueur de la CoC est décidée par le modèle : il écrit du texte puis émet `<|traj_future_start|>`. Rien n'impose une longueur minimale ; l'ancre peut être le premier jeton généré, auquel cas la CoC décodée est vide.
- Le seul plafond est `max_generation_length`, 256 par défaut, avec erreur si l'ancre n'est pas sortie (§2).

Si le raisonnement s'adapte, c'est donc par un comportement appris : une CoC plus ou moins longue selon la scène. Le code ne permet ni de le forcer ni de le désactiver. La distribution des longueurs de CoC, et l'existence de CoC vides, sont à observer sur GPU.

---

## 11. Modules candidats à LoRA

La recette ne contient ni LoRA ni PEFT (aucune occurrence dans le code, les configs ou `uv.lock`). Noms des couches linéaires, lus dans `T/modeling_cosmos3_edge.py` :

| Bloc | Chemin | Projections |
|---|---|---|
| LM | `vlm.model.language_model.layers.{0..27}.self_attn` | `q_proj` (2 048→2 048), `k_proj`, `v_proj` (2 048→1 024), `o_proj` (`:212-221`) |
| LM | `vlm.model.language_model.layers.{i}.mlp` | `fc1`, `fc2` (`:274-275`) |
| Vision | `vlm.model.visual.encoder.layers.{0..26}.self_attn` | `q_proj`, `k_proj`, `v_proj`, `out_proj` (`:457-460`) |
| Projecteur | `vlm.model.projector` | `linear_fc1`, `linear_fc2` (`:756-758`) |
| Expert | `expert.expert.layers.{0..27}.self_attn` et `.mlp` | mêmes noms que le LM, largeur 512 |

Différences avec 2 Super : le MLP du LM s'appelle `fc1`/`fc2`, pas `gate_proj`/`up_proj`/`down_proj` ; les noms `q_proj`, `k_proj`, `v_proj` existent à la fois dans le LM, l'expert et l'encodeur vision.

Trois points du code de la recette qui interagissent avec un entraînement partiel :
- `set_training_stage` remet `requires_grad` à vrai sur tout le VLM et tout l'expert, et il est rappelé à la fin de `from_pretrained` (`sft.py:115`, `:245-247`).
- Les multiplicateurs de taux d'apprentissage s'appliquent par préfixe de nom de paramètre (`R/trainer.py:319-328`).
- `save_pretrained` écrit un checkpoint complet chargeable par le paquet d'inférence (`sft.py:271-280`).

---

## 12. Licences et dataset

- **Code d'inférence** : Apache-2.0 (`EDGE/LICENSE`, `EDGE/README.md:402`).
- **Code de la recette** : Apache-2.0 (`R/LICENSE`).
- **Poids** : OpenMDW-1.1 (`EDGE/README.md:404` ; `C/README.md:31` ; texte dans `C/LICENSE`). Le texte accorde le droit d'utiliser les « Model Materials » « without restriction », demande de conserver la licence et les mentions d'origine en cas de redistribution, et dit ne poser aucune restriction sur les sorties générées (`C/LICENSE:12-30`). Lecture du texte, pas un avis juridique.
- **Modèle de base** : `nvidia/Cosmos3-Edge` (`C/README.md:9-10`).
- **Datasets déclarés par la fiche** : `nvidia/PhysicalAI-Autonomous-Vehicles` et `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` (`C/README.md:2-4`). PAI demande d'accepter ses propres conditions d'accès (`C/README.md:194` ; `EDGE/README.md:99-100`). Révision de PAI épinglée par la recette : `33f9bf4` (`R/README.md:88`).
- **Jeux tiers de la recette** : ONCE est sous CC BY-NC-SA 4.0, usage non commercial ; la licence de la recette ne couvre pas les jeux tiers (`R/README.md`, section « Optional ONCE and Tartan Drive data »).
- **Évaluations publiées** (`C/README.md:155-157`) : LingoQA 75,2 ; boucle fermée AlpaSim sur 875 scénarios NuRec ; boucle ouverte minADE_6 à 5 s de 0,52 m sur 1 392 échantillons PAI.

---

## 13. Ce que ça contredit dans ROADMAP, DECISIONS et `finetune_design.md` (signalé, non tranché)

**Dans la demande du 2026-11-22 elle-même.**

1. La ligne de risque prévue, « pas de code d'entraînement officiel pour Edge », est fausse : `recipes/alpamayo2edge_sft` existe depuis le 2026-11-10.
2. Le bloc 3 de la séance GPU 02 prévoyait d'adapter `probe_train_memory_a2.py`. Le chemin officiel passe par `TrainableAlpamayo2Edge` et `Alpamayo2EdgeCollator`, avec un plafond de 8 192 jetons et un padding à la plus longue séquence, pas 6 144 fixes.
3. « Plein fine-tuning puis LoRA » sur A100 40 Go : l'arithmétique du §5 donne 39,5 Gio d'états FP32 avant toute activation dans le schéma de précision de la recette. À mesurer, mais le calcul ne laisse pas de marge sur 40 Go.
4. « Blackout et contrefactuel occlusion sur les fixtures existantes » : les fixtures n'ont pas d'instruction de navigation, `afh` attend un tenseur caméras × temps sans doublon d'identifiant, et l'occlusion demande une calibration que le chargeur d'Edge ne fournit pas (§7).
5. Le A100 n'est pas un GPU validé par NVIDIA pour Edge ; seul le H100 80 Go est déclaré testé (`C/README.md:109-110`).

**ROADMAP.**

6. §1, but et tableau de référence : « Build, from NVIDIA Alpamayo 2 Super » et la colonne « Baseline (Alpamayo 2 Super) ». Le défaut de référence (issue alpamayo2 #9) est observé sur 2 Super ; rien ne dit encore si Edge le partage.
7. W1 décrit la recette 2 Super (trajectoire seule, 32 H100). La recette Edge supervise la CoC dans sa config native et vise 8 GPU.
8. W0 : « smoke training runs on 4 GPUs » ; la recette Edge est écrite pour 8 GPU et sa seule config mono-GPU est une config de debug à 1 caméra et 1 image.
9. W2 : « Camera remapping to the 7 Alpamayo slots ». Edge utilise 6 caméras, 22 images à cadence irrégulière, avec un budget de pixels par caméra.
10. W3 : la phrase sur le minADE de 0,8288 m concerne la recette 2 Super. Pour Edge, trois définitions de minADE coexistent (§8).
11. W6 : « 2 Super as teacher, 1.5 Nano as natural student ». Les recettes de distillation de NVIDIA n'acceptent ni 2 Super comme professeur ni Edge comme élève, et recommandent 128 H100 (§9). Edge est déjà le modèle déployé sur Thor.
12. W6, déploiement Thor : l'export réduit la tête de sortie aux 2 211 jetons que la CoC utilise. Un texte d'incertitude avec d'autres mots sortirait de ce vocabulaire tant qu'il n'est pas reconstruit.
13. §6, budget : chiffré pour LoRA sur 2 Super à 4 H100. La recette Edge fait du plein fine-tuning sur 8 GPU non nommés.
14. §8, risque « Loss of CoC when training on trajectory only (the official recipe's targets) » : vrai pour la config par défaut d'Edge (`trajectory`), faux pour `sft_joint_nav_cot`. Le poids du texte y est 0,2 contre 1,0 pour la trajectoire.
15. §8, risque de licence : les poids d'Edge sont sous OpenMDW-1.1, dont le texte est maintenant dans le dépôt ; les conditions du dataset restent à part.

**DECISIONS.**

16. D-001 (« Start from Alpamayo 2 Super ») et D-006 (pousser 2 Super dans les challenges) nomment 2 Super comme base.
17. D-003 et D-014 portent sur 1.5 et sur la recette 2 Super ; le critère de D-014 (bloc 2 de la séance 01 sur H200) n'a jamais été mesuré.
18. D-004 : « Fits on 4×H100 » et PEFT sur `TrainableAlpamayo2Super`. La recette Edge n'a pas de PEFT ; les noms de modules diffèrent (§11).
19. D-007 (texte et action ne se contredisent jamais) : compatible avec la recette, qui entraîne CoC et trajectoire dans la même séquence. À noter : l'expert est conditionné par la CoC de référence, et la CoC ne pèse que 0,2 dans la perte.
20. D-008 : le format balisé `<|cot_start|>…<|cot_end|>` ne s'applique pas à Edge. Sa CoC est du texte libre après `<think></think>`, à l'entraînement comme à l'inférence (§3).
21. D-010 : la correspondance identifiant → caméra tient. L'hypothèse implicite d'`afh`, un tenseur rectangulaire avec un identifiant par caméra, ne tient plus (§7).

**`finetune_design.md`.**

22. Le titre et les §1 à §3 décrivent le portage de la recette 1.5 vers 2 Super. Processeur, collate, masque, pertes et évaluation existent déjà pour Edge.
23. Tableau des inconnues, état pour Edge :

| # | Inconnue | Pour Edge |
|---|---|---|
| 1 | Jetons de la CoC à l'entraînement | Levée par le code : texte libre après `<think></think>`, sans balise. Reste à observer ce que le modèle génère réellement |
| 2, 3 | Configs des checkpoints | Config d'Edge déposée ; `tokenizer.json` manque |
| 4 | Vocabulaire des méta-actions | Ouverte : la consigne ne les demande pas, le décodeur sait les séparer |
| 5 | Forward avec gradients | Levée par la recette, sous `transformers` 5.17.0 et non 4.57 |
| 6 | Mémoire d'un pas d'entraînement | Ouverte : aucun chiffre dans la recette ; arithmétique au §5 |
| 7 | Noms des modules PEFT | Levée, noms différents de 2 Super (§11) |
| 8 | Longueur de séquence | Estimée : 2 892 jetons visuels calculés, 3 236 de prompt selon la fiche, plafond 8 192 |
| 9 | flash-attn | Tombe pour le chemin par défaut : SDPA partout, flash-attn optionnel |
| 10 | `traj_to_action` sur trajectoires amorties | Ouverte ; la classe d'espace d'action a changé (`…CourseActionSpace`) |
| 11 | Reproductibilité des rollouts | Ouverte ; le README prévient qu'un autre backend d'attention peut changer le texte tiré (`EDGE/README.md:92-93`) |
| 12 | Licence des poids dérivés | Texte OpenMDW-1.1 disponible ; conditions du dataset à part |

**Nouvelles inconnues propres à Edge.**

24. L'instruction de navigation est une entrée obligatoire de la tâche native, absente de PAI public, de nos fixtures et des manifestes `afh`.
25. Le « raisonnement adaptatif » n'a aucun réglage : la CoC peut-elle être vide, et à quelle fréquence ? (§10)
26. Environnement : `transformers` 5.17.0 contre 4.57.1 pour 2 Super et 1.5 ; `physical-ai-av` 0.2.0 pour l'inférence d'Edge et 0.2.2 dans le verrou de la recette (`R/uv.lock:1127-1128`). Un venv de plus.
27. Trois révisions HF différentes selon la source : `f8951d0` (README d'inférence, nos configs), `aaaa32c` (recette de déploiement), aucune (recette SFT).
