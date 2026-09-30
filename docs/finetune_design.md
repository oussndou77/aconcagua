# W1 — Note de conception : porter la recette SFT Alpamayo 1.5 vers Alpamayo 2 Super

*Livrable « design note » de W1 (`ROADMAP.md` §4). Rédigé à froid, sans GPU, le 2026-09-30.*

**Sources lues, par commit** (tout ce qui suit est tiré du code, pas des blogs) :

| Dépôt | Commit | Ce qui a été lu |
|---|---|---|
| `NVlabs/alpamayo-recipes` | `ae5bc10` (2026-09-23) | `recipes/alpamayo1_5_sft/` en entier (README, SKILL, configs, `models/`, `trainer.py`, `train_hf.py`, `performance/`, tests), `src/alpamayo/` (données, processeur, templates de conversation, masque de labels), `scripts/convert_checkpoint.py`, `scripts/convert_release_config_to_training.py`, `recipes/alpamayo1_x_rl/hydra_configs/alpamayo1_5_rvla_rl_pai.yaml` |
| `NVlabs/alpamayo` (paquet `alpamayo_r1`) | `11a0e01` (2026-09-09) | `diffusion/flow_matching.py`, `models/base_model.py` (jetons spéciaux), `helper.py`, `config.py` — c'est **ce paquet-là** que la recette 1.5 importe, pas `alpamayo1_5` |
| `NVlabs/alpamayo1.5` (paquet `alpamayo1_5`) | `36aeb4c` (2026-09-09) | tout `src/alpamayo1_5/` (modèle, config, helper, chargeur, tokenizers, action space, diffusion), `test_inference.py`, les 4 notebooks |
| `NVlabs/alpamayo2` (paquet `alpamayo2_super`) | `5e7975f` (2026-09-15) | tout `src/alpamayo2_super/` (modèle, config, helper, chargeur, profils d'entrée, template, tokenizers, expert, diffusion, `text_tasks.py`, `inference_smoke.py`), `examples/two_gpu_nav_cfg_demo.py`, notebooks |

**Configs des checkpoints** : `docs/configs/alpamayo2-super/` (`config.json`, `tokenizer_config.json`, `preprocessor_config.json`) et `docs/configs/alpamayo-1.5-10b/config.json`, ajoutés le 2026-09-30 (commit `804fd42`). Ils lèvent les inconnues 2, 3 et 7 de la première version de cette note ; tout ce qui est marqué **[config]** ci-dessous vient de ces fichiers. Les fiches modèle HF (licence des poids dérivés) restent non lues d'ici. Les chiffres de paramètres (10B = 8,2B + 2,3B ; 34B = 32B + 2B) viennent des README des dépôts.

---

## 0. Résumé en dix lignes

1. La recette 1.5 n'entraîne pas `alpamayo1_5` : elle convertit le checkpoint 1.5 au format Alpamayo 1 (`convert_checkpoint.py to-a1`) et entraîne les classes de `alpamayo_r1` (dépôt `NVlabs/alpamayo`). Les deux modèles sont architecturalement identiques ; seuls les chemins `_target_` de la config diffèrent.
2. Étape 1 = entropie croisée sur le VLM complet (vision à 0,1× le LR), DeepSpeed ZeRO-2, gradient checkpointing, bf16. Deux termes : CE sur les jetons de trajectoire discrets, CE sur tout le reste des labels, sommés.
3. Étape 2 = VLM gelé, expert de diffusion entraîné par flow matching (MSE entre le champ de vitesse prédit et `x − bruit`, `t ~ Beta(1.5, 1)`), conditionné par le cache KV du VLM tronqué juste après `<|traj_future_start|>`.
4. Les configs livrées ne supervisent **que** `traj_future` (nav) ou `answer` (VQA). Superviser le texte CoC (ce que W3 exige) demande une variante `vla_processor` que le code supporte déjà mais qu'aucune config ne sélectionne.
5. Alpamayo 2 Super partage avec 1.5 l'action space (unicycle accel/courbure, 64 points, 0,1 s), l'échantillonneur Euler 10 pas, le mécanisme expert-sur-cache-KV, les seuils de pixels, le schéma 4 images à `t0 − 0,3 … t0`.
6. Il diffère par : backbone `Qwen3VLForConditionalGeneration` 32B (LM 64 couches × 5120, 64 têtes / 8 KV ; vision 27 couches, deepstack `[8, 16, 24]`), expert 64 couches × 1536 en attention non causale, 7 caméras chargées puis profil 6 caméras `[0,1,2,3,5,6]` pour la trajectoire, jetons trajectoire histoire-puis-futur (`<i0>` = 151 669, 1 000 bins ; futur `<i1000>` = 152 669, 3 000 bins) alors que 1.5 met le futur avant l'histoire, 45 jetons d'histoire (15 × 3, `pad_origin_at_beginning: false`) contre 48, méta-actions `Longitudinal/Lateral/Lane` entre `<|meta_action_start|>` et `<|meta_action_end|>`, cache KV prérempli une fois et dupliqué.
7. Le code d'inférence de 2 Super contient déjà des `forward` d'entraînement (`Alpamayo2Super.forward`, `ExpertModel.forward`, `FlowMatching.construct_training_data`). Il manque le pipeline de données, le collate, le masque de labels, le trainer et la gestion mémoire d'un 34B.
8. Plan : LoRA sur le modèle de langage du VLM (étape 1), expert 2B entraîné en plein ou LoRA (étape 2), sharding FSDP/ZeRO-3 ou base quantifiée sur 4×H100, smoke 10 exemples, non-régression par le harnais `afh`.
9. Format de la cible texte de 2 Super (D-008, levée) : `<|cot_start|>cot<|cot_end|><|meta_action_start|>…<|meta_action_end|><|traj_future_start|>…` ; les jetons existent dans `models/utils.py` et sont ajoutés au tokenizer par `config.py`, le masque de labels par composante de la recette s'applique tel quel (§2.3).
10. Pour W3 sur 1.5 (D-003), il faut d'abord un `vla_processor` supervisant `cot` + `traj_future` et un jeu d'annotations JSON au format `nav_demo_samples.json` étendu par nos manifestes `afh.uncertainty_dataset`.

---

## 1. La recette `alpamayo1_5_sft` telle qu'elle est

### 1.1 Chaîne de dépendances

```
nvidia/Alpamayo-1.5-10B (HF)  --convert_checkpoint.py to-a1-->  Alpamayo-1.5-10B-A1-format
                                                                (config.json remappé : alpamayo1_5.* -> alpamayo_r1.*,
                                                                 model_type "alpamayo_r1", architectures ["AlpamayoR1"] ;
                                                                 poids symlinkés, inchangés)
recipes/alpamayo1_5_sft  --importe-->  alpamayo_r1 (git NVlabs/alpamayo)   # modèles, action space, diffusion, géométrie
                         --importe-->  alpamayo (recipes/src)               # données PAI, processeur Qwen, templates, métriques
```

Points à retenir :

- `pyproject.toml` de la recette : Python 3.12, `torch==2.8.0`, `transformers==4.57.1`, `deepspeed==0.19.1`, `flash-attn>=2.8.3` (compilé), `physical_ai_av>=0.2.0`, `alpamayo_r1` depuis git.
- `convert_checkpoint.py` ne touche pas aux poids : « the model weights are architecturally identical — only config.json metadata differs ». Le script `convert_release_config_to_training.py` (recette RL) fait la même chose vers `ReasoningVLA` et fixe `vlm_name_or_path` à `nvidia/Cosmos-Reason2-8B`.
- Le VLM de 1.5 est instancié comme `Qwen3VLForConditionalGeneration` (`base_model.py`, backend `qwenvl3`) avec `vlm_name_or_path` par défaut `Qwen/Qwen3-VL-8B-Instruct` ; le processeur d'images de l'inférence 1.5 vient de `Qwen/Qwen3-VL-2B-Instruct` (`helper.BASE_PROCESSOR_NAME`) — seul le tokenizer est remplacé par celui du modèle.

### 1.2 Format de données attendu

**Sur disque (dataset PAI local, `PhysicalAIAVDatasetLocalInterface`)** :

| Fichier | Rôle |
|---|---|
| `features.csv` (index `feature`, colonne `clip_files_in_zip` en JSON) | table des features et de leurs fichiers par chunk |
| `clip_index.parquet` (index `clip_id`, colonne `chunk`) | liste des clips et leur chunk |
| `metadata/feature_presence.parquet` | booléens par clip et feature |
| `<feature>.chunk_XXXX.zip` (caméras : `video` + `frame_timestamps` parquet ; `egomotion` : parquet) | données brutes |
| `camera_intrinsics`, `sensor_extrinsics` (parquet) | optionnel (`include_extr_intr`) |
| `reasoning/ood_reasoning.parquet` (colonne `events` : liste de `{event_start_timestamp, coc|cot}`) | optionnel, labels CoC ; filtrés à `t0 ≥ 1,6 s` et `t0 + 6,4 s ≤ 20 s` |

Télécharger avec `scripts/download_pai.py --chunk-ids "…" --camera … --calibration … --labels egomotion`. La recette nav ne télécharge que **4 caméras** (`front_wide`, `cross_left`, `cross_right`, `front_tele`).

**Par échantillon (`load_physical_aiavdataset(clip_id, t0_us, avdi, num_history_steps=16, num_future_steps=64, time_step=0.1)`)** :

| Clé | Forme | Contenu |
|---|---|---|
| `image_frames` | `(N_cam, 4, 3, H, W)` uint8 | 4 images par caméra à `t0 − 0,3 / 0,2 / 0,1 / 0 s`, triées par index caméra croissant |
| `camera_indices` | `(N_cam,)` int64 | indices canoniques 0..6 (0 cross_left, 1 front_wide, 2 cross_right, 3 rear_left, 4 rear_tele, 5 rear_right, 6 front_tele) |
| `ego_history_xyz` / `ego_history_rot` | `(1, 1, 16, 3)` / `(1, 1, 16, 3, 3)` | 16 poses à 10 Hz jusqu'à `t0` inclus, dans le repère ego à `t0` (dernier point = origine) ; `PAIDataset` fait `squeeze(0)` → `(1, 16, 3)` |
| `ego_future_xyz` / `ego_future_rot` | `(1, 1, 64, 3)` / `(1, 1, 64, 3, 3)` | 64 poses de `t0 + 0,1 s` à `t0 + 6,4 s`, même repère |
| `relative_timestamps` / `absolute_timestamps` | `(N_cam, 4)` | horodatages images (secondes depuis le min / µs relatifs au clip) |
| `t0_us`, `clip_id` | scalaires | |
| `nav_text` (nav), `cot` (reasoning), `question`/`answer` (LingoQA) | str | ajoutés par le dataset selon la tâche |

Le `t0` par défaut est `5 100 000 µs` (`use_default_keyframe: true`) ; avec `reasoning_metadata`, `t0` = `event_start_timestamp` de l'événement CoC.

**Annotations nav (`PAIDatasetWithNav`)** : liste JSON `[{"clip_id", "t0_relative" (µs), "nav_text", "cot"?}]`. Un élément = un exemple ; les clips hors `chunk_ids` sont filtrés. C'est le format le plus simple à réutiliser pour nos propres cibles (§3).

**Prétraitement (`alpamayo.processor.qwen_processor.QwenProcessor._preprocess_data`)**, exécuté dans `__getitem__` :

1. tri des images par `camera_indices` (asserté croissant dans `construct_image`) ;
2. construction de la conversation par `get_template("r1_5").build_conversation(...)` :
   - système : `You are a driving assistant that generates safe and accurate actions.` ;
   - utilisateur, dans l'ordre de `components_order` : `image` (avec, si `include_camera_ids`/`include_frame_nums`, le texte `Front camera: ` puis `frame k ` avant chaque image), `traj_history` (`<|traj_history_start|>` + `<|traj_history|>` × `tokens_per_history_traj` + `<|traj_history_end|>`), `route` (`<|route_start|>{nav_text}<|route_end|>`, absent si `nav_text` manque), `question`, `prompt` (phrase construite : `output the chain-of-thought reasoning of the driving process, then output meta actions, then output the future trajectory.` selon les composantes demandées) ;
   - assistant : `cot` (`<|cot_start|>{data["cot"]}<|cot_end|>`), `meta_action` (`<|meta_action_start|>{data["meta_action_strings"]}<|meta_action_end|>`), `traj_future` (`<|traj_future_start|>` + `<|traj_future|>` × `tokens_per_future_traj` + `<|traj_future_end|>`), `answer` ;
   - en `generation_mode`, la **dernière** composante n'a que son jeton d'ouverture (`ask_for_component`) et le prompt utilisateur liste `components_prompt` au lieu de `components_order`.
3. `apply_chat_template(tokenize=False, continue_final_message=generation_mode)` ;
4. images `uint8 → float/255`, `image_processor(images, do_rescale=False)` ; chaque `<image>` est développé en `image_grid_thw.prod() // merge_size²` jetons image ;
5. retour `{"text", "pixel_values", "image_grid_thw"}` stocké sous `tokenized_data`.

**Collate (`QwenProcessor.collate_fn`)** : padding **à gauche**, tokenisation du texte par lot, concaténation des `pixel_values`/`image_grid_thw`, `image_frames` non empilés. `labels_mask` = union des spans `<|X_start|> … <|X_end|>` (fin incluse) pour chaque `X` de `label_components` (`get_label_mask`), plus le `<|im_end|>` de l'assistant (`get_role_eos_mask`). En `generation_mode`, masque tout à `False`.

**Fusion des jetons trajectoire (`fuse_traj_tokens`, dans le `forward`)** : les `<|traj_history|>` sont remplacés par `hist_traj_tokenizer.encode(...) + hist_token_start_idx`, les `<|traj_future|>` par `traj_tokenizer.encode(...) + future_token_start_idx` (étape 1 seulement, via `TrajectoryFusionWithFutureMixin`). Les jetons discrets sont `<i0> … <i{traj_vocab_size-1}>` ajoutés au tokenizer ; `future_token_start_idx = hist_token_start_idx = id(<i0>)`, puis `hist_token_start_idx += traj_tokenizer.vocab_size` si un tokenizer d'histoire séparé existe. **Donc sur 1.x le bloc futur précède le bloc histoire dans le vocabulaire.**

Valeurs pour 1.5 **[config]** (`docs/configs/alpamayo-1.5-10b/config.json`) : `traj_vocab_size: 4000`, `traj_token_start_idx: 151669`, `tokens_per_history_traj: 48`, `tokens_per_future_traj: 128`, `vocab_size: 155697`, `min_pixels 163840`, `max_pixels 196608`, `vlm_name_or_path nvidia/Cosmos-Reason2-8B`, `attn_implementation flash_attention_2`, `add_special_tokens: true`, `include_camera_ids`/`include_frame_nums: true`. Histoire : `DeltaTrajectoryTokenizer` par défaut (1 000 bins, origine préfixée → 16 deltas × (dx, dy, dz) = 48). Futur : `DiscreteTrajectoryTokenizer` sur l'action space unicycle (64 × (accélération, courbure) = 128, `dims_min/max ±10`, `num_bins 3000`). Le futur occupe `<i0>…<i2999>` (151 669 → 154 668), l'histoire `<i3000>…<i3999>` (154 669 → 155 668) ; jetons trajectoire spéciaux : `history_start 155674`, `history_end 155676`, `future_start 155681`, `future_end 155683`, `history 155684`, `future 155685`.

**Variantes `vla_processor` livrées** :

| Variante | `components_order` | `label_components` | ids caméra / n° image |
|---|---|---|---|
| `default` | image, traj_history, prompt, traj_future | traj_future | non / non |
| `nav` | image, traj_history, **route**, prompt, traj_future | traj_future | oui / oui |
| `vqa` | image, question, answer | answer | oui / oui |

Aucune ne supervise `cot`. La composante existe pourtant dans le template (`construct_cot`) et le prompt (`"output the chain-of-thought reasoning of the driving process"`), et `PAIDataset` fournit `data["cot"]` dès que `reasoning_metadata` est renseigné. Les métriques d'évaluation (`ReasoningSampler`, `traj_only_generation: false`, `max_generation_length: 256`) échantillonnent bien CoC + trajectoire.

### 1.3 Étape 1 — VLM par entropie croisée

**Classe** : `TrainableReasoningVLA.from_alpamayo_checkpoint(checkpoint_path, vlm_name_or_path="Qwen/Qwen3-VL-8B-Instruct")` (`models/sft_base_model.py`). Construit `Qwen3VLForConditionalGeneration` depuis la config (vocabulaire déjà agrandi), charge les tenseurs `vlm.*` des shards safetensors (`load_alpamayo1_vlm`), instancie `traj_tokenizer` depuis `traj_tokenizer_cfg`. Pas d'expert, pas de diffusion.

**`forward(tokenized_data, ego_history_*, ego_future_*, labels_mask)`** :

```
input_ids  = fuse_traj_tokens(input_ids, traj)            # histoire ET futur remplacés par des <i…>
labels     = where(labels_mask, input_ids, -100)
outputs    = vlm(input_ids, labels, pixel_values, image_grid_thw, attention_mask)
traj_mask  = (labels ∈ [future_token_start_idx, +traj_vocab_size)) | labels == <|traj_future_start|> | labels == <|traj_future_end|>
loss       = CE_mean(logits, labels | traj_mask)  +  CE_mean(logits, labels | (labels != -100) & ~traj_mask)
```

Ce que chaque terme entraîne :

- **`future_traj`** : prédire, jeton par jeton (décalage d'un cran, `_compute_next_token_loss`), le jeton d'ouverture, les 128 jetons `<i…>` du futur discret et le jeton de fermeture. C'est ce qui donne au VLM seul une capacité de planification (métriques `min_ade` d'`evaluate_hf.py` sur l'étape 1 possible via `ReasoningSampler`).
- **`others`** : tout label restant dans le masque : texte CoC si `cot` est supervisé, `answer` en VQA, le `<|im_end|>` assistant. Les deux moyennes sont **sommées avec un poids 1 chacune** (les `loss_weights` de la config sont commentés dans le code), donc un exemple à 20 jetons de texte pèse autant que ses 130 jetons de trajectoire.
- Jetons image, histoire, prompt : jamais dans le masque, donc jamais dans la perte ; le VLM apprend uniquement à **générer** les composantes assistant.
- Paramètres entraînés : **tout le VLM** (vision, projecteur, LM, embeddings agrandis). `lr_multiplier: {vlm.model.visual: 0.1}` (`ReasoningVLA_Trainer.create_optimizer`, préfixe le plus long gagne, decay/no-decay séparés).

**Hyperparamètres livrés (`sft_base.yaml` + `sft_stage1_nav.yaml`)** : batch 1 par GPU, accumulation 4 (→ 32 séquences/pas sur 8 GPU), LR 1e-5, warm-up 500, cosinus vers `min_lr 1e-6`, bf16, DeepSpeed ZeRO-2 (`overlap_comm`, buckets 2e8), gradient checkpointing non réentrant, `dataloader_num_workers 8`, `remove_unused_columns: False`, `ddp_find_unused_parameters: false`. Le `ds_config._dtype` est forcé à float32 pour ne pas caster les entrées (l'encodeur de trajectoire veut du float32). `num_train_epochs 700` sur les 20 exemples nav = test de surapprentissage (la perte doit tomber vers 0 en quelques centaines de pas).

### 1.4 Étape 2 — expert de diffusion par flow matching

**Classe** : `TrainableAlpamayoR1.from_pretrained(pretrained_model_name_or_path=<A1-format>, cotrain_vlm=False, stage1_vlm_checkpoint_path=<sortie étape 1>)` (`models/sft_alpamayo_r1.py`). Charge le checkpoint complet (VLM + expert + projections), puis écrase les poids `vlm.*` par ceux de l'étape 1 (`preserve_model_device_and_dtype=True`), puis `requires_grad = cotrain_vlm` sur le VLM.

**Architecture de l'expert (identique dans `alpamayo_r1` et `alpamayo1_5`)** : une copie du décodeur texte du VLM (`AutoModel.from_config(text_config)` avec les surcharges `expert_cfg`, sans `embed_tokens`), attention **non causale** entre les jetons d'action (`expert_non_causal_attention: True`) ; forcé en `sdpa` si le VLM est en flash-attention. Pour 1.5 **[config]** : `expert_cfg = {hidden_size 2048, intermediate_size 8256, num_attention_heads 16, head_dim 128}`, le nombre de couches étant celui du `text_config` de Cosmos-Reason2-8B. `action_in_proj = PerWaypointActionInProjV2` **[config]** : encodage de Fourier log-espacé (`num_fourier_feats 20`, `max_freq 100`) de chaque dimension d'action et du pas de temps, MLP `num_enc_layers 2` × `hidden_size 512`, LayerNorm → un jeton par point de trajectoire. `action_out_proj` : `torch.nn.Linear` `hidden → 2`. Action space **[config]** : `accel_mean 0.029`, `accel_std 0.681`, `curvature_mean 2.7e-4`, `curvature_std 0.0261` (mêmes valeurs dans les deux checkpoints).

**Action space** : `UnicycleAccelCurvatureActionSpace` (64 points, `dt 0,1`, accélération bornée ±9,8 m/s², courbure ±0,33 m⁻¹, normalisées par moyenne/écart-type). `traj_to_action` ajuste (a, κ) aux 64 poses futures par lissage régularisé ; `action_to_traj` intègre depuis la dernière pose d'histoire.

**`forward`** (`sft_alpamayo_r1.py`) :

```
input_ids   = fuse_traj_tokens(...)                       # histoire seulement fusionnée (mixin de base)
vlm_outputs = vlm(input_ids, labels, use_cache=True, …)   # sous no_grad si cotrain_vlm=False
cut         = dernier index de <|traj_future_start|> + 1
x           = action_space.traj_to_action(hist, fut)      # (B, 64, 2)
t ~ 0.999 − 0.999·Beta(1.5, 1)      noise ~ N(0, I)      noisy_x = t·x + (1−t)·noise
emb         = action_in_proj(noisy_x, t)                  # (B, 64, hidden)
kv          = vlm_outputs.past_key_values.crop(cut) ; keys/values détachés
pos_ids     = MRoPE 3×(B,64) décalés de rope_deltas + longueur du cache
pred        = action_out_proj(expert(inputs_embeds=emb, past_key_values=kv, is_causal=False)).view(B, 64, 2)
loss        = MSE(pred, x − noise)     (+ vlm_outputs.loss si cotrain_vlm)
```

Ce que la perte entraîne : le champ de vitesse `v(x_t, t)` d'un flot rectiligne bruit → action, conditionné **uniquement** par le cache KV du préfixe (images, histoire, prompt, CoC cible, jeton `<|traj_future_start|>`). L'échantillonnage (`FlowMatching._euler`) part de `x ~ N(0, I)` (facteur `temperature` sur ce bruit dans les releases 1.5 et 2S, absent d'`alpamayo_r1`), intègre 10 pas Euler `x ← x + Δt·v` de `t = 0` à `1`, puis `action_to_traj`. Le lien texte → trajectoire n'existe donc que par le cache KV : c'est l'endroit exact où D-007 (texte et action jamais contradictoires) se joue à l'entraînement, et où le harnais mesure la fidélité à l'évaluation.

Paramètres entraînés : expert + `action_in_proj` + `action_out_proj` (+ VLM si `cotrain_vlm`). Config : **pas de DeepSpeed, pas de gradient checkpointing** (`sft_stage2_nav.yaml`), `ddp_find_unused_parameters: true` (le VLM gelé), LR 1e-4 par défaut de `sft_base`, 3 époques, batch 1/GPU.

Note : la release `alpamayo1_5` ne contient **pas** `construct_training_data` / `compute_loss_from_pred` (sa `FlowMatching` est inférence seule) — la recette les prend dans `alpamayo_r1`. La release `alpamayo2_super`, elle, les embarque, et son `diffusion_cfg` **[config]** fixe `train_timestep_sampler: beta`, `train_ignore_guidance_rate: 0.1`, `inference_guidance_weight: 3.0`, `use_classifier_free_guidance: false` (le `diffusion_cfg` de 1.5 ne porte que `int_method: euler`).

### 1.5 Configuration matérielle

| Élément | Valeur |
|---|---|
| Validation NVIDIA | **8× H100 80 GB, un nœud**, `torchrun --nproc_per_node 8` |
| Étape 1 | ZeRO-2 + grad-ckpt obligatoires ; en cas d'OOM : garder batch 1, monter l'accumulation, ne jamais couper le grad-ckpt |
| Étape 2 | ni DeepSpeed ni grad-ckpt ; en cas d'OOM : baisser le batch, geler plus (`cotrain_vlm: false` déjà), ou plus de 8 GPU |
| Inférence 1.5 (README) | ~24 GB (1 échantillon), ~40 GB (16 échantillons), ~60 GB (16 + CFG) sur H100 |
| flash-attn | compilé à l'installation, 5–10 min par nœud ; `sdpa` documenté comme repli à l'inférence 1.5 |
| Pipeline données | `performance.zip_cache` (cache des zips par instance), `collate_cache`, tf32, cudnn benchmark ; `docs/training_efficiency/` documente le décodage vidéo en place |
| Multi-nœud | mêmes chemins absolus partout (données, annotations, checkpoint, venv) |

Rien dans la recette ne donne un temps par pas ni un débit. Pour notre budget (4×H100, D-004), l'étape 1 en plein fine-tuning de 8 B en ZeRO-2 est déjà serrée (poids bf16 16 GB + états Adam fp32 partitionnés sur 4 ≈ 24 GB/GPU + activations) ; LoRA est ce qui la rend confortable.

---

## 2. Diff exhaustif Alpamayo 1.5 vs Alpamayo 2 Super (codes d'inférence)

Convention : **1.5** = `alpamayo1_5` `36aeb4c` ; **2S** = `alpamayo2_super` `5e7975f`.

### 2.1 Backbone et classes

| | 1.5 | 2S |
|---|---|---|
| Taille (README) | 10 B : VLM Cosmos-Reason2 8,2 B + expert 2,3 B | 34 B : VLM Cosmos 3 32 B + expert 2 B |
| Classe VLM | `Qwen3VLForConditionalGeneration` codée en dur (`_initialize_qwenvl3_vlm`), config lue depuis `vlm_name_or_path` (`nvidia/Cosmos-Reason2-8B`) puis `vocab_size` remplacé (155 697) | `vlm_class: Qwen3VLForConditionalGeneration` **[config]** ; `vlm_config` imbriqué : `text_config` 64 couches, `hidden_size 5120`, `intermediate_size 25600`, 64 têtes / 8 KV, `head_dim 128`, MRoPE entrelacé `[24, 20, 20]`, `rope_theta 5e6`, `max_position_embeddings 262144`, `vocab_size 155776` (155 697 arrondi au multiple de 128) ; `vision_config` 27 couches, `hidden_size 1152`, 16 têtes, `patch_size 16`, `spatial_merge_size 2`, `temporal_patch_size 2`, `deepstack_visual_indexes [8, 16, 24]`, `out_hidden_size 5120` ; `image_token_id 151655` |
| Indices MRoPE | `position_ids` 3×(B, L) décalés par `rope_deltas` (« Qwen 2.5 VL »-style) | identique (`build_expert_pos_ids_and_attn_mask`, commentaire « Qwen-VL MRoPE ») |
| Tokenizer | `AutoProcessor.from_pretrained(vlm_name_or_path)`, `<i0>…<i{traj_vocab_size-1}>` puis jetons spéciaux (si `add_special_tokens`) | `AutoTokenizer.from_pretrained(chemin_checkpoint, fix_mistral_regex=True)`, `<i0>…<i999>` (histoire) puis `<i1000>…<i3999>` (futur), puis jetons spéciaux |
| Processeur image | `Qwen/Qwen3-VL-2B-Instruct` + `min_pixels 163840`, `max_pixels 196608` | celui du checkpoint, mêmes seuils par défaut, `fix_mistral_regex` |
| Config | `Alpamayo1_5Config(ReasoningVLAConfig)` : `traj_vocab_size`, `tokens_per_*`, `attn_implementation` (défaut flash-attn 2), `keep_same_dtype`, `include_camera_ids: False`, `include_frame_nums: False` | `Alpamayo2SuperConfig(PretrainedConfig)` : `history_vocab_size 1000`, `future_vocab_size 3000`, `tokens_per_history_traj 48` (vérifié = 3 × 16 par assertion), `tokens_per_future_traj 128`, `padding_side left`, `include_camera_ids: True`, `token_layout camera_ts`, `frame_label frame_num`, `loss_weights {future_traj 1, others 1}`, `cotrain_expert_vlm`, `enable_expert`, `expert_config` (`ExpertModelConfig` avec son propre `llm_config`), `logit_cap` optionnel |
| Expert | copie du `text_config` du VLM avec surcharges `expert_cfg` (`hidden_size 2048`, `intermediate_size 8256`, 16 têtes **[config]**), attribut direct `model.expert`, `action_in_proj`/`action_out_proj`/`diffusion`/`action_space` au niveau du modèle | module `ExpertModel` séparé (`model.expert.expert`, `.action_in_proj`, `.action_out_proj`, `.diffusion`, `.action_space`), `llm_config` **[config]** = `qwen3_vl_text` 64 couches, `hidden_size 1536`, `intermediate_size 6144`, 16 têtes / 8 KV, `head_dim 128`, même MRoPE que le VLM ; `expert_non_causal_attention: true` ; `_keys_to_ignore_on_load_missing` pour les buffers Fourier et les statistiques d'action space (persistants en 2S, non persistants en 1.5) |
| Gel | rien n'est gelé à l'inférence | `vlm.requires_grad_(False)` si `enable_expert` et non `cotrain_expert_vlm` ; le checkpoint **[config]** a `enable_expert: true`, `cotrain_expert_vlm: false`, `loss_weights {future_traj 1.0, others 1.0}`, pas de `logit_cap` |
| Chargement | `.from_pretrained(id, dtype=bf16[, attn_implementation="sdpa"]).to("cuda")` | `.from_pretrained(id, dtype=bf16, device_map="cuda:0")` |
| Forward d'entraînement dans la release | non (`ReasoningVLA` n'a que `generate_text`) | **oui** : `Alpamayo2Super.forward(tokenized_data, traj_data, labels_mask)` (CE deux termes pondérés par `loss_weights`, `logit_cap` tanh) et `ExpertModel.forward(traj_data, vlm_outputs, cache_attention_mask)` (flow matching) |

### 2.2 Caméras : nombre et ordre

| | 1.5 | 2S |
|---|---|---|
| Chargeur par défaut | **4** : `[cross_left, front_wide, cross_right, front_tele]` → indices `[0, 1, 2, 6]` | **7** : anneau complet `[0, 1, 2, 3, 4, 5, 6]` ; ajoute `camera_names`, `camera_calibrations` (intrinsèques/extrinsèques), `ego_t0*`, `ego_*_tvals`, `ego_t0_xyz`, `ego_t0_inv_quat`, `prediction_start_offset` |
| Sélection par tâche | libre : le notebook `inference_cam_num` montre 1, 2 ou 4 caméras ; le nom de caméra inconnu tombe sur l'index 0 (`.get(cam_name, 0)`) | **profils fixes** (`input_profiles.py`) : `trajectory`, `meta_action`, `auto_labeling`, `grounding` → `[0, 1, 2, 3, 5, 6]` (sans `rear_tele`) ; `vqa` → `[0, 1, 2, 3, 4, 5]` (sans `front_tele`) ; `select_task_input` **exige** l'anneau 7 en entrée, dans l'ordre, et enregistre `input_profile` |
| Ordre | tri croissant par index caméra dans le chargeur ; `construct_image` asserte l'ordre croissant | identique |
| Images par caméra | 4, à `t0 − 0,3 / 0,2 / 0,1 / 0 s` | identique (`frame_indices (0,1,2,3)`) |
| Étiquettes texte | `Front camera: ` sur la première image de chaque caméra puis `frame k ` avant chaque image, **seulement si** `camera_indices` est passé à `create_message` | toujours (`include_camera_ids True`, `frame_label frame_num`) ; mêmes noms d'affichage (`CAMERA_INDICES_TO_DISPLAY_NAMES`) |
| Images passées au processeur | tenseurs uint8 dans le contenu des messages, `apply_chat_template(tokenize=True)` | `images.flatten(0,1).float()/255`, `processor(text, images, do_rescale=False)` |

Conséquence pour `afh` (D-010, adoptée) : avec le chargeur 4 caméras de 1.5, l'index tenseur 3 est `front_tele` (caméra 6), pas `rear_left` ; depuis alpamayo-faithfulness PR #7 (`afh/cameras.py`), `apply_degradation`, `target_severity` et `target_text` prennent `camera_indices` et raisonnent en identifiants caméra. `runners/probe_blackout_a15.py` le passe et l'enregistre dans le JSON.

### 2.3 Jetons spéciaux, jetons de raisonnement et de trajectoire

Listes `SPECIAL_TOKENS_KEYS`, dans l'ordre d'ajout au vocabulaire (l'ordre fixe les ids) :

| Position | 1.5 | 2S (= `alpamayo_r1`) |
|---|---|---|
| 0–2 | `prompt_start`, `prompt_end`, `image_start` | idem |
| 3 | `_padding_0` | `image_pre_tkn` |
| 4–5 | `image_end`, `traj_history_start` | idem |
| 6 | `_padding_1` | `traj_history_pre_tkn` |
| 7–9 | `traj_history_end`, `cot_start`, `cot_end` | idem |
| 10–11 | `_padding_2`, `_padding_3` | **`meta_action_start`, `meta_action_end`** |
| 12 | `traj_future_start` | idem |
| 13 | `_padding_4` | `traj_future_pre_tkn` |
| 14–17 | `traj_future_end`, `traj_history`, `traj_future`, `image_pad` | idem |
| 18–21 | `_padding_5…8` | `vectorized_wm`, `vectorized_wm_start`, `vectorized_wm_end`, `vectorized_wm_pre_tkn` |
| 22–28 | `route_start`, `route_pad`, `route_end`, `question_start`, `question_end`, `answer_start`, `answer_end` | idem |

Les 29 positions coïncident : 1.5 a masqué par des `_padding_k` les jetons qu'elle n'expose pas (méta-actions, world model vectorisé). **1.5 n'a donc aucun jeton de méta-action nommé.**

Jetons de trajectoire :

| | 1.5 | 2S |
|---|---|---|
| Vocabulaire | `traj_vocab_size 4000` **[config]**, `<i0>` = 151 669 ; **futur d'abord** (`future_token_start_idx` = 151 669, 3 000 bins), histoire ensuite (`hist_token_start_idx` = 154 669, 1 000 bins) | **histoire d'abord** (`history_id0` = 151 669, 1 000 bins), futur ensuite (`future_id0` = 152 669, 3 000 bins) **[config]** ; mêmes ids de jetons spéciaux de trajectoire que 1.5 (`history_start 155674` … `future_pad 155685`) |
| Histoire | **48** jetons = 16 × (dx, dy, dz) : `DeltaTrajectoryTokenizer` par défaut **[config]** (origine préfixée, ±4 m en xy, ±10 m en z, 1 000 bins) ; `hist_xyz[:, :1]` passé en histoire, la trajectoire d'histoire en « futur » | **45** jetons = 15 × 3 : `pad_origin_at_beginning: false` **[config]** (les 16 poses finissant à l'origine donnent 15 deltas, décodés par somme cumulée inversée) ; l'assertion de `Alpamayo2SuperConfig` vérifie `45 == 3 × 15` |
| Futur | 128 jetons = 64 × (a, κ) : `DiscreteTrajectoryTokenizer` **[config]**, `dims_min/max ±10`, 3 000 bins, action space unicycle | identique **[config]** ; `MaskDiscreteTrajectoryLogitsProcessor` masque `[151669, 155669)` pendant la génération texte |
| Fusion | `TrajectoryFusionMixin.fuse_traj_tokens` (histoire seule à l'inférence ; futur aussi via le mixin de la recette) ; `replace_pad_token` = `masked_scatter` silencieux | `models.utils.fuse_traj_tokens(hist_tok, fut_tok, input_ids, traj_data, traj_ids)` ; **vérifie** que le nombre de `<|traj_history|>`/`<|traj_future|>` par ligne égale le nombre d'ids encodés (ValueError sinon) ; futur fusionné dès que `ego_future_xyz` est dans `traj_data` (auto-labeling) |

Jetons de raisonnement :

| | 1.5 | 2S |
|---|---|---|
| Ouverture du CoC | l'assistant commence par `<|cot_start|>` (`create_message`, `continue_final_message=True`) ; le modèle génère `… <|cot_end|><|traj_future_start|>` | **format balisé (D-008)** : `SPECIAL_TOKENS_KEYS` (`models/utils.py`) contient `cot_start`/`cot_end` (ids 155 677 / 155 678, ajoutés au tokenizer par `build_alpamayo2_super_tokenizer` dans `config.py`) ; `get_component_str` n'émet que le jeton d'ouverture d'une composante demandée, de sorte qu'un prompt terminé par `<|cot_start|>` fait générer `cot <|cot_end|> <|meta_action_start|>…<|meta_action_end|> <|traj_future_start|>…`. Lecture du code de la release (`5e7975f`) à garder en tête : `helper.create_messages` passe `components_order=["image","traj_history","prompt"]` et `components_prompt=["cot","traj_future"]`, donc `build_conversation` ne construit pas la composante `cot`, le tour assistant vide est retiré et le prompt part avec `add_generation_prompt=True` ; c'est `build_conversation` en mode entraînement (`cot` dans `components_order`) qui pose `<|cot_start|>…<|cot_end|>`. `extract_text_tokens` ne cherche pas ces jetons (il coupe à `<|traj_future_start|>`/`<|im_end|>`) |
| Extraction | `extract_between_special_tokens` sur `cot`, `meta_action`, `answer` | `extract_text_tokens` : texte après le dernier `<|im_start|>assistant\n`, coupé à `<|traj_future_start|>` ou `<|im_end|>` ; puis `split_cot_and_meta_action` (regex `Longitudinal:|Lateral:|Lane:`) ; renvoie aussi `raw_outputs`, `answer`, `box`, `cot_auto_labeling`. Pour l'entraînement, le masque de labels par composante (`get_label_mask` sur `<|cot_start|>…<|cot_end|>`, `<|meta_action_start|>…<|meta_action_end|>`, `<|traj_future_start|>…<|traj_future_end|>`) s'applique tel quel |
| Prompt trajectoire | `output the chain-of-thought reasoning of the driving process, then output the future trajectory.` | identique via `construct_user_prompt(components_prompt=["cot", "traj_future"])` |
| Arrêt | `StopAfterEOS(<|traj_future_start|>)` puis `replace_padding_after_eos` | identique, plus `_append_text_eos_mask` (masque les `eos_token_id` texte, sauf l'ancre) |
| Génération | `vlm.generate(num_return_sequences=num_traj_samples, output_logits=True)` ; `rope_deltas` lus sur `vlm.model` | `_generate_with_shared_prefill` : **prefill une fois** de `input_ids[:, :-1]` avec les pixels, cache dupliqué `batch_repeat_interleave(n)`, puis `generate(num_return_sequences=1, output_logits=False)` ; `max_new_tokens = max(256, 128)` |
| Signature | `sample_trajectories_from_data_with_vlm_rollout(data, top_p, top_k, temperature, num_traj_samples, num_traj_sets, diffusion_kwargs, max_generation_length=, return_extra=)` → `(pred_xyz, pred_rot[, extra])` | `sample_trajectories_from_data(data, top_p, top_k, temperature, num_traj_samples=1, num_traj_sets, max_generation_length, diffusion_kwargs, return_extra)` → `(pred_xyz, pred_rot, logprob[, extra])`, `logprob` = zéros |
| Nav | `<|route_start|>{nav}<|route_end|>` dans le texte utilisateur ; CFG dédié `..._cfg_nav` qui rejoue le préfixe sans le span route | pas de jetons route : composante `nav_instruction` = texte brut (`two_gpu_nav_cfg_demo.prepare_nav_model_inputs`) ; `ExpertModel` **refuse** `use_classifier_free_guidance` ; le CFG nav vit dans la démo 2 GPU (VLM sur `cuda:0`, expert et deux caches sur `cuda:1`, ~67 + 71 GiB) |
| VQA | `<|question_start|>…<|question_end|>` / `<|answer_start|>` (`create_vqa_message`, `generate_text`) | question en texte brut, sans histoire ; réponse libre ; grounding = même chemin avec une question de boîte (`DEFAULT_GROUNDING_QUESTION`), sortie JSON `bbox_2d` détectée par `_looks_like_grounding_box_text` |

### 2.4 Méta-actions

| | 1.5 | 2S |
|---|---|---|
| Existence | aucune (jetons remplacés par `_padding_2/3`, pas de template) | tâche texte `meta_action` (`text_tasks.py`) |
| Prompt | — | `output the chain-of-thought reasoning of the driving process, then output meta actions, then output the future trajectory.` (`_TASK_PROMPTS["meta_action"] = ["cot", "meta_action", "traj_future"]`) ; utilisateur = images + histoire + prompt, `add_generation_prompt=True` |
| Format de sortie | — | texte CoC puis un bloc commençant à la première occurrence de `Longitudinal:`, `Lateral:` ou `Lane:` (`_META_ACTION_SPLIT_RE`) ; les valeurs ne sont pas énumérées dans le code (le notebook n'a pas de sortie enregistrée) — **à capturer sur GPU** |
| Rôle dans la trajectoire | — | la génération s'arrête à `<|traj_future_start|>` comme pour le CoC ; `generate_text` ne lance pas l'expert (`max_new_tokens 512`) ; pour obtenir aussi une trajectoire il faudrait un `sample_trajectories_from_data` avec `components_prompt` incluant `meta_action`, ce que `helper.create_messages` ne fait pas (liste codée en dur `["cot", "traj_future"]`) |
| Côté template | `construct_meta_action` existe dans les templates de la recette (`alpamayo.chat_template.components`) avec `<|meta_action_start|>…<|meta_action_end|>` et `data["meta_action_strings"]` | même `construct_meta_action` dans `chat_template/conversation.py` ; jetons `meta_action_start` 155 679 / `meta_action_end` 155 680 dans le tokenizer ; cible d'entraînement balisée comme le CoC (D-008) |

### 2.5 `prepare_model_inputs` et le chemin des entrées

| Étape | 1.5 (pas de fonction dédiée) | 2S `helper.prepare_model_inputs(data, model.config, model.tokenizer)` |
|---|---|---|
| Messages | `helper.create_message(frames.flatten(0,1), camera_indices, nav_text=None, use_nav_prompt=False)` : chaîne système + images (tenseurs) + texte utilisateur `traj_history(48) [route] prompt` + assistant `<|cot_start|>` | `create_messages` → `build_conversation(components_order=["image","traj_history","prompt"], components_prompt=["cot","traj_future"], generation_mode=True, include_camera_ids, camera_ids=data["camera_indices"], include_frame_nums)` ; tour assistant vide supprimé |
| Texte | `processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, continue_final_message=True, return_dict=True, return_tensors="pt")` — tokenisation et images en un appel | `apply_chat_template(tokenize=False, add_generation_prompt=not has_assistant_content, add_vision_id=False, continue_final_message=has_assistant_content)` puis `processor(text=, images=, videos=None, padding=False, return_tensors="pt", do_rescale=False)` |
| Images | uint8 telles quelles | `float()/255` si uint8 |
| Lot | ce que le processeur produit | **un seul échantillon** (ValueError sinon) — le batching multi-clips reste à écrire pour l'entraînement |
| Sortie | `{"tokenized_data": BatchEncoding, "ego_history_xyz", "ego_history_rot"}` construit par l'appelant | même dictionnaire, construit par la fonction |
| Placement | `helper.to_device(model_inputs, "cuda")` | identique |
| Fusion trajectoire | dans le modèle, histoire seule | dans le modèle, histoire (+ futur si présent) |
| Tâches texte | `create_vqa_message` | `text_tasks.prepare_text_generation_inputs(data, config, tokenizer, task, future_xyz, future_rot, question)` : `meta_action` (images + histoire + prompt), `auto_labeling` (images + histoire + **futur vrai ou prédit** + prompt JSON ; validation stricte des tvals et des horodatages caméra ≤ t0 + 0,05 s), `vqa` (images + question) |

### 2.6 Diffusion et expert : ce qui est identique

- `FlowMatching` : Euler `time_steps = linspace(0, 1, n+1)`, `x ← x + Δt·v`, `temperature` sur le bruit initial, `inference_step` par défaut 10, `_guided_v` CFG `(1−w)·v_u + w·v_g`.
- `PerWaypointActionInProjV2`, `MLPEncoder`, `FourierEncoderV2`, `RMSNorm` : mêmes formules (2S cast explicitement en float32 puis vers le dtype du MLP).
- `UnicycleAccelCurvatureActionSpace`, `DiscreteTrajectoryTokenizer`, `DeltaTrajectoryTokenizer` (2S ajoute `pad_origin_at_beginning`), `geometry.rotation`, `diffusion_expert_cuda_graph.py` (fichier identique octet pour octet).
- Conditionnement : `offset = find_eos_offset(sequences, <|traj_future_start|>) + 1`, masque 4D bloquant `[offset, −n_diffusion_tokens)` et le padding gauche, `position_ids = rope_deltas + offset + arange(64)`, `prompt_cache.crop(prefill_seq_len)` après chaque pas.

---

## 3. Plan de portage LoRA, par étapes

Chaque étape produit un artefact vérifiable seul (principe 1 de la roadmap). Les étapes 0 à 2 sont à froid ou sur un pod d'inférence ; 3 à 6 demandent 4×H100.

### Étape 0 — Configuration (faite) et premières observations sur le pod (1 h)

Les configs sont dans `docs/configs/` (inconnues 2, 3, 7 levées) et le format de cible CoC est fixé par D-008 (inconnue 1). Reste sur le pod : un rollout avec `return_extra=True` pour archiver `extra["raw_outputs"][0]` (vérifie que `<|cot_end|>` et `<|traj_future_start|>` sont bien émis dans cet ordre) et une sortie `meta_action` pour le vocabulaire `Longitudinal/Lateral/Lane` (inconnue 4).

### Étape 1 — Adaptateur de données (froid, testable)

Réutiliser, sans les copier, `alpamayo.data.pai.PAIDataset` et `PhysicalAIAVDatasetLocalInterface` (recette) en remplaçant le chargeur par `alpamayo2_super.load_physical_aiavdataset` (7 caméras) suivi de `select_task_input(data, "trajectory")`. Sortie par exemple :

```
image_frames (6, 4, 3, H, W) uint8 · camera_indices [0,1,2,3,5,6] · ego_history_* (1, 16, …) · ego_future_* (1, 64, …)
cot : str (cible texte) · [meta_action_strings : str] · nav_text : absent
```

Nos cibles viennent de `afh.uncertainty_dataset` (manifeste JSONL : clip, `t0_us`, spec de dégradation, `target_text`, `target_trajectory`). L'adaptateur applique `afh.degradation.apply_degradation` sur `image_frames` **après** `select_task_input` (les indices tenseur 0–5 correspondent alors aux caméras `[0,1,2,3,5,6]` : remapper `cameras` du spec en conséquence ou faire évoluer `afh` pour raisonner en ids caméra), remplace `ego_future_xyz[..., :2]` par la trajectoire cible amortie (z conservé) et met `cot = target_text`. 40 % d'exemples propres (`CLEAN_FRACTION`), cible = raisonnement propre du modèle capturé par `runners/run_inference_a2.py`. Le test D-007 (`tests/test_uncertainty_cold.py` du harnais) reste la garde.

Format d'annotation : étendre `nav_demo_samples.json` en `[{"clip_id", "t0_relative", "cot", "degradation": {...}, "target_future_xy": [...]}]` pour rester lisible par un `PAIDatasetWithNav` dérivé.

### Étape 2 — Processeur et collate pour 2S (froid, testable avec un tokenizer factice)

Écrire un `Alpamayo2Processor` miroir de `QwenProcessor` : `build_conversation` en `generation_mode=False` avec `components_order = ["image", "traj_history", "prompt", "cot", "traj_future"]` (ou `[..., "cot", "meta_action", "traj_future"]`), `apply_chat_template(tokenize=False)`, `processor(text, images)` comme `prepare_model_inputs`, développement des jetons image identique, padding gauche (`config.padding_side`). `labels_mask` : `get_label_mask` sur `["cot", "meta_action", "traj_future"]` (D-008) plus le `<|im_end|>` assistant. Vérifier à froid que le nombre de `<|traj_history|>` = **45** et `<|traj_future|>` = 128 par ligne (2S le vérifie au `forward`, autant l'attraper au collate).

### Étape 3 — Étape 1 LoRA sur le VLM (4×H100)

- Modèle : `Alpamayo2Super.from_pretrained(id, dtype=bf16)` (`enable_expert: true` dans le checkpoint) ; geler tout ; PEFT LoRA sur les 64 couches du LM (`Qwen3VLTextModel`) : `q_proj`, `k_proj`, `v_proj`, `o_proj` + `gate_proj`, `up_proj`, `down_proj`, r = 16–32, α = 2r, dropout 0,05 — restreindre par expression régulière au sous-module `language_model` : l'encodeur visuel (`Qwen3VLVisionModel`, 27 blocs) a d'autres noms de projections (`attn.qkv`, `attn.proj`, `mlp.linear_fc1/2`) et reste gelé, comme le projecteur, les embeddings (155 776 × 5120), `lm_head` et l'expert. Le vocabulaire est déjà agrandi dans le checkpoint : pas de `resize_token_embeddings`.
- Perte : `Alpamayo2Super.forward` tel quel (deux CE, `loss_weights`, `logit_cap`). Un test 10 exemples doit voir les deux termes descendre ; surveiller séparément `future_traj` et `others` (les exposer dans l'output, un patch de 5 lignes à proposer en amont).
- Mémoire : 32 B en bf16 = 64 GB de poids ; sur 4 GPU il faut FSDP (`full_shard`, LoRA hors sharding) ou ZeRO-3 pour les poids gelés, plus grad-ckpt. Repli : base en 8 bits (bitsandbytes) → ~34 GB, sur 2 GPU. Longueur de séquence estimée : 24 images × (196608 / (16·16·4) ≈ 192 jetons, `patch_size 16`, `spatial_merge_size 2`) ≈ 4,6 k jetons image + ~250 jetons texte/trajectoire — c'est la raison du grad-ckpt.
- Hyperparamètres de départ : LR 1e-4 (LoRA), warm-up 100, cosinus, batch 1 × accumulation 8 sur 4 GPU (32 séquences/pas comme la recette), 3 époques sur le manifeste W3.
- Sortie : adaptateurs LoRA + config ; fusion (`merge_and_unload`) pour l'inférence et l'étape 4.

### Étape 4 — Étape 2 sur l'expert (4×H100)

- Charger le checkpoint fusionné de l'étape 3, `vlm.requires_grad_(False)` (déjà fait par le constructeur si `cotrain_expert_vlm` est faux), entraîner `model.expert` (2 B : plein fine-tuning tient sur un GPU ; LoRA sur `expert.expert.layers.*` si l'on veut limiter la dérive).
- Passe : `vlm_outputs = model.vlm(**tokenized_data, use_cache=True)` sous `no_grad`, cache tronqué au dernier `<|traj_future_start|>` + 1 (comme `sft_alpamayo_r1.py`), puis `model.expert(traj_data, vlm_outputs, cache_attention_mask=attention_mask tronqué)` — `ExpertModel.forward` calcule déjà les `position_ids` MRoPE, le bruit et la perte MSE. Le `traj_data` contient la **trajectoire cible** (amortie sous dégradation), donc l'expert apprend la politique de prudence en actions (a, κ).
- Vérifier que `traj_to_action` accepte une trajectoire amortie qui s'arrête (`_LOW_SPEED_CURVATURE_THRESHOLD_MPS = 0.6`) : c'est l'un des points où D-007 peut casser silencieusement (courbure indéfinie à vitesse nulle).

### Étape 5 — Smoke test 10 exemples et non-régression

- 10 exemples du manifeste (4 propres, 6 dégradés), 300 pas : perte étape 1 → ~0, perte étape 2 en baisse monotone.
- Régénérer avec `sample_trajectories_from_data` (K = 5, graines fixes) sur les mêmes clips : les textes propres doivent reproduire le CoC de base (BLEU/exact), les textes dégradés doivent scorer ≥ 0,7 en `afh.degradation.uncertainty_score`, les trajectoires dégradées doivent ralentir (`afh.eval_uncertainty`).
- Non-régression : `afh.runner` axes 1–3 et `minADE` sur les clips propres de `fixtures/records_a2.json` avant/après (critère bloquant, principe 3).

### Étape 6 — PR amont

Structure `recipes/alpamayo2_super_sft/` calquée sur `alpamayo1_5_sft` (README, SKILL, `configs/{sft_base, sft_stage1_lora, sft_stage2_expert}.yaml`, `models/`, `train_hf.py`), chat template `r2` enregistré dans `alpamayo.chat_template._TEMPLATES`, tests statiques ajoutés à `tests/test_recipe_static_contracts.py`. Ne rien dupliquer d'`alpamayo2_super` : dépendance git comme `alpamayo_r1`.

---

## 4. Inconnues qui ne se lèveront que sur GPU (ou avec l'accès HF)
Douze inconnues au départ ; état au 2026-09-30, après l'ajout des configs (`docs/configs/`, commit `804fd42`) et D-008/D-010.

| # | Inconnue | État | Source de la levée / ce qui reste |
|---|---|---|---|
| 1 | Jetons du CoC à l'entraînement de 2S (avec ou sans `<|cot_start|>`) | **Levée** (D-008) | `SPECIAL_TOKENS_KEYS` de `models/utils.py` et `build_alpamayo2_super_tokenizer` de `config.py` (NVlabs/alpamayo2) : cible balisée `<|cot_start|>cot<|cot_end|><|meta_action_start|>…<|meta_action_end|><|traj_future_start|>…` ; masque de labels par composante inchangé. Archiver un `raw_outputs` sur le pod pour confirmer l'ordre d'émission |
| 2 | `config.json` de 2S | **Levée** | `docs/configs/alpamayo2-super/config.json` : `Qwen3VLForConditionalGeneration`, LM 64 × 5120 (64 têtes / 8 KV), vision 27 couches deepstack `[8,16,24]`, expert 64 × 1536, vocabulaire 155 776, `loss_weights` égaux, `enable_expert: true`, `cotrain_expert_vlm: false`, pas de `logit_cap`, futur 128 = 64 × (a, κ) 3 000 bins `id0 152669`, histoire 45 = 15 × 3 1 000 bins `id0 151669` (`pad_origin_at_beginning: false`) |
| 3 | `config.json` de 1.5 | **Levée** | `docs/configs/alpamayo-1.5-10b/config.json` : `traj_vocab_size 4000`, `traj_token_start_idx 151669`, futur d'abord puis histoire, 48 / 128 jetons, `add_special_tokens: true`, `attn_implementation flash_attention_2`, expert 2048 / 8256 / 16 têtes |
| 4 | Vocabulaire des méta-actions (`Longitudinal/Lateral/Lane`) et fréquence sans prompt dédié | **GPU** | une génération `meta_action` sur le pod ; le code n'énumère pas les valeurs |
| 5 | `Alpamayo2Super.forward` / `ExpertModel.forward` avec gradients sous `transformers 4.57.x` (`DynamicCache.crop` après un `forward`, `rope_deltas`, masque 4D avec `sdpa`, `is_causal=False`) | **GPU** | smoke test de l'étape 3 ; note : le checkpoint 2S a été sauvé avec `transformers 4.57.6`, la recette épingle 4.57.1 |
| 6 | Empreinte mémoire et débit de l'étape 3 (activations ~5 k jetons, cache KV 6 caméras) | **GPU** | décide FSDP bf16 vs base 8 bits (D-009, proposée) |
| 7 | Noms des modules PEFT | **Levée** | `vlm_config.architectures = ["Qwen3VLForConditionalGeneration"]` : LM `q_proj/k_proj/v_proj/o_proj` + `gate_proj/up_proj/down_proj` sous `language_model`, à exclure l'encodeur visuel dont les projections portent d'autres noms |
| 8 | Longueur de séquence par profil et taille des `pixel_values` | **GPU** (estimable) | `preprocessor_config.json` donne `patch_size 16`, `merge_size 2`, `min/max_pixels` ; ≈ 192 jetons par image, à mesurer sur un lot réel |
| 9 | flash-attn sur le pod | **GPU** | compilation à l'installation ; repli `sdpa` |
| 10 | `traj_to_action` sur les trajectoires amorties jusqu'à l'arrêt | **GPU** (ou CPU avec `alpamayo2_super` installé) | seuil `_LOW_SPEED_CURVATURE_THRESHOLD_MPS = 0.6` |
| 11 | Reproductibilité des rollouts avec le prefill partagé (graines) | **GPU** | deux rollouts à graine fixe sur le pod |
| 12 | Licence de redistribution des poids dérivés | **Ousseynou** | fiche modèle HF (`OpenMDW-1.1` d'après les README) ; risque n° 2 de la roadmap, à trancher avant tout partage de checkpoint |

## 5. Ce que cette note décide et ce qu'elle laisse à Ousseynou
Décidé ici (dans le périmètre de D-004) : LoRA sur le LM du VLM d'abord, expert ensuite, cibles générées par `afh`, réutilisation des forward d'entraînement de la release 2S, aucun code NVlabs copié (dépendances git). Dans `DECISIONS.md` :

- **D-008 (levée)** : cible CoC de 2S balisée `<|cot_start|>…<|cot_end|><|meta_action_start|>…<|meta_action_end|><|traj_future_start|>…` (sources `models/utils.py` et `helper.py` de NVlabs/alpamayo2) ; l'étape 2 s'écrit sur cette base.
- **D-010 (adoptée)** : `afh` raisonne en identifiants caméra (`afh/cameras.py`, alpamayo-faithfulness PR #7) ; `runners/probe_blackout_a15.py` passe `camera_indices` au harnais.
- **D-009 (proposée)** : FSDP bf16 vs base quantifiée pour l'étape 3 — attendre l'inconnue 6.

