# Revue de la recette officielle Alpamayo 2 Super (alpamayo-recipes #39) et d'alpamayo2 #18

*Lecture du code, 2026-10-11. Aucune exécution. Sert de source à D-014 et au tableau des inconnues de `finetune_design.md` §4.*

| Dépôt | Commit lu | Objet |
|---|---|---|
| `NVlabs/alpamayo-recipes` | `3f6504e` (main, 2026-10-06), recette ajoutée par `c29dc61` (#39) | `recipes/alpamayo2_sft/` |
| `NVlabs/alpamayo2` | `5977692` (main, 2026-10-06) = #18 | extraction du texte généré |
| `NVlabs/alpamayo2` épinglé par la recette | `9596749` (2026-08-03), `recipes/alpamayo2_sft/uv.lock:110` | antérieur à #18 et à `5e7975f` |

Entre `9596749` et `5977692`, `helper.py` et `chat_template/conversation.py` sont identiques (`git diff --stat 9596749 5977692` ne les liste pas) : le prompt d'inférence et le template d'entraînement sont les mêmes dans l'environnement de la recette et sur `main`.

Abréviation : `R/` = `recipes/alpamayo2_sft/`, `A2/` = `src/alpamayo2_super/` d'`NVlabs/alpamayo2`.

---

## 1. Format des cibles

**Le CoC n'est pas une cible de la recette.** Les deux configs de prétraitement livrées sont trajectoire seule :

```yaml
# R/configs/preprocess/trajectory.yaml:2-4 (entraînement) et trajectory_generation.yaml:2-4 (validation, generation_mode: true)
components_order: [image, traj_history, prompt, traj_future]
components_prompt: [traj_future]
label_components: [traj_future]
```

`R/README.md:340` : « The provided PAI subset has no reasoning parquet, so CoC/meta-action SFT is not enabled. »

**Encadrement par `<|cot_start|>…<|cot_end|>`.** La recette ne l'exerce pas, mais son chemin le produirait si l'on ajoutait `cot` à `components_order` :

- `R/processor.py:61-78` appelle le `build_conversation` de la release puis `apply_chat_template(add_generation_prompt=False, continue_final_message=generation_mode)`.
- `A2/chat_template/conversation.py:390-393` appelle `construct_cot` (`:183-210`), qui passe par `get_component_str(SPECIAL_TOKENS["cot_start"], SPECIAL_TOKENS["cot_end"], content_str=data["cot"])` (`:26-48`) : en mode entraînement, le texte est `<|cot_start|>{cot}<|cot_end|>`. Même chose pour `meta_action` (`:394-397`, `construct_meta_action` : `<|meta_action_start|>{meta_action_strings}<|meta_action_end|>`).
- `R/dataset.py:88-89` remplit `sample["cot"]` depuis le parquet de raisonnement (`src/alpamayo/data/pai_utils.py:234-243`) quand `reasoning_metadata` est donné. Il n'existe aucun champ `meta_action_strings` dans ce chemin de données.

**Le masque de labels couvre toute la réponse de l'assistant, pas une liste de composantes.** `R/processor.py:147-158` construit `labels_mask` avec `get_assistant_mask` (`R/label_mask.py:11-49` : de `<|im_start|>assistant\n` + 3 jusqu'à `<|im_end|>` inclus). `label_components` n'est utilisé qu'au contrôle de cohérence du lot (`R/processor.py:112-118`), jamais pour le masque. Avec la config livrée, la réponse de l'assistant est `<|traj_future_start|><i…>×128<|traj_future_end|><|im_end|>`.

**Méta-actions** : non supervisées (pas dans `components_order`, pas de données, README `:340`).

**Ce que dit #18 du format généré par la release** (indice, pas observation) : les tests ajoutés par NVIDIA modélisent une génération trajectoire **sans balises** :

```python
# NVlabs/alpamayo2 tests/test_text_extraction.py:170-181
meta_action = "Longitudinal: brake\nLateral: straight\nLane: keep"
text = f"A red light is ahead.\n{meta_action}{TRAJECTORY_START}<i42>"
...
assert result["answer"] == result["cot"] == ["A red light is ahead."]
assert result["meta_action"] == [meta_action]
```

et les docstrings parlent de génération « no-special » (`A2/models/token_utils.py:126`, `:141`). Le prompt d'inférence de la release ne pose aucune composante assistant : `A2/helper.py:37-38` (`components_order=["image","traj_history","prompt"]`, `components_prompt=["cot","traj_future"]`), `:45` (tour assistant vide retiré), `:80` (`add_generation_prompt=True`).

## 2. Ce qui est supervisé

**Étape 1 (`training_stage: vlm`, `R/configs/sft_stage1.yaml:5-6`)** : tout le VLM est entraînable, l'expert est gelé (`R/models/trainable_alpamayo2_super.py:68-70`). Le `forward` (`:150-186`) reprend la perte de la release en deux termes :

- `future_traj_ce` : entropie croisée sur les jetons `<i1000>…<i3999>`, `<|traj_future_start|>`, `<|traj_future_end|>` (`:168-177`), pondérée par `loss_weights["future_traj"]` ;
- `others_ce` : entropie croisée sur le reste du masque (`:178-180`), pondérée par `loss_weights["others"]`. Avec la config livrée, c'est le seul `<|im_end|>` de l'assistant.
- `loss = future_traj_ce + others_ce` (`:182`) ; poids 1,0 / 1,0 dans le checkpoint (`docs/configs/alpamayo2-super/config.json`). `logit_cap` appliqué s'il existe (`:164-166`), absent du checkpoint. Taux d'apprentissage de la vision ×0,1 (`R/configs/sft_base.yaml:71-72`, README `:173`).

**Étape 2 (`training_stage: expert`, `R/configs/sft_stage2.yaml:5-7`)** : VLM gelé, expert seul entraînable (`:68-70`). `_forward_expert` (`:188-…`) passe le VLM sous `torch.no_grad()` sauf si `cotrain_expert_vlm` (`:212`), tronque le cache KV après `<|traj_future_start|>` (`:221`), détache clés et valeurs (`:222-225`), et calcule la perte de flow matching de la release en float32 (`:308`, `compute_loss_from_pred`), pondérée par `loss_weights["future_traj"]` (`:235`). La CE du VLM n'est ajoutée que si `cotrain_expert_vlm` (`:236-241`) ; le checkpoint a `cotrain_expert_vlm: false`, donc **l'étape 2 supervise l'expert seul**.

## 3. Matériel

| Élément | Valeur | Source |
|---|---|---|
| Étape 1 | 4 nœuds × 8 H100 80 GB (32 GPU), plein fine-tuning | `R/README.md:46`, `:353` |
| Étape 2 | ≥ 2 nœuds × 8 H100 | `R/README.md:47` |
| Évaluation trajectoire | 1 H100 80 GB, un seul processus (`generate()` incompatible avec ZeRO-3) | `R/README.md:48`, `R/evaluate_hf.py:72-76` |
| Sharding | DeepSpeed ZeRO-3 (pas FSDP) | `R/configs/sft_base.yaml:87`, `R/configs/deepspeed/zero3.json:9` |
| Gradient checkpointing | oui en étape 1 (non réentrant), non en étape 2 | `R/configs/sft_base.yaml:84-86`, `R/configs/sft_stage2.yaml:17` |
| Lot | 1 par GPU, accumulation 16 (étape 1) / 32 (étape 2), longueur paddée fixe 6 144 | `R/configs/sft_base.yaml:46-47`, `:67-69`, `sft_stage2.yaml:14` |
| Optimiseur | AdamW fusionné, LR 5e-5 (étape 1) / 4e-4 (étape 2), warm-up 4 000, cosinus vers 1 % | `sft_base.yaml:70-82`, `sft_stage2.yaml:15` |
| Smoke | 1 nœud × 8 H100, **1 caméra, 1 image**, état d'optimiseur déchargé sur CPU | `R/configs/sft_stage1_single_node_debug.yaml:5-7`, `:23` ; README `:274` |
| Attention | aucune `attn_implementation` passée : défaut `transformers` (sdpa) ; flash-attn installé (`2.8.3.post1`) mais non sélectionné | `grep` sans résultat dans `R/`, `R/uv.lock` |
| Versions | torch 2.8.0, transformers 4.57.1, deepspeed 0.19.1, Python 3.12 | `R/pyproject.toml`, `R/uv.lock` |

Signaux de référence publiés : deux pas d'étape 1 (perte 1,8095 → 1,7255), `eval_loss` 1,6998 sur le bloc 99, `eval_min_ade` 0,8288 m sur 100 clips à 6 échantillons (`R/README.md:318-324`).

## 4. LoRA / PEFT

**Absent.** Aucune occurrence de `peft` ou `lora` dans `R/` (code, configs, README), `peft` absent de `R/uv.lock`. La seule granularité est `set_training_stage` (tout le VLM ou tout l'expert, `:60-71`) plus le multiplicateur de LR par préfixe (`R/trainer.py:23-61`).

## 5. Format de données et licence

- Format : arborescence PAI locale lue par `PhysicalAIAVDatasetLocalInterface` (`features.csv`, `clip_index.parquet`, `metadata/feature_presence.parquet`, zips par chunk), téléchargée par `scripts/download_pai.py --chunk-ids 0-100` avec les 7 caméras, l'égomotion et la calibration (`R/README.md`, section « Prepare PAI Chunks 0–99 »).
- Échantillon : `load_physical_aiavdataset` de la release, 7 caméras × 4 images, 16 poses d'histoire, 64 futures (`R/dataset.py:74-84`), `t0` fixé à 5,1 s pour tous les clips (`use_default_keyframe: true`, `sft_base.yaml:30`), `sample["cot"]` si parquet de raisonnement (`:88-89`).
- Découpage : chunks 0–98 en entraînement (9 881 clips), chunk 99 tenu à part (100 clips) (`sft_base.yaml:29`, `:39`, README `:141`).
- Licence : la recette ne discute pas la licence du dataset. Le README racine indique « NVIDIA AV Dataset License » (`README.md:216`) et renvoie aux licences respectives (`:297`) ; le README de la recette demande seulement d'accepter les licences du modèle et du dataset (`R/README.md:349`). Le texte de la licence est sur Hugging Face, inaccessible depuis cette session : **la compatibilité avec un usage interne n'est pas vérifiable d'ici** (risque n° 2 de la roadmap, à trancher par Ousseynou).

## 6. alpamayo2 #18

- `A2/models/token_utils.py:135-188` : `extract_text_tokens` prend un argument `task` (défaut `"trajectory"`). Pour `trajectory` et `meta_action`, rien ne change : coupe à `<|traj_future_start|>` puis séparation CoC / méta-actions sur la première tête `Longitudinal:`, `Lateral:` ou `Lane:` (`:122-132`). Pour `vqa` et `auto_labeling`, le texte n'est plus coupé à un `<|traj_future_start|>` littéral ni scindé aux têtes d'axe (`:155-162`, `:174`).
- Appelants : `A2/models/alpamayo2_super.py:447` (`return_extra` de `sample_trajectories_from_data`, `task="trajectory"`) et `A2/text_tasks.py:428` (`task=task`).
- **Inférence seulement.** Le chemin d'entraînement de la recette n'appelle pas `extract_text_tokens`, et l'environnement de la recette épingle `9596749`, antérieur à #18.

## 7. Points qui contredisent ou rendent caducs ROADMAP et DECISIONS (signalés, non tranchés)

1. **ROADMAP W1 et D-003** : « No official recipe exists (verified 2026-09-30) » est faux depuis #39 (commit du 2026-09-30, fusionné ensuite). La prémisse de D-003 (« no 2 Super training recipe exists ») tombe.
2. **ROADMAP W0 et §6 (budget)** : la recette est un plein fine-tuning sur 32 H100 (étape 1) et 16+ (étape 2) ; la roadmap prévoit 4–8 H100 et « LoRA on 2 Super: 4×H100 ≈ $12/h ». La recette n'offre pas de LoRA : l'adopter telle quelle sort du budget, l'adopter avec LoRA demande une couche PEFT à écrire.
3. **D-008** : (a) le seul indice venu de NVIDIA (#18) va contre l'hypothèse « la release émet `<|cot_start|>` » ; (b) le motif de rejet « un masque sur la réponse entière demanderait un autre masque » ne tient plus : la recette officielle utilise précisément ce masque, et il est compatible avec un template balisé. La question reste ouverte (inconnue 1), mais elle se reformule : **les cibles CoC balisées que produirait le template d'entraînement officiel sont-elles cohérentes avec ce que la release génère ?**
4. **W3 (CoC sous dégradation)** : la recette ne supervise ni le CoC ni les méta-actions, et son évaluation trajectoire conditionne l'expert sur un prompt qui se termine par `<|traj_future_start|>` (`trajectory_generation.yaml`, `generation_mode: true`, `traj_future` dernière composante → `A2/chat_template/conversation.py:357-359`), donc **sans CoC généré**. Le 0,8288 m n'est pas comparable aux mesures `afh`, qui passent par le CoC.
5. **`t0` fixe à 5,1 s** pour tous les clips d'entraînement : cohérent avec nos fixtures (`0ea6fd88`, 5,1 s) mais pas avec un échantillonnage aux événements CoC (`event_t0s`) si l'on ajoute le parquet de raisonnement (`use_default_keyframe: false` exigerait alors que tous les clips aient un événement).
