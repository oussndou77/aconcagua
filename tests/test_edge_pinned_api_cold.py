#!/usr/bin/env python3
"""
Cold check of every API the Edge probes call, against the pinned sources — NO GPU, NO torch.

The probes of docs/gpu_session_02.md run in the recipe environment, which cannot be built
here. This test reads the two pinned checkouts as text (scripts/fetch_edge_sources.sh puts
them under .deps/) and verifies, with `ast`, that each function exists with the arguments
the probes pass, that the constants the probes copy are the pinned ones, and that the
commands of the session document use options the pinned scripts and configs define.

It skips when the checkouts are absent. The one file it executes is the recipe's
`src/alpamayo/data/pai_metadata.py`, which declares itself dependency-free: the navigation
counterfactuals of block 3 go through the recipe's own validator.

Not covered here (no source in these two repositories): `physical_ai_av` (compared between
0.2.0 and 0.2.2 in docs/gpu_session_02.md, block 0 notes) and `transformers` 5.17.0.
"""

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys

import pytest

import runners.probe_edge_common as ec
import runners.probe_train_memory_edge as pmem

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E = "src/alpamayo2_edge"
R = "recipes/alpamayo2edge_sft"


def _need(path):
    if path is None:
        pytest.skip("pinned checkout absent: run scripts/fetch_edge_sources.sh")
    return path


def _read(root, rel):
    with open(os.path.join(root, rel)) as fh:
        return fh.read()


def _func(root, rel, name, cls=None):
    tree = ast.parse(_read(root, rel))
    scope = tree.body
    if cls is not None:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    return next(n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def _params(fn):
    a = fn.args
    return {"positional": [x.arg for x in a.posonlyargs + a.args], "kwonly": [x.arg for x in a.kwonlyargs],
            "all": [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs], "kwargs": a.kwarg is not None}


def _literal(root, rel, name):
    for node in ast.parse(_read(root, rel)).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {rel}")


def _head(repo):
    return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


# --------------------------------------------------------------------------- pins

def test_checkouts_are_at_the_pinned_commits(edge_src, recipes_src):
    assert _head(_need(edge_src)) == ec.EDGE_REPO_COMMIT
    assert _head(_need(recipes_src)) == ec.RECIPES_COMMIT
    # the recipe pins the very inference commit the probes assume
    assert f"alpamayo2-edge.git@{ec.EDGE_REPO_COMMIT}" in _read(recipes_src, f"{R}/pyproject.toml")
    assert 'name = "alpamayo2edge-sft"' in _read(recipes_src, f"{R}/pyproject.toml")
    assert 'name = "alpamayo2-edge"' in _read(edge_src, "pyproject.toml")
    assert ec.PAI_REVISION in _read(recipes_src, f"{R}/README.md")
    assert ec.HF_REVISION in _read(edge_src, "README.md")


# --------------------------------------------------------------------------- inference package

def test_inference_entry_points_take_the_arguments_the_probes_pass(edge_src):
    edge = _need(edge_src)
    load_model = _params(_func(edge, f"{E}/checkpoint.py", "load_model"))
    assert load_model["positional"] == ["model_id"] and load_model["kwargs"]
    assert {"device", "dtype", "attn_implementation"} <= set(load_model["kwonly"])
    assert "revision" in _params(_func(edge, f"{E}/checkpoint.py", "resolve_checkpoint"))["kwonly"]

    assert _params(_func(edge, f"{E}/helper.py", "prepare_model_inputs"))["positional"] == \
        ["data", "model_config", "tokenizer"]
    assert _params(_func(edge, f"{E}/helper.py", "to_device"))["positional"][:2] == ["data", "device"]

    loader = _params(_func(edge, f"{E}/load_physical_aiavdataset.py", "load_physical_aiavdataset"))
    assert loader["positional"] == ["clip_id", "t0_us"] and {"nav_text", "avdi"} <= set(loader["kwonly"])

    sampler = _params(_func(edge, f"{E}/models/alpamayo2_edge.py", "sample_trajectories_from_data",
                            cls="Alpamayo2Edge"))
    assert sampler["positional"][:2] == ["self", "data"]
    assert {"top_p", "top_k", "temperature", "num_traj_samples", "num_traj_sets",
            "max_generation_length", "diffusion_kwargs", "return_extra"} <= set(sampler["all"])

    assert _params(_func(edge, f"{E}/models/utils.py", "fuse_traj_tokens"))["positional"] == \
        ["history_traj_tokenizer", "future_traj_tokenizer", "input_ids", "traj_data", "traj_ids"]
    flow = _params(_func(edge, f"{E}/diffusion/flow_matching.py", "sample", cls="FlowMatching"))
    assert {"inference_step", "generator"} <= set(flow["all"])


def test_recorder_and_error_handling_match_the_release_code(edge_src):
    edge = _need(edge_src)
    model = _read(edge, f"{E}/models/alpamayo2_edge.py")
    # the recorder reads the prompt from the `input_ids` keyword of vlm.generate
    assert re.search(r"self\.vlm\.generate\(\s*input_ids=", model)
    assert 'extra["generated_ids"]' in model and "return pred_xyz, pred_rot, logprob, extra" in model
    # a generation without anchor raises ValueError with this prefix (run_edge_inference)
    utils = _read(edge, f"{E}/models/expert_utils.py")
    assert utils.count("No <traj_future_start> token found") == 2
    # keys the probes read from `extra`
    extraction = _read(edge, f"{E}/models/token_utils.py")
    for key in ('"raw_outputs"', '"cot"', '"meta_action"', '"answer"'):
        assert key in extraction
    # the unsteered run calls vlm.generate the way the release's VQA path does
    vqa = _read(edge, f"{E}/text_tasks.py")
    assert "**tokenized, generation_config=generation" in vqa
    # the release's own seeding, mirrored by run_edge_inference
    inference = _read(edge, f"{E}/inference.py")
    assert "torch.manual_seed(settings[\"seed\"])" in inference
    assert f"seed={ec.DEFAULT_SEED}," in inference
    assert f"max_generation_length={ec.DEFAULT_MAX_GENERATION_LENGTH}," in inference
    assert f'diffusion_kwargs={{"inference_step": {ec.DEFAULT_DIFFUSION_STEPS}}}' in inference


def test_constants_copied_from_the_inference_package(edge_src):
    edge = _need(edge_src)
    assert _literal(edge, f"{E}/chat_template.py", "CLOSED_THINKING") == ec.CLOSED_THINKING
    assert set(_literal(edge, f"{E}/chat_template.py", "CAMERA_DISPLAY_NAMES")) == set(ec.EDGE_CAMERA_NAMES)
    assert _literal(edge, f"{E}/sample.py", "CAMERA_FRAME_COUNTS") == ec.EDGE_CAMERA_FRAME_COUNTS
    cameras = _literal(edge, f"{E}/load_physical_aiavdataset.py", "_CAMERAS")
    assert {i: name.lower() for i, name in cameras} == ec.EDGE_CAMERA_NAMES
    sample = _read(edge, f"{E}/sample.py")
    assert "(163840, 196608) if camera in (1, 6) else (81920, 98304)" in sample
    # the loader refuses an event before 2.0 s, the lower bound the probes document
    assert "2_000_000 <= t0_us" in _read(edge, f"{E}/load_physical_aiavdataset.py")


# --------------------------------------------------------------------------- recipe

def test_recipe_classes_take_the_arguments_the_memory_probe_passes(recipes_src):
    rec = _need(recipes_src)
    collator = _params(_func(rec, f"{R}/processor.py", "__init__", cls="Alpamayo2EdgeCollator"))
    assert collator["positional"] == ["self", "model_config", "max_length", "pad_to_fixed_length"]
    preprocess = _params(_func(rec, f"{R}/processor.py", "preprocess", cls="Alpamayo2EdgeProcessor"))
    assert preprocess["positional"] == ["self", "data", "components_order", "components_prompt",
                                        "label_components", "generation_mode"]
    processor = _read(rec, f"{R}/processor.py")
    assert "self.data_processor = Alpamayo2EdgeProcessor(model_config)" in processor
    assert "self.processor = get_processor(tokenizer, model_config)" in processor
    for key in ('"tokenized_data": tokenized_data', '"traj_data": traj_data', '"labels_mask": labels_mask'):
        assert key in processor
    # the collator's own length error is classified too_long by probe_common.classify_error
    assert "exceeds max_length=" in processor

    forward = _params(_func(rec, f"{R}/models/sft.py", "forward", cls="Alpamayo2EdgeSFTMixin"))
    assert forward["positional"][:4] == ["self", "tokenized_data", "traj_data", "labels_mask"]
    sft = _read(rec, f"{R}/models/sft.py")
    for field in ("loss", "future_traj_ce", "others_ce", "expert_diffusion", "vlm_ce"):
        assert re.search(rf"^\s+{field}: torch\.FloatTensor \| None = None$", sft, re.M), field
    assert "Joint training requires trainer.gradient_checkpointing=false" in sft
    assert "class TrainableAlpamayo2Edge(Alpamayo2EdgeSFTMixin, Alpamayo2Edge)" in \
        _read(rec, f"{R}/models/trainable_alpamayo2_edge.py")
    # packaged with its configs, which the probe reads through importlib.resources
    assert '"alpamayo2edge_sft.configs"' in _read(rec, f"{R}/pyproject.toml")
    assert os.path.exists(os.path.join(rec, R, "configs/__init__.py"))


def _yaml_value(text, key):
    m = re.search(rf"^\s*{re.escape(key)}:\s*(.+?)\s*(?:#.*)?$", text, re.M)
    assert m, key
    return m.group(1)


def test_recipe_configs_hold_the_values_the_memory_probe_reproduces(recipes_src):
    rec = _need(recipes_src)
    release = _read(rec, f"{R}/configs/model/release.yaml")
    assert _yaml_value(release, "_target_") == \
        "alpamayo2edge_sft.models.trainable_alpamayo2_edge.TrainableAlpamayo2Edge.from_pretrained"
    assert _yaml_value(release, "training_stage") == "joint"
    assert _yaml_value(release, "dtype") == "float32" and _yaml_value(release, "attn_implementation") == "sdpa"
    assert "pretrained_model_name_or_path:" in release

    nav_cot = _read(rec, f"{R}/configs/preprocess/nav_cot.yaml")
    for key, expected in pmem.NATIVE_PREPROCESS.items():
        assert _yaml_value(nav_cot, key) == "[" + ", ".join(expected) + "]", key

    base = _read(rec, f"{R}/configs/sft_base.yaml")
    assert int(_yaml_value(base, "max_length")) == pmem.MAX_LENGTH
    assert _yaml_value(base, "pad_to_fixed_length") == "false"
    assert float(_yaml_value(base, "learning_rate")) == pmem.OPTIMIZER["lr"]
    assert (float(_yaml_value(base, "adam_beta1")), float(_yaml_value(base, "adam_beta2"))) == pmem.OPTIMIZER["betas"]
    assert float(_yaml_value(base, "adam_epsilon")) == pmem.OPTIMIZER["eps"]
    assert float(_yaml_value(base, "weight_decay")) == pmem.OPTIMIZER["weight_decay"]
    assert float(_yaml_value(base, "max_grad_norm")) == pmem.MAX_GRAD_NORM
    assert _yaml_value(base, "optim") == "adamw_torch_fused" and _yaml_value(base, "bf16") == "true"
    assert _yaml_value(base, "gradient_checkpointing") == "false"

    debug = _read(rec, f"{R}/configs/sft_joint_single_node_debug.yaml")
    assert "- camera_front_wide_120fov" in debug and "frame_offsets: [0.0]" in debug   # last fallback config


def test_demo_asset_is_the_one_the_tests_and_the_session_use(recipes_src):
    rec = _need(recipes_src)
    asset = json.loads(_read(rec, ec.EVENTS_ASSET))
    copy = json.load(open(os.path.join(ROOT, "tests/data/edge_demo_events.json")))
    assert [{k: row[k] for k in ("clip_id", "t0_relative", "nav_text")} for row in asset] == copy
    events = ec.load_events(os.path.join(rec, ec.EVENTS_ASSET))
    assert len(events) == 19 and len({e.clip_id for e in events}) == 18
    assert all(e.cot for e in events)                       # block 4 takes its reasoning target here
    assert all(ec.MIN_T0_US <= e.t0_us <= ec.MAX_T0_US for e in events)


def test_navigation_counterfactuals_pass_the_recipe_validator(recipes_src):
    rec = _need(recipes_src)
    path = os.path.join(rec, "src/alpamayo/data/pai_metadata.py")
    spec = importlib.util.spec_from_file_location("pinned_pai_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)                         # dependency-free by its own docstring
    assert (module.EDGE_MIN_T0_US, module.EDGE_MAX_T0_US) == (ec.MIN_T0_US, ec.MAX_T0_US)
    checked = 0
    for event in ec.load_events(os.path.join(rec, ec.EVENTS_ASSET)):
        assert module.validate_navigation(event.nav_text) == event.nav_text
        cfs = ec.nav_counterfactuals(event.nav_text)
        assert [c["label"] for c in cfs] == ["mirror", "straight"]
        for cf in cfs:
            assert module.validate_navigation(cf["nav_text"]) == cf["nav_text"]
            checked += 1
    assert checked == 38
    with pytest.raises(ValueError):
        module.validate_navigation("Go somewhere nice")     # the validator does reject free text
    # block 5 downloads one chunk: the range of the session document is half-open
    assert module.parse_chunk_ids("1134-1135") == [1134]
    # the live wrapper imports the same function from the installed package
    assert "from alpamayo.data.pai_metadata import validate_navigation" in \
        open(os.path.join(ROOT, "runners/probe_edge_common.py")).read()


# --------------------------------------------------------------------------- session document

def test_session_02_recipe_commands_exist_at_the_pinned_commit(recipes_src):
    rec = _need(recipes_src)
    doc = open(os.path.join(ROOT, "docs/gpu_session_02.md")).read()
    # scripts/download_pai.py options
    (download,) = re.findall(r"scripts/download_pai\.py((?:[^\n]*\\\n)*[^\n]*)", doc)
    script = _read(rec, "scripts/download_pai.py")
    for flag in set(re.findall(r"(?<![\w-])(--[a-z][a-z-]+)", download)):
        assert f'"{flag}"' in script, flag
    for camera in re.findall(r"camera_[a-z_0-9]+fov", download):
        assert camera in ec.EDGE_CAMERA_NAMES.values(), camera
    assert len(set(re.findall(r"camera_[a-z_0-9]+fov", download))) == 6
    # evaluate_hf: module, config name and every override key
    (evaluate,) = re.findall(r"alpamayo2edge_sft\.evaluate_hf((?:[^\n]*\\\n)*[^\n]*)", doc)
    assert os.path.exists(os.path.join(rec, R, "evaluate_hf.py"))
    (config_name,) = re.findall(r"--config-name (\w+)", evaluate)
    eval_cfg = _read(rec, f"{R}/configs/{config_name}.yaml")
    assert "mode: trajectory" in eval_cfg and "deepspeed: null" in eval_cfg
    base = _read(rec, f"{R}/configs/sft_base.yaml")
    for override in re.findall(r"\s([a-z_.]+)=", evaluate):
        leaf = override.rsplit(".", 1)[-1]
        assert re.search(rf"^\s*{leaf}:", base, re.M), override
    # the environment commands are the recipe README's
    readme = _read(rec, f"{R}/README.md")
    assert "uv sync --active --locked" in readme and "uv sync --active --locked" in doc
    assert "hf download nvidia/Alpamayo2-Edge" in readme
    assert "UV_HTTP_TIMEOUT=900" in doc


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
