---
datasets:
- nvidia/PhysicalAI-Autonomous-Vehicles
- nvidia/PhysicalAI-Autonomous-Vehicles-NuRec
pipeline_tag: robotics
license: openmdw-1.1
language:
- en
base_model:
- nvidia/Cosmos3-Edge
tags:
- alpamayo
- autonomous-driving
- vision-language-action
- pytorch
---
# Alpamayo 2 Edge

[**Code**](https://github.com/NVlabs/alpamayo2-edge) | [**Alpamayo Overview**](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/)

## Model Overview:

### Description:

NVIDIA Alpamayo 2 Edge is a 2B-parameter reasoning vision-language-action model that uses adaptive reasoning to generate driving trajectories and causal explanations. Built on the Nemotron-based backbone of NVIDIA Cosmos 3 Edge, it supports fine-tuning and efficient in-vehicle deployment on NVIDIA DRIVE AGX Thor.

Alpamayo 2 Edge was developed by NVIDIA as a part of the broader [Alpamayo Open Platform](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/).

### License/Terms of Use:

**Model weights:** The model weights are released under the [OpenMDW-1.1](https://openmdw.ai/license/1-1/) license.

**Source code:** Apache License 2.0, as provided in the [Alpamayo 2 Edge source repository](https://github.com/NVlabs/alpamayo2-edge).

### Deployment Geography:

Global

### Use Case:

Developers and researchers working on autonomous vehicle systems who need a compact reasoning model for in-vehicle planning and decision-making. Alpamayo 2 Edge generates navigation-conditioned driving trajectories together with causal explanations, and it can answer free-form questions about a driving scene. It serves as a starting point for fine-tuning and for efficient deployment on in-vehicle platforms.

### Release Date:

**Hugging Face:** 11/10/2026 via https://huggingface.co/nvidia/Alpamayo2-Edge

## Model Architecture:

**Architecture Type:** Transformer

**Network Architecture:** Vision-Language-Action (VLA) model with a VLM backbone and a diffusion-based action decoder.

**This model was developed based on:** Cosmos 3 Edge (VLM backbone) with a diffusion-based action decoder.

**Number of model parameters:**

- Backbone: 2.45B parameters (1.96B language model, 0.41B SigLIP2 vision encoder, 0.08B projector)
- Action Expert: 0.2B parameters

**Weight precision:** BF16

## Input:

**Input Type(s):** Image, Text, Egomotion History

**Input Format(s):**

- Image: Red, Green, Blue (RGB)
- Text: String
- Egomotion History: Floating-point values `(x, y, z), R_rot`

**Input Parameters:**

- Image: Two-dimensional (2D), multi-camera, multi-timestep
- Text: One-dimensional (1D)
- Egomotion History: Three-dimensional (3D) translation and nine-dimensional (9D, 3x3) rotation, multi-timestep

**Other Properties Related to Input:** Trajectory prediction uses six cameras and 22 image frames per observation: five frames each for the front wide and front tele cameras (camera IDs `1` and `6`) at −2, −1, −0.5, −0.1, and 0 seconds, and three frames each for the cross left, cross right, rear left, and rear right cameras (camera IDs `0`, `2`, `3`, and `5`) at −1, −0.1, and 0 seconds. It also takes a navigation instruction as text and 20 historical ego poses at fixed 0.1-second steps from −1.9 to 0 seconds, expressed in the ego frame at the query time. Free-form visual question answering uses the same six cameras with four frames per camera. Images require capture timestamps, which the prompt encodes as relative-time labels. The inference code resizes each frame to a per-camera pixel budget and normalizes it with the checkpoint processor's mean and standard deviation. The driving prompt ends with an empty `<think></think>` block, and the model writes its driving reasoning directly after it; the free-form question prompt omits the block. Apply the checkpoint chat template with `enable_thinking=False`, because the template enables thinking by default.

## Output:

**Output Type(s):** Text, Trajectory

**Output Format(s):**

- Text: String (Chain-of-Causation driving reasoning or visual question answers)
- Trajectory: Floating-point values `(x, y, z), R_rot`

**Output Parameters:**

- Text: One-dimensional (1D)
- Trajectory: Three-dimensional (3D) translation and nine-dimensional (9D, 3x3) rotation, multi-timestep

**Other Properties Related to Output:** The trajectory API returns 64 waypoints spanning 0.1 through 6.4 seconds at 0.1-second intervals. Each trajectory contains ego-frame XYZ positions and 3x3 rotation matrices, and each trajectory candidate has its own independently sampled reasoning text. The action expert predicts per-step acceleration and curvature, which a unicycle model integrates in a course-heading action space: positions follow the course (the rear axle's direction of travel), z stays at the query-time height, and the yaw-only rotations estimate the body heading, which differs from the path tangent by the rear-axle sideslip of the training vehicle. Text outputs are variable-length Chain-of-Causation reasoning or answers to free-form questions.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:

**Runtime Engine(s):**

- Python 3.12
- PyTorch (version 2.8.0) and torchvision (version 0.23.0)
- Hugging Face Transformers (version 5.17.0)
- Attention: PyTorch SDPA by default; FlashAttention 2.8.3 is optional and is required for accelerated inference

**Supported Hardware Microarchitecture Compatibility:**

- Tested: NVIDIA H100 80GB HBM3
- Other GPU architectures have not yet been validated.

**Preferred/Supported Operating System(s):** Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):

Alpamayo 2 Edge 2B trained

Can be integrated into in-vehicle autonomous driving software for end-to-end perception, reasoning, and motion planning.

## Training, Testing, and Evaluation Datasets:

## Training Dataset:

**Data Modality:**

- Image
- Text
- Video
- Actions

**Image Training Data Size:** More than 1 Billion Images

**Text Training Data Size:** Less than a Billion Tokens

**Video Training Data Size:** 10,000 to 1 Million Hours

**Data Collection Method by dataset:** Automatic/Sensors

**Labeling Method by dataset:** Hybrid: Automated/Manually-Labelled

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**
The dataset comprises roughly 115,000 hours of multi-camera driving video with corresponding egomotion and trajectory annotations.
It includes roughly 3,700,000 Chain-of-Causation (CoC) reasoning traces that provide decision-grounded, causally linked explanations of driving behaviors.
Content includes machine-generated data from vehicle sensors (cameras, IMUs, and GPS) and synthetic reasoning traces.
It also includes navigation instructions and visual question-answer data.
CoC annotations are in English and use a structured format that links driving decisions to causal factors.
Sensors include RGB cameras, inertial measurement units, and GPS.

### Evaluation Dataset:

**Quantitative Evaluation Benchmarks:**

- Reasoning Evaluation using [LingoQA](https://github.com/wayveai/LingoQA): Lingo-Judge Score of 75.2.
- Closed-Loop Evaluation using [AlpaSim](https://github.com/NVlabs/alpasim) on 875 scenarios from the [PhysicalAI-AV-NuRec Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec), with three rollouts per scenario: 15.15 ± 0.04 km driven per at-fault incident (off-road or at-fault collision; mean ± standard deviation over the three rollout sets).
- Open-Loop Evaluation on 1,392 samples from the [PhysicalAI-AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles): minADE_6 at 5s of 0.52m.

Additional driving visual question answering results:

| Benchmark | Score |
| --- | --- |
| Ego3D, accuracy | 84.94 |
| Ego3D, RMSE in meters (lower is better) | 5.97 |
| VLADBench | 69.94 |
| SURDS | 86.50 |
| WaymoQA, safety-critical | 71.09 |
| WaymoQA, all | 75.77 |
| nuReasoning, choice accuracy | 78.58 |
| nuReasoning, numeric | 48.67 |
| KITScenes, scene | 89.46 |
| KITScenes, motion | 70.00 |
| AUTOPILOT-VQA, Kaggle private-leaderboard score × 100 | 55.71 |

**Data Collection Method by dataset:** Automatic/Sensors

**Labeling Method by dataset:** Hybrid: Automatic/Sensors/Manually-Labelled

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** Open-loop evaluation uses 1,392 multi-camera driving samples from the PhysicalAI-AV dataset, and closed-loop evaluation uses 875 driving scenarios from the PhysicalAI-AV-NuRec dataset in AlpaSim. Reasoning and visual question answering use the public benchmarks listed above. Driving data are collected from RGB cameras and vehicle sensors.

## Inference:

**Acceleration Engine:** Other: PyTorch, Hugging Face Transformers

**Hardware Requirements (GPU Architecture, Model):**

- Tested on: 1 NVIDIA H100 80GB HBM3 GPU
- Measured configuration: one PhysicalAI-AV sample with 22 images (3,236 prompt tokens), BF16, FlashAttention 2, batch size 1, and six trajectory candidates, each with its own reasoning.
- Measured peak: 12.06 GiB GPU process memory, with 8.40 GiB peak PyTorch allocation and 11.38 GiB peak PyTorch reservation.
- We recommend at least 16 GB of GPU memory for this configuration. GPUs other than the H100 above have not been validated, and memory use also varies with the number of trajectory candidates, the input size, and the attention backend.

**Sampling:** Each trajectory candidate samples its own reasoning, followed by ten Euler flow-matching steps by default. The inference CLI generates `--num-traj-sets` × `--num-traj-samples` candidates (both default to 1); `--diffusion-steps` changes the number of flow-matching steps.

For inference scripts, the notebook (including free-form visual question answering), and input preparation, see the [code repository](https://github.com/NVlabs/alpamayo2-edge). The examples load camera images and ego history from the [PhysicalAI-Autonomous-Vehicles dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles), which requires accepting its own access terms.

## Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.
Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.
For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards.
Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Support

📣 **Usage questions**: post on the [Alpamayo NV Developer Forum](https://forums.developer.nvidia.com/c/autonomous-vehicles/alpamayo/766).

🐛 **Code bugs / documentation issues / feature requests**: file a GitHub issue using the appropriate template (Bug report, Documentation request, or Feature request) at https://github.com/NVlabs/alpamayo2-edge/issues/new/choose . The relevant NVIDIA responder is auto-assigned.

🚨 **Security vulnerabilities**: please use [NVIDIA's Vulnerability Disclosure Program](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). Do not file public issues.
