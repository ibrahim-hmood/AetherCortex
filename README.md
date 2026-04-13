<p align="center">
  <img src="aethercortex_logo.png" width="400" alt="AetherCortex Logo">
</p>

# AetherCortex (v4.2)
An explicit, mathematically rigorous neurological simulation representing the human brain structure utilizing **Spiking Neural Networks (SNN)** and **Leaky Integrate-and-Fire (LIF)** logic. 

> [!IMPORTANT]
> **V4.2 Evolutionary Phase**: She is no longer a purely episodic infant. AetherCortex now possesses **Structural Long-Term Memory (LTM)** and a **Focused Curriculum** system, allowing her to anchor permanent knowledge into her synaptic architecture.

## Latest Breakthroughs: Long-Term Memory (v4.1 - v4.2)

### 1. Synaptic Permanence (Myelination)
Unlike standard neural networks that forget as they learn new data, AetherCortex utilizes **Synaptic Permanence**. 
*   **Gold Medal Synapses**: High-reward neural pathways (`Dopamine > 1.5`) trigger physical myelination, increasing a synapse's "Permanence."
*   **Structural Stability**: Permanent synapses are shielded from the nightly **Pruning Cycle (Deep Sleep)**, ensuring her identity and core concepts (like the color RED) survive training gaps.
*   **Apathy-Driven Forgetting**: To avoid learning incorrect information, synapses that lead to sustained low-reward outcomes slowly "Demyelinate," allowing her to unlearn noise.

### 2. Focused Curriculum & Dopamine Gating
The training system now supports a **Focused Reward Pipeline** (`--focus [WORD]`).
*   **Parental Feedback**: When her input context matches the focus word, her brain's dopamine production is doubled.
*   **Directed Growth**: This forces her brain to prioritize the myelination of specific "First Words," accelerating her transition from random babbling to coherent speech.

### 3. Neuro-Monitor v4.2 (Real-Time Maturity Probe)
The browser-based dashboard now provides a direct biological window into her maturing connectome:
*   **Maturity Tags**: Each brain region displays a **Maturity % (🌱 → 🔒)**, indicating how much of its gray matter has been structuraly locked-in.
*   **Gold Auras**: Excited regions glow with a golden hue when their permanent memories are activated, signaling the use of her Long-Term Memory.

Unlike traditional AIs (which rely on highly structured numerical convolutions and algorithmic Transformers), this model structurally replicates the physical wiring of biological cortices. It utilizes time sequences and membrane potentials to organically link abstract textual concepts directly to massive visual and spatial reconstructions.

## How it Works
The model relies entirely on **Predictive Coding** and biological inference. 
1. **The Senses (Sensory Tokenizer)**: Real-world data (images, videos, text files) are physically converted into binary sequences acting as electrical currents. 
2. **The Thalamus**: Inputs are routed into the Thalamic Bottleneck. 
3. **Primary Cortices (Visual & Auditory)**: It processes geometry natively via `ConvLIFCortexLayer` structures. Spatial connections mirror Receptive Fields (where neurons only connect to localized clusters of their adjacent neurons) rather than arbitrary dense math maps.
4. **Prefrontal Cortex (Attention via Recurrence)**: Features are bound through Top-Down Recurrence (`RecurrentLIFCortexLayer`). Natively circulating energy locally allows adjectives ("Green", "Red") to accurately lock their bindings natively to structural components over biological Time (`15 T-Steps`).
5. **Generative Cross-Modal Routes**: Output from the Prefrontal loop projects heavily **backwards** onto the visual motor strip, permitting the engine to actively "hallucinate" videos/pixels that match its internal conceptual semantic mapping!

## Specialized Biological Features

Unlike standard ANN models, this system implements several "hard" biological constraints:

*   **Structural Neuroplasticity (Sleep Phase)**: Every 5 epochs, the system enters a "Deep Sleep" cycle. It permanently **prunes** weak, noisy synapses (Forgetting) and **spawns** new synaptic wires in areas of high inter-modal firing demand (Learning).
*   **Metabolic Austerity (ATP Budgeting)**: Firing spikes is energetically expensive. The model operates under a **Metabolic Cost Penalty**, forcing it to find the sparsest, most efficient path to solve a problem—exactly like a calorie-restricted living brain.
*   **Sequential Phoneme Timing**: The model doesn't process text as a "block." It reads and speaks chronologically, one character per biological time-step. This mirrors the real-time motor control required for human speech.
*   **Competitive Inhibition (GABA Tone)**: Localized suppression fields prevent neural runaway. Neurons must fight to overcome neighboring inhibition, ensuring that only the highest-clarity features are promoted to higher thought.

## What is this Useful For?
* **Organic General Intelligence Research**: Understanding how pure spatial clustering and predictive coding can replicate standard Transformer math via pure voltage integration loops.
* **Neuromorphic Hardware Engineering**: Because code is structured on discrete biological ticks and simple LIF potential summations, this model perfectly translates onto extreme low-power Spiking chips (like IBM TrueNorth or Intel Loihi) executing at massive speeds for almost ~0 Watts.
* **Temporal Motion Engineering**: Exploring chronological "Video-to-Text" bindings via strictly temporal integration.  

## How to Use It

1. **Populate Dataset**
Drop descriptive target media (videos `.mp4` and images `.png`) into the `dataset/` directory.
The physical *filename* acts exactly as the linguistic prompt (e.g. `A cat sleeping on the couch.png`).

2. **Train the SNN**
Execute the biological generation loop:
```bash
python train.py
```
*The model uses a built-in 50% Sensory Dropout (Cortical Blindfolding) during training. This prevents it from taking auto-endcodable shortcut approximations and forces the text-structures to explicitly bridge missing pixel data.*

3. **Inference & Generation**
Once loss converges contextually, prompt the system to generate video, audio, text, and imagery natively:
```bash
python main.py
```
*The system will request a text prompt. It translates the linguistic prompt into auditory spikes, parses it through the prefrontal oscillating loops, and fires its visual imaginations directly back inversely into fully rendered `.png` and `.mp4` formats on disk.*
