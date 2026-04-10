# The Biomimetic SNN Engine
An explicit, mathematically rigorous neurological simulation representing the human brain structure utilizing **Spiking Neural Networks (SNN)** and **Leaky Integrate-and-Fire (LIF)** logic. 

Unlike traditional AIs (which rely on highly structured numerical convolutions and algorithmic Transformers), this model structurally replicates the physical wiring of biological cortices. It utilizes time sequences and membrane potentials to organically link abstract textual concepts directly to massive visual and spatial reconstructions.

## How it Works
The model relies entirely on **Predictive Coding** and biological inference. 
1. **The Senses (Sensory Tokenizer)**: Real-world data (images, videos, text files) are physically converted into binary sequences acting as electrical currents. 
2. **The Thalamus**: Inputs are routed into the Thalamic Bottleneck. 
3. **Primary Cortices (Visual & Auditory)**: It processes geometry natively via `ConvLIFCortexLayer` structures. Spatial connections mirror Receptive Fields (where neurons only connect to localized clusters of their adjacent neurons) rather than arbitrary dense math maps.
4. **Prefrontal Cortex (Attention via Recurrence)**: Features are bound through Top-Down Recurrence (`RecurrentLIFCortexLayer`). Natively circulating energy locally allows adjectives ("Green", "Red") to accurately lock their bindings natively to structural components over biological Time (`15 T-Steps`).
5. **Generative Cross-Modal Routes**: Output from the Prefrontal loop projects heavily **backwards** onto the visual motor strip, permitting the engine to actively "hallucinate" videos/pixels that match its internal conceptual semantic mapping!

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
