GNN for Circuit Switching-Activity Prediction

A portfolio overview of my work on graph-neural-network-based switching-activity prediction for digital circuits.

This repository intentionally exposes the model-definition code for the RTL-to-synthesis stage. It is not a complete, turnkey reproduction of the full research flow: dataset generation, netlist/SAIF processing, training and evaluation scripts, and the later synthesis-to-route model are not included here.

---

Project at a glance

The broader project explores whether a circuit can be represented as a graph and learned by a GNN to estimate internal signal activity, reducing dependence on expensive activity-propagation or simulation steps.

The work has two distinct stages:

1. RTL activity to synthesized-circuit activity — the model architecture included in this repository.
2. Synthesis activity to routed-circuit activity — a subsequent research extension developed separately and intentionally not published in this repository.

The public code should therefore be read as a resume/portfolio explanation of the model architecture, not as a complete implementation of the end-to-end experimental pipeline.

Research background and motivation

This work was motivated by graph-learning approaches to circuit activity and power estimation, including the ideas explored in GRIPT: Graph Attention-Assisted Inductive Methodology for Fast and Accurate Average Power Estimation From RTL Simulation Skipping Gate-Level Simulation.

GRIPT demonstrates the use of graph-based learning and attention mechanisms to estimate circuit switching activity from RTL-simulation information. This project takes inspiration from that research direction and explores an Edge-Aware Attentive GraphSAGE architecture with its own node/edge feature representation and output formulation.

- Reference paper: "GRIPT — DOI: 10.1109/TCAD.2025.3563420" (https://doi.org/10.1109/TCAD.2025.3563420)

This repository does not claim to reproduce GRIPT. It documents the model code available here and the direction of the associated project.

---

Public model: Edge-Aware Attentive GraphSAGE

The implementation in "model.py" defines an "EdgeAwareGraphSAGE" model using PyTorch and PyTorch Geometric-style graph data.

Model inputs

For a graph with N nodes and E directed edges:

Input| Shape| Feature layout
Node features| "[N, 5]"| "[p0, p1, pt1, pt0, inv]"
Edge features| "[E, 17]"| "[SC, TC_00_00, ..., TC_11_11]"
Edge index| "[2, E]"| Source-to-destination connectivity

The feature names above are preserved exactly as defined in the model. Their detailed extraction and preprocessing are outside this repository.

Architecture

flowchart TD
    A["Circuit graph input"] --> B["Node features: 5 channels"]
    A --> C["Directed edges + 17 edge features"]
    B --> D["Learned node feature attention"]
    C --> E["Learned edge feature attention"]
    D --> F["Node encoder"]
    E --> G["Edge encoder"]
    F --> H["Edge-aware attentive GraphSAGE"]
    G --> H
    H --> I["Layer 1"]
    I --> J["Layer 2"]
    J --> K["Four independent prediction heads"]
    K --> L["p0, p1, psw01, psw10"]

Main design components

- Learned feature gating: separate feature-attention modules reweight node and edge channels before encoding.
- Node and edge encoders: map the input features into a common hidden representation.
- Edge-aware message passing: each message uses the source-node representation together with its encoded edge features.
- Attention over incoming messages: attention scores use source, edge, and destination representations; scores are normalized over incoming edges at each destination.
- Residual node update: aggregated messages are combined with the current destination-node representation through an update MLP and residual connection.
- Independent output heads: four separate heads predict "p0", "p1", "psw01", and "psw10". Each output is independently sigmoid-constrained by default; the outputs are not treated as a four-class softmax.
- Optional attention outputs: the forward method can return node-feature, edge-feature, and neighbor-attention tensors for inspection.

Model configuration

Parameter| Default
Node input dimension| 5
Edge input dimension| 17
Hidden dimension| 64
GraphSAGE layers| 2
Dropout| 0.1
Output heads| 4
Output constraint| Independent sigmoid

Target masking

The model forward pass computes predictions for all graph nodes. The accompanying model documentation specifies that primary-input nodes are excluded from target loss/metric calculations because their activity is supplied as input; internal nodes remain prediction targets. The mask is applied by the training/evaluation pipeline, not by the model's forward method.

---

Broader research direction

The overall investigation continued beyond the model shown in this repository:

flowchart LR
    A["RTL simulation activity"] --> B["Synthesized circuit representation"]
    B --> C["RTL-to-synthesis activity prediction"]
    C --> D["Synthesis activity"]
    D --> E["Synthesis-to-route activity estimation"]
    E --> F["Routed activity"]

Repository boundary: the public "model.py" corresponds to the RTL-to-synthesis model stage. I subsequently continued the work on synthesis-to-route activity prediction using a separate model and research methodology.

That synthesis-to-route extension is not included in this repository. Its implementation and methodological details are being withheld while the associated research work remains unpublished and is being prepared for dissemination. This is an intentional research/publication boundary, not an indication that the extension is part of the public code.

The end-to-end results below are included to communicate the broader project's experimental progress. They are not results that can be reproduced from this repository alone, because the later model and the full experimental pipeline are not present here.

---

Experimental results — broader prediction flow

The following results are from the broader experimental evaluation of the combined prediction flow. Errors are shown as reported for each benchmark.

Per-circuit results

Circuit| Switching error (%)| Total error (%)
Priority| 1.2800| 1.7964
adder| 29.2806| 17.5090
arbiter| 18.1440| 17.7632
c1355| 8.5308| 5.0000
c17| 14.0299| 2.9289
c1908| 2.4609| 2.5806
c2670| 12.8169| 11.8421
c3540| 7.6923| 7.6503
c432| 1.7921| 0.7800
c499| 17.7215| 15.4839
c5315| 7.3733| 5.3292
c6288| 13.7856| 3.9604
c7552| 8.8328| 7.4118
c880| 10.0000| 10.5691
cavlc| 7.9365| 9.0535
dec| 1.7391| 0.0000
int2float| 9.4595| 9.2848
voter| 6.9264| 6.6778

Visual comparison

The charts below visualize the same per-circuit values listed in the table. Lower error is better.

Switching error (%)

xychart-beta
    x-axis ["Priority", "adder", "arbiter", "c1355", "c17", "c1908", "c2670", "c3540", "c432", "c499", "c5315", "c6288", "c7552", "c880", "cavlc", "dec", "int2float", "voter"]
    y-axis "Error (%)" 0 --> 30
    bar [1.2800, 29.2806, 18.1440, 8.5308, 14.0299, 2.4609, 12.8169, 7.6923, 1.7921, 17.7215, 7.3733, 13.7856, 8.8328, 10.0000, 7.9365, 1.7391, 9.4595, 6.9264]

Total error (%)

xychart-beta
    x-axis ["Priority", "adder", "arbiter", "c1355", "c17", "c1908", "c2670", "c3540", "c432", "c499", "c5315", "c6288", "c7552", "c880", "cavlc", "dec", "int2float", "voter"]
    y-axis "Error (%)" 0 --> 20
    bar [1.7964, 17.5090, 17.7632, 5.0000, 2.9289, 2.5806, 11.8421, 7.6503, 0.7800, 15.4839, 5.3292, 3.9604, 7.4118, 10.5691, 9.0535, 0.0000, 9.2848, 6.6778]

Aggregate reported errors

Metric| Reported average
Average switching error| 9.9890%
Average total error| 7.5345%

These averages are reported for the successfully evaluated circuits. They should be interpreted as broader-flow results, not as standalone performance numbers for the public GraphSAGE code.

Benchmarks not included in the reported average

The following benchmark output files were missing during this evaluation and were skipped:

"bar", "ctrl", "div", "i2c", "log2", "max", "memctrl", "multiplier", "router", "sin", "sqrt", "square".

- Circuits evaluated: 18
- Circuits skipped because output files were missing: 12

---

Repository contents

GNN_for_PowerEstimation/
├── README.md   # Project context, architecture, and reported results
└── model.py    # EdgeAwareGraphSAGE model definition

The model imports "FeatureAttention" and "EdgeAwareSAGELayer" from "utilis.attention". That helper module, along with the dataset/feature-generation and training/evaluation pipeline, is not currently included in this public repository. Consequently, the repository is intended as a concise portfolio reference to the architecture rather than a self-contained runnable package.

Technologies

- Python
- PyTorch
- PyTorch Geometric data conventions
- Graph Neural Networks
- Edge-aware message passing and attention

Research status and scope

This repository makes the public model architecture and the high-level project context visible while keeping the unpublished synthesis-to-route extension private.

For clarity:

- Included: the RTL-to-synthesis Edge-Aware Attentive GraphSAGE model definition in "model.py".
- Explained at a high level only: the subsequent synthesis-to-route research direction.
- Not included: the unpublished model implementation, its internal methodology, private data-processing pipeline, and the complete end-to-end execution flow.

The intent is to provide an accurate overview of the work for a technical portfolio or resume while respecting the ongoing research and publication process.