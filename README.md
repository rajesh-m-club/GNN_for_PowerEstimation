GNN-Based Switching Activity Prediction for Power Estimation

A Graph Neural Network (GNN)-based research project exploring the prediction of switching activity in digital circuits using circuit netlist representations.

The project investigates how graph-based learning can capture structural relationships between cells and nets to estimate switching activity, supporting power-estimation workflows.

---

Project Overview

Dynamic power consumption in digital circuits is strongly influenced by switching activity. Accurate estimation of signal transitions is therefore an important part of power analysis and optimization.

Traditional approaches often rely on simulation-based activity extraction, which can become computationally expensive when evaluating large designs or exploring multiple implementation stages.

This project explores the use of Graph Neural Networks (GNNs) to learn circuit-level structural relationships and predict switching activity.

The circuit is represented as a directed graph:

- Nodes: Represent circuit cells.
- Edges: Represent connectivity between cells.
- Node features: Capture switching probabilities and other cell-related information.
- Edge features: Capture signal-transition and connectivity information.

The GNN learns from these graph representations to estimate switching-related outputs.

Research Workflow

The broader research investigates two prediction stages:

1. RTL-to-Synthesis: Predicting switching activity at the synthesis stage using RTL-derived information.
2. Synthesis-to-Route: Investigating switching activity estimation between synthesis and routed implementation stages.

The overall workflow can be represented as:

       RTL Design
           |
           v
   RTL Activity Information
           |
           v
   GNN-Based Prediction
           |
           v
   Synthesis Activity
           |
           v
  Synthesis-to-Route
  Activity Estimation
           |
           v
    Routed Activity
           |
           v
      Power Analysis

Repository scope: This public repository exposes the RTL-to-synthesis model implementation. The separate synthesis-to-route research model is not included because its associated research is unpublished. Its implementation and methodology are intentionally withheld.

The reported experimental results below belong to the broader prediction workflow and should not be interpreted as results reproducible from the public model file alone.

---

Public Model Architecture

The publicly available implementation defines an Edge-Aware GraphSAGE architecture.

Unlike conventional graph models that use only node information, this architecture incorporates both node and edge features during message passing.

1. Graph Representation

Component| Description| Dimension
Node features| Switching probabilities and cell-related features| 5
Edge features| Connectivity and transition-related features| 17
Hidden representation| Learned graph embedding| 64
Message-passing layers| Edge-aware GraphSAGE layers| 2
Output heads| Independent switching-related predictions| 4

2. Node Features

Each node is represented using five features:

[p0, p1, pt1, pt0, inv]

These features provide information about signal probabilities and cell-related characteristics.

3. Edge Features

Each directed edge is represented using 17 features:

[SC, TC_00_00, ..., TC_11_11]

These features provide additional information about connectivity and transition relationships between circuit elements.

4. Architecture Details

The model consists of the following stages:

1. Feature Attention: Learns the relative importance of node and edge features.
2. Node and Edge Encoders: Transform the input features into a common hidden representation.
3. Edge-Aware GraphSAGE Layers: Perform message passing using both node and edge information.
4. Attention-Based Aggregation: Aggregates incoming messages at destination nodes.
5. Residual Node Updates: Preserves and updates node representations across graph layers.
6. Prediction Heads: Produces four independent switching-related outputs.

5. Prediction Outputs

The model uses four independent sigmoid output heads:

[p0, p1, psw01, psw10]

These represent the predicted output quantities associated with logic-state probabilities and switching transitions.

The outputs are independently estimated rather than normalized using a softmax operation.

Primary architectural configuration:

Parameter| Value
Node feature dimension| 5
Edge feature dimension| 17
Hidden dimension| 64
GraphSAGE layers| 2
Dropout| 0.1
Prediction heads| 4

---

Experimental Results

The following results summarize the reported evaluation across benchmark circuits.

Two error metrics are presented:

- Switch Error: Reported switching-activity prediction error.
- Total Error: Reported total prediction error.

Switching Error by Circuit

"Switching Error" (assets/switching_error.png)

Total Error by Circuit

"Total Error" (assets/total_error.png)

Detailed Results

Circuit| Switch Error (%)| Total Error (%)
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

Overall Reported Performance

Metric| Result
Average Switch Error| 9.9890%
Average Total Error| 7.5345%
Circuits Evaluated| 18
Circuits Skipped| 12

Twelve benchmark entries were skipped because their corresponding result files were missing during evaluation.

The reported averages are retained from the original evaluation output.

---

Repository Structure

GNN_for_PowerEstimation/
│
├── README.md
│
├── model.py
│
└── assets/
    ├── switching_error.png
    └── total_error.png

Public Files

File| Description
"model.py"| Public Edge-Aware GraphSAGE model definition
"README.md"| Project overview, architecture and experimental results
"assets/"| Experimental result visualizations

Implementation note: The current public repository does not contain the complete training and evaluation pipeline. The model also imports attention and GraphSAGE components from "utilis.attention", which is not included in the public repository. Therefore, this repository is intended primarily to document the research and expose the public model architecture, rather than provide a standalone reproducible implementation.

---

Research Context

This work is inspired by research on graph-based approaches for power estimation and switching-activity prediction.

Related research:

GRIPT — Graph-based research related to power estimation.

"Read the research paper (IEEE DOI)" (https://doi.org/10.1109/TCAD.2025.3563420)

This project is an independent implementation and research exploration. It does not claim to reproduce the GRIPT methodology or its reported results.

---

Key Areas of Work

- Circuit netlist representation using directed graphs.
- Feature engineering for cell and connectivity information.
- Edge-aware graph neural network architecture.
- Attention-based feature extraction and message aggregation.
- Switching-activity prediction using multiple output heads.
- Evaluation across benchmark digital circuits.
- Investigation of GNN-based approaches for power-estimation workflows.

---

Future Research Directions

Potential directions for further investigation include:

- Improving prediction accuracy across different circuit architectures.
- Evaluating generalization to unseen circuit designs.
- Investigating the relationship between circuit topology and switching behavior.
- Exploring graph-based models for faster power-estimation workflows.
- Extending graph-learning approaches to additional stages of the digital implementation flow.

---

Acknowledgment

This project was developed as part of research into machine-learning-assisted power estimation for digital integrated circuits.

Research focus: Graph Neural Networks · VLSI Design · Switching Activity Prediction · RTL · Synthesis · Physical Design · Power Estimation