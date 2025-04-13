# <img src="assets/images/icon.jpg" alt="SVG Image" width="40px"> Awesome-Long-Chain-of-Thought-Reasoning


[![arXiv](https://img.shields.io/badge/arXiv-Long_Chain_of_Thought-b31b1b.svg)](https://arxiv.org/pdf/2503.09567) 
[![Paper](https://img.shields.io/badge/Paper-908-green.svg)](https://github.com//LightChen233/Awesome-Long-Chain-of-Thought-Reasoning)
[![Last Commit](https://img.shields.io/github/last-commit/LightChen233/Awesome-Long-Chain-of-Thought-Reasoning)](https://github.com/LightChen233/Awesome-Long-Chain-of-Thought-Reasoning)
[![Contribution Welcome](https://img.shields.io/badge/Contributions-welcome-blue)]()

![image](./assets/images/overall.png)


<!-- omit in toc -->
## 🔥 News

- **2025.04**: 🎉🎉🎉 We have updated the number of reviewed papers to over 900. Additionally, we have enhanced the presentation with more engaging teaser figure.
- **2025.03**: 🎉🎉🎉 We have published a survey paper titled "[Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models](https://arxiv.org/pdf/2503.09567)". Please feel free to cite or open pull requests for your awesome studies.

<!-- omit in toc -->
## 🌟 Introduction

Welcome to the repository associated with our survey paper, "Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models". This repository contains **resources and updates** related to our ongoing Long CoT research. For a detailed introduction, please refer to [our survey paper](https://arxiv.org/pdf/2503.09567).

Recent advancements in reasoning with large language models (RLLMs), such as OpenAI-O1 and DeepSeek-R1, have demonstrated their impressive capabilities in complex domains like mathematics and coding. A central factor in their success lies in the application of long chain-of-thought (Long CoT) characteristics, which enhance reasoning abilities and enable the solution of intricate problems.

![image](./assets/images/develop.png)

However, despite these developments, a comprehensive survey on Long CoT is still lacking, limiting our understanding of its distinctions from traditional short chain-of-thought (Short CoT) and complicating ongoing debates on issues like "overthinking" and "test-time scaling." This survey seeks to fill this gap by offering a unified perspective on Long CoT. (1) We first distinguish Long CoT from Short CoT and introduce a novel taxonomy to categorize current reasoning paradigms. (2) Next, we explore the key characteristics of Long CoT: deep reasoning, extensive exploration, and feasible reflection, which enable models to handle more complex tasks and produce more efficient, coherent outcomes compared to the shallower Short CoT. (3) We then investigate key phenomena such as the emergence of Long CoT with these characteristics, including overthinking, and test-time scaling, offering insights into how these processes manifest in practice. (4) Finally, we identify significant research gaps and highlight promising future directions, including the integration of multi-modal reasoning, efficiency improvements, and enhanced knowledge frameworks. By providing a structured overview, this survey aims to inspire future research and further the development of logical reasoning in artificial intelligence.

![image](./assets/images/intro.jpg)

<!-- omit in toc -->
## 🔮 Contents

- [Awesome-Long-CoT](#)
  - [Part 1: Analysis and Evaluation](#analysis-and-evaluation)
    - [Analysis & Explanation for Long CoT](#analysis-explanation-for-long-cot)
    - [Long CoT Evaluations](#long-cot-evaluations)
  - [Part 2: Deep Reasoning](#deep-reasoning)
    - [Deep Reasoning Format](#deep-reasoning-format)
    - [Deep Reasoning Learning](#deep-reasoning-learning)
  - [Part 3: Feasible Reflection](#feasible-reflection)
    - [Feedback](#feedback)
    - [Refinement](#refinement)
  - [Part 4: Extensive Exploration](#extensive-exploration)
    - [Exploration Scaling](#exploration-scaling)
    - [Internal Exploration](#internal-exploration)
    - [External Exploration](#external-exploration)
  - [Part 5: Future and Frontiers](#future)
    - [Agentic & Embodied Long CoT](#agentic-embodied-long-cot)
    - [Efficient Long CoT](#efficient-long-cot)
    - [Knowledge-Augmented Long CoT](#knowledge-augmented-long-cot)
    - [Multilingual Long CoT](#multilingual-long-cot)
    - [Multimodal Long CoT](#multimodal-long-cot)
    - [Safety for Long CoT](#safety-long-cot)

![image](./assets/images/contents.jpg)

<h2 id="analysis-and-evaluation">1. Analysis and Evaluation</h2>


<h3 id="analysis-explanation-for-long-cot">1.1 Analysis & Explanation for Long CoT</h3>
<img src="./assets/images/analysis.jpg" style="width: 580pt">
<ul>
<li><i><b>Explainable AI in Large Language Models: A Review</b></i>, Sauhandikaa et al., <a href="http://ieeexplore.ieee.org/abstract/document/10895578" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.00-blue" alt="PDF Badge"></a></li>
<li><i><b>Xai meets llms: A survey of the relation between explainable ai and large language models</b></i>, Cambria et al., <a href="https://arxiv.org/abs/2407.15248" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.07-red" alt="arXiv Badge"></a></li>
<li><i><b>When a language model is optimized for reasoning, does it still show embers of autoregression? An analysis of OpenAI o1</b></i>, McCoy et al., <a href="https://arxiv.org/abs/2410.01792" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Think or Step-by-Step? UnZIPping the Black Box in Zero-Shot Prompts</b></i>, Sadr et al., <a href="https://arxiv.org/abs/2502.03418" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="long-cot-external-behavior-analysis">1.1.1 Long CoT External Behavior Analysis</h4>
</ul>

<ul>
<li><i><b>Language Models Can Predict Their Own Behavior</b></i>, Ashok et al., <a href="https://arxiv.org/abs/2502.13329" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>

<b>Aha Moment Phenomenon</b>
<ul>
<li><i><b>There May Not be Aha Moment in R1-Zero-like Training — A Pilot Study</b></i>, Liu et al., <a href="https://oatllm.notion.site/oat-zero" target="_blank"><img src="https://img.shields.io/badge/Notion-2025.00-white" alt="Notion Badge"></a></li>
<li><i><b>Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning</b></i>, Guo et al., <a href="https://arxiv.org/abs/2501.12948" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Open R1</b></i>, Team et al., <a href="https://github.com/huggingface/open-r1" target="_blank"><img src="https://img.shields.io/badge/Github-2025.01-white" alt="Github Badge"></a></li>
<li><i><b>Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning</b></i>, Xie et al., <a href="https://arxiv.org/abs/2502.14768" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>R1-Zero's" Aha Moment" in Visual Reasoning on a 2B Non-SFT Model</b></i>, Zhou et al., <a href="https://arxiv.org/abs/2503.05132" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>MM-Eureka: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning</b></i>, Meng et al., <a href="https://arxiv.org/abs/2503.07365" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Understanding Aha Moments: from External Observations to Internal Mechanisms</b></i>, Yang et al., <a href="https://arxiv.org/abs/2504.02956" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.04-red" alt="arXiv Badge"></a></li>
</ul>

<b>Inference Test-Time Scaling Phenomenon</b>
<ul>
<li><i><b>Greedy Policy Search: A Simple Baseline for Learnable Test-Time Augmentation</b></i>, Lyzhov et al., <a href="https://proceedings.mlr.press/v124/lyzhov20a.html" target="_blank"><img src="https://img.shields.io/badge/PDF-2020.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Large language monkeys: Scaling inference compute with repeated sampling</b></i>, Brown et al., <a href="https://arxiv.org/abs/2407.21787" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.07-red" alt="arXiv Badge"></a></li>
<li><i><b>The Impact of Reasoning Step Length on Large Language Models</b></i>, Jin et al., <a href="https://aclanthology.org/2024.findings-acl.108/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Inference scaling laws: An empirical analysis of compute-optimal inference for problem-solving with language models</b></i>, Wu et al., <a href="https://arxiv.org/abs/2408.00724" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.08-red" alt="arXiv Badge"></a></li>
<li><i><b>Unlocking the Capabilities of Thought: A Reasoning Boundary Framework to Quantify and Optimize Chain-of-Thought</b></i>, Chen et al., <a href="https://openreview.net/forum?id=pC44UMwy2v" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>From Decoding to Meta-Generation: Inference-time Algorithms for Large Language Models</b></i>, Welleck et al., <a href="https://openreview.net/forum?id=eskQMcIbMS" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Openai o1 system card</b></i>, Jaech et al., <a href="https://arxiv.org/abs/2412.16720" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning</b></i>, Guo et al., <a href="https://arxiv.org/abs/2501.12948" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>When More is Less: Understanding Chain-of-Thought Length in LLMs</b></i>, Wu et al., <a href="https://arxiv.org/abs/2502.07266" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Inference-Time Computations for LLM Reasoning and Planning: A Benchmark and Insights</b></i>, Parashar et al., <a href="https://arxiv.org/abs/2502.12521" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Examining False Positives under Inference Scaling for Mathematical Reasoning</b></i>, Wang et al., <a href="https://arxiv.org/abs/2502.06217" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>ECM: A Unified Electronic Circuit Model for Explaining the Emergence of In-Context Learning and Chain-of-Thought in Large Language Model</b></i>, Chen et al., <a href="https://arxiv.org/abs/2502.03325" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>PhD Knowledge Not Required: A Reasoning Challenge for Large Language Models</b></i>, Anderson et al., <a href="https://arxiv.org/abs/2502.01584" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Scaling Test-Time Compute Without Verification or RL is Suboptimal</b></i>, Setlur et al., <a href="https://arxiv.org/abs/2502.12118" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>The Relationship Between Reasoning and Performance in Large Language Models--o3 (mini) Thinks Harder, Not Longer</b></i>, Ballon et al., <a href="https://arxiv.org/abs/2502.15631" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Inference-Time Scaling for Complex Tasks: Where We Stand and What Lies Ahead</b></i>, Balachandran et al., <a href="https://arxiv.org/abs/2504.00294" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.04-red" alt="arXiv Badge"></a></li>
</ul>

<b>Long CoT Emergence Phenomenon</b>
<ul>
<li><i><b>Star: Bootstrapping reasoning with reasoning</b></i>, Zelikman et al., <a href="https://openreview.net/pdf?id=_3ELRdg2sgI" target="_blank"><img src="https://img.shields.io/badge/PDF-2022.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters</b></i>, Wang et al., <a href="https://aclanthology.org/2023.acl-long.153/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.07-blue" alt="PDF Badge"></a></li>
<li><i><b>LAMBADA: Backward Chaining for Automated Reasoning in Natural Language</b></i>, Kazemi et al., <a href="https://aclanthology.org/2023.acl-long.361/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.07-blue" alt="PDF Badge"></a></li>
<li><i><b>What Makes Chain-of-Thought Prompting Effective? A Counterfactual Study</b></i>, Madaan et al., <a href="https://aclanthology.org/2023.findings-emnlp.101.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Automatic Prompt Augmentation and Selection with Chain-of-Thought from Labeled Data</b></i>, Shum et al., <a href="https://aclanthology.org/2023.findings-emnlp.811/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>MoT: Memory-of-Thought Enables ChatGPT to Self-Improve</b></i>, Li et al., <a href="https://aclanthology.org/2023.emnlp-main.392/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>The llama 3 herd of models</b></i>, Dubey et al., <a href="https://arxiv.org/abs/2407.21783" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.07-red" alt="arXiv Badge"></a></li>
<li><i><b>Do Large Language Models Latently Perform Multi-Hop Reasoning?</b></i>, Yang et al., <a href="https://aclanthology.org/2024.acl-long.550/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Chain of Thoughtlessness? An Analysis of CoT in Planning</b></i>, Stechly et al., <a href="https://openreview.net/forum?id=kPBEAZU5Nm" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Chain-of-Thought Reasoning Without Prompting</b></i>, Wang et al., <a href="https://openreview.net/forum?id=4Zt7S0B0Jp" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Qwen2.5 technical report</b></i>, Yang et al., <a href="https://arxiv.org/abs/2412.15115" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning</b></i>, Guo et al., <a href="https://arxiv.org/abs/2501.12948" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Though</b></i>, Xiang et al., <a href="https://arxiv.org/abs/2501.04682" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Think or Step-by-Step? UnZIPping the Black Box in Zero-Shot Prompts</b></i>, Sadr et al., <a href="https://arxiv.org/abs/2502.03418" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning</b></i>, Xie et al., <a href="https://arxiv.org/abs/2502.14768" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning</b></i>, Ma et al., <a href="https://arxiv.org/abs/2502.15401" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs</b></i>, Gandhi et al., <a href="https://arxiv.org/abs/2503.01307" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Style over Substance: Distilled Language Models Reason Via Stylistic Replication</b></i>, Lippmann et al., <a href="https://arxiv.org/abs/2504.01738" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.04-red" alt="arXiv Badge"></a></li>
<li><i><b>Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning</b></i>, Wang et al., <a href="https://arxiv.org/abs/2504.03635" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.04-red" alt="arXiv Badge"></a></li>
</ul>

<b>Overthinking Phenomenon</b>
<ul>
<li><i><b>The Impact of Reasoning Step Length on Large Language Models</b></i>, Jin et al., <a href="https://aclanthology.org/2024.findings-acl.108/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Unlocking the Capabilities of Thought: A Reasoning Boundary Framework to Quantify and Optimize Chain-of-Thought</b></i>, Chen et al., <a href="https://openreview.net/forum?id=pC44UMwy2v" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Compositional Hardness of Code in Large Language Models--A Probabilistic Perspective</b></i>, Wolf et al., <a href="https://arxiv.org/abs/2409.18028" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.09-red" alt="arXiv Badge"></a></li>
<li><i><b>DynaThink: Fast or Slow? A Dynamic Decision-Making Framework for Large Language Models</b></i>, Pan et al., <a href="https://aclanthology.org/2024.emnlp-main.814/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.11-blue" alt="PDF Badge"></a></li>
<li><i><b>What Are Step-Level Reward Models Rewarding? Counterintuitive Findings from MCTS-Boosted Mathematical Reasoning</b></i>, Ma et al., <a href="https://arxiv.org/abs/2412.15904" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Do not think that much for 2+ 3=? on the overthinking of o1-like llms</b></i>, Chen et al., <a href="https://arxiv.org/abs/2412.21187" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Rethinking External Slow-Thinking: From Snowball Errors to Probability of Correct Reasoning</b></i>, Gan et al., <a href="https://arxiv.org/abs/2501.15602" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Complexity Control Facilitates Reasoning-Based Compositional Generalization in Transformers</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2501.08537" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning</b></i>, Xie et al., <a href="https://arxiv.org/abs/2502.14768" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>When More is Less: Understanding Chain-of-Thought Length in LLMs</b></i>, Wu et al., <a href="https://arxiv.org/abs/2502.07266" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>ECM: A Unified Electronic Circuit Model for Explaining the Emergence of In-Context Learning and Chain-of-Thought in Large Language Model</b></i>, Chen et al., <a href="https://arxiv.org/abs/2502.03325" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks</b></i>, Cuadron et al., <a href="https://arxiv.org/abs/2502.08235" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>OVERTHINKING: Slowdown Attacks on Reasoning LLMs</b></i>, Kumar et al., <a href="https://arxiv.org/abs/2502.02542" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>

<b>PRM v.s. ORM Phenomenon</b>
<ul>
<li><i><b>Concrete problems in AI safety</b></i>, Amodei et al., <a href="https://arxiv.org/abs/1606.06565" target="_blank"><img src="https://img.shields.io/badge/arXiv-2016.06-red" alt="arXiv Badge"></a></li>
<li><i><b>The effects of reward misspecification: Mapping and mitigating misaligned models</b></i>, Pan et al., <a href="https://arxiv.org/abs/2201.03544" target="_blank"><img src="https://img.shields.io/badge/arXiv-2022.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Goal misgeneralization in deep reinforcement learning</b></i>, Di Langosco et al., <a href="https://proceedings.mlr.press/v162/langosco22a/langosco22a.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2022.10-blue" alt="PDF Badge"></a></li>
<li><i><b>Can language models learn from explanations in context?</b></i>, Lampinen et al., <a href="https://aclanthology.org/2022.findings-emnlp.38" target="_blank"><img src="https://img.shields.io/badge/PDF-2022.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Causal Abstraction for Chain-of-Thought Reasoning in Arithmetic Word Problems</b></i>, Tan et al., <a href="https://aclanthology.org/2023.blackboxnlp-1.12" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Processbench: Identifying process errors in mathematical reasoning</b></i>, Zheng et al., <a href="https://arxiv.org/abs/2412.06559" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models</b></i>, Xu et al., <a href="https://arxiv.org/abs/2501.09686" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models</b></i>, Song et al., <a href="https://arxiv.org/abs/2501.03124" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning</b></i>, Guo et al., <a href="https://arxiv.org/abs/2501.12948" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Do We Need to Verify Step by Step? Rethinking Process Supervision from a Theoretical Perspective</b></i>, Jia et al., <a href="https://arxiv.org/abs/2502.10581" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Unveiling and Causalizing CoT: A Causal Pespective</b></i>, Fu et al., <a href="https://arxiv.org/abs/2502.18239" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning</b></i>, Xie et al., <a href="https://arxiv.org/abs/2502.14768" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems</b></i>, Peng et al., <a href="https://arxiv.org/abs/2502.19328" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Process-based Self-Rewarding Language Models</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2503.03746" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation</b></i>, Baker et al., <a href="https://openai.com/index/chain-of-thought-monitoring/" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.03-blue" alt="PDF Badge"></a></li>
<li><i><b>Rewarding Curse: Analyze and Mitigate Reward Modeling Issues for LLM Reasoning</b></i>, Li et al., <a href="https://arxiv.org/abs/2503.05188" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>

<b>Reasoning Boundary Phenomenon</b>
<ul>
<li><i><b>The Expressive Power of Transformers with Chain of Thought</b></i>, Merrill et al., <a href="https://openreview.net/pdf?id=CDmerQ37Zs" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Chain of Thought Empowers Transformers to Solve Inherently Serial Problems</b></i>, Li et al., <a href="https://openreview.net/pdf?id=3EWTEy9MTM" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Mathprompter: Mathematical reasoning using large language models</b></i>, Imani et al., <a href="https://arxiv.org/abs/2303.05398" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective</b></i>, Feng et al., <a href="https://openreview.net/forum?id=qHrADgAdYu" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.09-blue" alt="PDF Badge"></a></li>
<li><i><b>When Do Program-of-Thought Works for Reasoning?</b></i>, Bi et al., <a href="https://ojs.aaai.org/index.php/AAAI/article/view/29721/31237" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.00-blue" alt="PDF Badge"></a></li>
<li><i><b>MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning</b></i>, Sprague et al., <a href="https://openreview.net/forum?id=jenyYQzue1" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.01-blue" alt="PDF Badge"></a></li>
<li><i><b>How Far Are We on the Decision-Making of LLMs? Evaluating LLMs' Gaming Ability in Multi-Agent Environments</b></i>, Huang et al., <a href="https://arxiv.org/abs/2403.11807" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Unlocking the Capabilities of Thought: A Reasoning Boundary Framework to Quantify and Optimize Chain-of-Thought</b></i>, Chen et al., <a href="https://openreview.net/forum?id=pC44UMwy2v" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Not All LLM Reasoners Are Created Equal</b></i>, Hosseini et al., <a href="https://openreview.net/forum?id=aPAWbip1xV" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.10-blue" alt="PDF Badge"></a></li>
<li><i><b>Exploring the Compositional Deficiency of Large Language Models in Mathematical Reasoning Through Trap Problems</b></i>, Zhao et al., <a href="https://aclanthology.org/2024.emnlp-main.915/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.11-blue" alt="PDF Badge"></a></li>
<li><i><b>GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity?</b></i>, Zhou et al., <a href="https://arxiv.org/abs/2502.05252" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Lower Bounds for Chain-of-Thought Reasoning in Hard-Attention Transformers</b></i>, Amiri et al., <a href="https://arxiv.org/abs/2502.02393" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>The Lookahead Limitation: Why Multi-Operand Addition is Hard for LLMs</b></i>, Baeumel et al., <a href="https://arxiv.org/abs/2502.19981" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Reasoning Beyond Limits: Advances and Open Problems for LLMs</b></i>, Ferrag et al., <a href="https://arxiv.org/abs/2503.22732" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="long-cot-internal-mechanism-analysis">1.1.2 Long CoT Internal Mechanism Analysis</h4>
</ul>

<b>Knowledge Incorporating Mechanism</b>
<ul>
<li><i><b>Why think step by step? Reasoning emerges from the locality of experience</b></i>, Prystawski et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/e0af79ad53a336b4c4b4f7e2a68eb609-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Thinking llms: General instruction following with thought generation</b></i>, Wu et al., <a href="https://arxiv.org/abs/2410.10630" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>On the reasoning capacity of ai models and how to quantify it</b></i>, Radha et al., <a href="https://arxiv.org/abs/2501.13833" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Exploring Concept Depth: How Large Language Models Acquire Knowledge and Concept at Different Layers?</b></i>, Jin et al., <a href="https://aclanthology.org/2025.coling-main.37/" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training</b></i>, Ou et al., <a href="https://arxiv.org/abs/2502.11196" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning</b></i>, Xie et al., <a href="https://arxiv.org/abs/2502.14768" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Enhancing llm reliability via explicit knowledge boundary modeling</b></i>, Zheng et al., <a href="https://arxiv.org/abs/2503.02233" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>

<b>Reasoning Interal Mechanism</b>
<ul>
<li><i><b>How Large Language Models Implement Chain-of-Thought?</b></i>, Wang et al., <a href="https://openreview.net/pdf?id=b2XfOm3RJa" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.09-blue" alt="PDF Badge"></a></li>
<li><i><b>How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model</b></i>, Hanna et al., <a href="https://openreview.net/forum?id=p4PckNQR8k" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.09-blue" alt="PDF Badge"></a></li>
<li><i><b>System 2 Attention (is something you might need too)</b></i>, Weston et al., <a href="https://arxiv.org/abs/2311.11829" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.11-red" alt="arXiv Badge"></a></li>
<li><i><b>How to think step-by-step: A mechanistic understanding of chain-of-thought reasoning</b></i>, Dutta et al., <a href="https://openreview.net/forum?id=uHLDkQVtyC" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.07-blue" alt="PDF Badge"></a></li>
<li><i><b>An Investigation of Neuron Activation as a Unified Lens to Explain Chain-of-Thought Eliciting Arithmetic Reasoning of LLMs</b></i>, Rai et al., <a href="https://aclanthology.org/2024.acl-long.387/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>What Happened in LLMs Layers when Trained for Fast vs. Slow Thinking: A Gradient Perspective</b></i>, Li et al., <a href="https://arxiv.org/abs/2410.23743" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Finite State Automata Inside Transformers with Chain-of-Thought: A Mechanistic Study on State Tracking</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2502.20129" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>The Validation Gap: A Mechanistic Analysis of How Language Models Compute Arithmetic but Fail to Validate It</b></i>, Bertolazzi et al., <a href="https://arxiv.org/abs/2502.11771" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Layer by Layer: Uncovering Hidden Representations in Language Models</b></i>, Skean et al., <a href="https://arxiv.org/abs/2502.02013" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Back Attention: Understanding and Enhancing Multi-Hop Reasoning in Large Language Models</b></i>, Yu et al., <a href="https://arxiv.org/abs/2502.10835" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>



<h3 id="long-cot-evaluations">1.2 Long CoT Evaluations</h3>
<h4 id="advanced-evaluation">1.2.3 Advanced Evaluation</h4>
</ul>

<b>AI for Research</b>
<ul>
<li><i><b>ScienceWorld: Is your Agent Smarter than a 5th Grader?</b></i>, Wang et al., <a href="https://aclanthology.org/2022.emnlp-main.775/" target="_blank"><img src="https://img.shields.io/badge/PDF-2022.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Can llms generate novel research ideas? a large-scale human study with 100+ nlp researchers</b></i>, Si et al., <a href="https://arxiv.org/abs/2409.04109" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.09-red" alt="arXiv Badge"></a></li>
<li><i><b>Mle-bench: Evaluating machine learning agents on machine learning engineering</b></i>, Chan et al., <a href="https://arxiv.org/abs/2410.07095" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Chain of ideas: Revolutionizing research via novel idea development with llm agents</b></i>, Li et al., <a href="https://arxiv.org/abs/2410.13185" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>HardML: A Benchmark For Evaluating Data Science And Machine Learning knowledge and reasoning in AI</b></i>, Pricope et al., <a href="https://arxiv.org/abs/2501.15627" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking</b></i>, Li et al., <a href="https://arxiv.org/abs/2502.20730" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Large Language Models Penetration in Scholarly Writing and Peer Review</b></i>, Zhou et al., <a href="https://arxiv.org/abs/2502.11193" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Towards an AI co-scientist</b></i>, Gottweis et al., <a href="https://arxiv.org/abs/2502.18864" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research</b></i>, Wu et al., <a href="https://arxiv.org/abs/2502.04644" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Open Deep Research</b></i>, Team et al., <a href="https://github.com/nickscamara/open-deep-research" target="_blank"><img src="https://img.shields.io/badge/Github-2025.02-white" alt="Github Badge"></a></li>
<li><i><b>Enabling AI Scientists to Recognize Innovation: A Domain-Agnostic Algorithm for Assessing Novelty</b></i>, Wang et al., <a href="https://arxiv.org/abs/2503.01508" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>

<b>Agentic & Embodied Reasoning</b>
<ul>
<li><i><b>WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents</b></i>, Yao et al., <a href="https://openreview.net/forum?id=R9KnuFlvnU" target="_blank"><img src="https://img.shields.io/badge/PDF-2022.00-blue" alt="PDF Badge"></a></li>
<li><i><b>ScienceWorld: Is your Agent Smarter than a 5th Grader?</b></i>, Wang et al., <a href="https://aclanthology.org/2022.emnlp-main.775/" target="_blank"><img src="https://img.shields.io/badge/PDF-2022.12-blue" alt="PDF Badge"></a></li>
<li><i><b>WebArena: A Realistic Web Environment for Building Autonomous Agents</b></i>, Zhou et al., <a href="https://openreview.net/forum?id=oKn9c6ytLx" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.01-blue" alt="PDF Badge"></a></li>
<li><i><b>How Far Are We on the Decision-Making of LLMs? Evaluating LLMs' Gaming Ability in Multi-Agent Environments</b></i>, Huang et al., <a href="https://arxiv.org/abs/2403.11807" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.03-red" alt="arXiv Badge"></a></li>
<li><i><b>CogAgent: A Visual Language Model for GUI Agents</b></i>, Hong et al., <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Hong_CogAgent_A_Visual_Language_Model_for_GUI_Agents_CVPR_2024_paper.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.06-blue" alt="PDF Badge"></a></li>
<li><i><b>OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments</b></i>, Xie et al., <a href="https://openreview.net/forum?id=tN61DTr4Ed" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>ToolComp: A Multi-Tool Reasoning & Process Supervision Benchmark</b></i>, Nath et al., <a href="https://arxiv.org/abs/2501.01290" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks</b></i>, Wang et al., <a href="https://arxiv.org/abs/2501.11733" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2502.12054" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Text2World: Benchmarking Large Language Models for Symbolic World Model Generation</b></i>, Hu et al., <a href="https://arxiv.org/abs/2502.13092" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>WebGames: Challenging General-Purpose Web-Browsing AI Agents</b></i>, Thomas et al., <a href="https://arxiv.org/abs/2502.18356" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>VEM: Environment-Free Exploration for Training GUI Agent with Value Environment Model</b></i>, Zheng et al., <a href="https://arxiv.org/abs/2502.18906" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Mobile-Agent-V: Learning Mobile Device Operation Through Video-Guided Multi-Agent Collaboration</b></i>, Wang et al., <a href="https://arxiv.org/abs/2502.17110" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Generating Symbolic World Models via Test-time Scaling of Large Language Models</b></i>, Yu et al., <a href="https://arxiv.org/abs/2502.04728" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>UI-R1: Enhancing Action Prediction of GUI Agents by Reinforcement Learning</b></i>, Lu et al., <a href="https://arxiv.org/abs/2503.21620" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>

<b>Multimodal Reasoning</b>
<ul>
<li><i><b>Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering</b></i>, Lu et al., <a href="https://openreview.net/forum?id=HjwK-Tc_Bc" target="_blank"><img src="https://img.shields.io/badge/PDF-2022.11-blue" alt="PDF Badge"></a></li>
<li><i><b>A Multi-Modal Neural Geometric Solver with Textual Clauses Parsed from Diagram</b></i>, Zhang et al., <a href="https://doi.org/10.24963/ijcai.2023/376" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.00-blue" alt="PDF Badge"></a></li>
<li><i><b>MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts</b></i>, Lu et al., <a href="https://openreview.net/forum?id=KUNzEQMWU7" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Plot2code: A comprehensive benchmark for evaluating multi-modal large language models in code generation from scientific plots</b></i>, Wu et al., <a href="https://arxiv.org/abs/2405.07990" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.05-red" alt="arXiv Badge"></a></li>
<li><i><b>M<sup>3</sup>CoT: A Novel Benchmark for Multi-Domain Multi-step Multi-modal Chain-of-Thought</b></i>, Chen et al., <a href="https://aclanthology.org/2024.acl-long.446/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns</b></i>, Chia et al., <a href="https://aclanthology.org/2024.findings-acl.962/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Can LLMs Solve Molecule Puzzles? A Multimodal Benchmark for Molecular Structure Elucidation</b></i>, Guo et al., <a href="https://openreview.net/forum?id=t1mAXb4Cop" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset</b></i>, Wang et al., <a href="https://openreview.net/forum?id=QWTCcxMpPA" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems?</b></i>, Zhang et al., <a href="https://link.springer.com/chapter/10.1007/978-3-031-73242-3_10" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.10-blue" alt="PDF Badge"></a></li>
<li><i><b>HumanEval-V: Evaluating Visual Understanding and Reasoning Abilities of Large Multimodal Models Through Coding Tasks</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2410.12381" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>A Survey of Mathematical Reasoning in the Era of Multimodal Large Language Model: Benchmark, Method & Challenges</b></i>, Yan et al., <a href="https://arxiv.org/abs/2412.11936" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>CoMT: A Novel Benchmark for Chain of Multi-modal Thought on Large Vision-Language Models</b></i>, Cheng et al., <a href="https://arxiv.org/abs/2412.12932" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>CMMaTH: A Chinese Multi-modal Math Skill Evaluation Benchmark for Foundation Models</b></i>, Li et al., <a href="https://aclanthology.org/2025.coling-main.184/" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>ChartMimic: Evaluating LMM's Cross-Modal Reasoning Capability via Chart-to-Code Generation</b></i>, Yang et al., <a href="https://openreview.net/forum?id=sGpCzsfd1K" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Can Large Language Models Unveil the Mysteries? An Exploration of Their Ability to Unlock Information in Complex Scenarios</b></i>, Wang et al., <a href="https://arxiv.org/abs/2502.19973" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>EnigmaEval: A Benchmark of Long Multimodal Reasoning Challenges</b></i>, Wang et al., <a href="https://arxiv.org/abs/2502.08859" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Code-Vision: Evaluating Multimodal LLMs Logic Understanding and Code Generation Capabilities</b></i>, Wang et al., <a href="https://arxiv.org/abs/2502.11829" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Exploring and Evaluating Multimodal Knowledge Reasoning Consistency of Multimodal Large Language Models</b></i>, Jia et al., <a href="https://arxiv.org/abs/2503.04801" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>MMSciBench: Benchmarking Language Models on Multimodal Scientific Problems</b></i>, Ye et al., <a href="https://arxiv.org/abs/2503.01891" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>LEGO-Puzzles: How Good Are MLLMs at Multi-Step Spatial Reasoning?</b></i>, Tang et al., <a href="https://arxiv.org/abs/2503.19990" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="benchmarks">1.2.4 Benchmarks</h4>
</ul>

<b>Outcome Benchmarks</b>
<ul>
<li><i><b>On the measure of intelligence</b></i>, Chollet et al., <a href="https://arxiv.org/abs/1911.01547" target="_blank"><img src="https://img.shields.io/badge/arXiv-2019.11-red" alt="arXiv Badge"></a></li>
<li><i><b>What Disease Does This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams</b></i>, Jin et al., <a href="https://www.mdpi.com/2076-3417/11/14/6421" target="_blank"><img src="https://img.shields.io/badge/PDF-2021.07-blue" alt="PDF Badge"></a></li>
<li><i><b>Training verifiers to solve math word problems</b></i>, Cobbe et al., <a href="https://arxiv.org/abs/2110.14168" target="_blank"><img src="https://img.shields.io/badge/arXiv-2021.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Measuring Mathematical Problem Solving With the MATH Dataset</b></i>, Hendrycks et al., <a href="https://openreview.net/forum?id=7Bywt2mQsCe" target="_blank"><img src="https://img.shields.io/badge/PDF-2021.10-blue" alt="PDF Badge"></a></li>
<li><i><b>Competition-Level Code Generation with AlphaCode</b></i>, Li et al., <a href="https://arxiv.org/abs/2203.07814" target="_blank"><img src="https://img.shields.io/badge/arXiv-2022.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them</b></i>, Suzgun et al., <a href="https://aclanthology.org/2023.findings-acl.824/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.07-blue" alt="PDF Badge"></a></li>
<li><i><b>Assessing and Enhancing the Robustness of Large Language Models with Task Structure Variations for Logical Reasoning</b></i>, Bao et al., <a href="https://arxiv.org/abs/2310.09430" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.10-red" alt="arXiv Badge"></a></li>
<li><i><b>AI for Math or Math for AI? On the Generalization of Learning Mathematical Problem Solving</b></i>, Zhou et al., <a href="https://openreview.net/forum?id=xlnvZ85CSo" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.00-blue" alt="PDF Badge"></a></li>
<li><i><b>OlympicArena: Benchmarking Multi-discipline Cognitive Reasoning for Superintelligent AI</b></i>, Huang et al., <a href="https://openreview.net/forum?id=ayF8bEKYQy" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.00-blue" alt="PDF Badge"></a></li>
<li><i><b>Putnam-AXIOM: A Functional and Static Benchmark for Measuring Higher Level Mathematical Reasoning</b></i>, Gulati et al., <a href="https://openreview.net/forum?id=YXnwlZe0yf" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.00-blue" alt="PDF Badge"></a></li>
<li><i><b>Let's verify step by step</b></i>, Lightman et al., <a href="https://openreview.net/forum?id=v8L0pN6EOi" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.01-blue" alt="PDF Badge"></a></li>
<li><i><b>SWE-bench: Can Language Models Resolve Real-world Github Issues?</b></i>, Jimenez et al., <a href="https://openreview.net/forum?id=VTF8yNQM66" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Benchmarking large language models on answering and explaining challenging medical questions</b></i>, Chen et al., <a href="https://arxiv.org/abs/2402.18060" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Achieving> 97% on GSM8K: Deeply Understanding the Problems Makes LLMs Better Solvers for Math Word Problems</b></i>, Zhong et al., <a href="https://arxiv.org/abs/2404.14963" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.04-red" alt="arXiv Badge"></a></li>
<li><i><b>Mhpp: Exploring the capabilities and limitations of language models beyond basic code generation</b></i>, Dai et al., <a href="https://arxiv.org/abs/2405.11430" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.05-red" alt="arXiv Badge"></a></li>
<li><i><b>AIME 2024</b></i>, AI-MO et al., <a href="https://huggingface.co/datasets/AI-MO/aimo-validation-aime" target="_blank"><img src="https://img.shields.io/badge/Huggingface-2024.07-yellow" alt="Huggingface Badge"></a></li>
<li><i><b>AMC 2023</b></i>, AI-MO et al., <a href="https://huggingface.co/datasets/AI-MO/aimo-validation-amc" target="_blank"><img src="https://img.shields.io/badge/Huggingface-2024.07-yellow" alt="Huggingface Badge"></a></li>
<li><i><b>GPQA: A Graduate-Level Google-Proof Q&A Benchmark</b></i>, Rein et al., <a href="https://openreview.net/forum?id=Ti67584b98" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.07-blue" alt="PDF Badge"></a></li>
<li><i><b>OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems</b></i>, He et al., <a href="https://aclanthology.org/2024.acl-long.211/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark</b></i>, Wang et al., <a href="https://openreview.net/forum?id=y10DM6R2r3" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Frontiermath: A benchmark for evaluating advanced mathematical reasoning in ai</b></i>, Glazer et al., <a href="https://arxiv.org/abs/2411.04872" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>HumanEval Pro and MBPP Pro: Evaluating Large Language Models on Self-invoking Code Generation</b></i>, Yu et al., <a href="https://arxiv.org/abs/2412.21199" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>LiveBench: A Challenging, Contamination-Limited LLM Benchmark</b></i>, White et al., <a href="https://openreview.net/forum?id=sKYHBTAxVa" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.00-blue" alt="PDF Badge"></a></li>
<li><i><b>LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code</b></i>, Jain et al., <a href="https://openreview.net/forum?id=chfJJYC3iL" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>JustLogic: A Comprehensive Benchmark for Evaluating Deductive Reasoning in Large Language Models</b></i>, Chen et al., <a href="https://arxiv.org/abs/2501.14851" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Humanity's Last Exam</b></i>, Phan et al., <a href="https://arxiv.org/abs/2501.14249" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding</b></i>, Zuo et al., <a href="https://arxiv.org/abs/2501.18362" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Theoretical Physics Benchmark (TPBench)--a Dataset and Study of AI Reasoning Capabilities in Theoretical Physics</b></i>, Chung et al., <a href="https://arxiv.org/abs/2502.15815" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>AIME 2025</b></i>, OpenCompass et al., <a href="https://huggingface.co/datasets/opencompass/AIME2025" target="_blank"><img src="https://img.shields.io/badge/Huggingface-2025.02-yellow" alt="Huggingface Badge"></a></li>
<li><i><b>ThinkBench: Dynamic Out-of-Distribution Evaluation for Robust LLM Reasoning</b></i>, Huang et al., <a href="https://arxiv.org/abs/2502.16268" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>MATH-Perturb: Benchmarking LLMs' Math Reasoning Abilities against Hard Perturbations</b></i>, Huang et al., <a href="https://arxiv.org/abs/2502.06453" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>ProBench: Benchmarking Large Language Models in Competitive Programming</b></i>, Yang et al., <a href="https://arxiv.org/abs/2502.20868" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>EquiBench: Benchmarking Code Reasoning Capabilities of Large Language Models via Equivalence Checking</b></i>, Wei et al., <a href="https://arxiv.org/abs/2502.12466" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning</b></i>, Lin et al., <a href="https://arxiv.org/abs/2502.01100" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>DivIL: Unveiling and Addressing Over-Invariance for Out-of-Distribution Generalization</b></i>, WANG et al., <a href="https://openreview.net/forum?id=2Zan4ATYsh" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.02-blue" alt="PDF Badge"></a></li>
<li><i><b>SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines</b></i>, Du et al., <a href="https://arxiv.org/abs/2502.14739" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>DeepSeek-R1 Outperforms Gemini 2.0 Pro, OpenAI o1, and o3-mini in Bilingual Complex Ophthalmology Reasoning</b></i>, Xu et al., <a href="https://arxiv.org/abs/2502.17947" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>QuestBench: Can LLMs ask the right question to acquire information in reasoning tasks?</b></i>, Li et al., <a href="https://arxiv.org/abs/2503.22674" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad</b></i>, Petrov et al., <a href="https://arxiv.org/abs/2503.21934" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Benchmarking Reasoning Robustness in Large Language Models</b></i>, Yu et al., <a href="https://arxiv.org/abs/2503.04550" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>From Code to Courtroom: LLMs as the New Software Judges</b></i>, He et al., <a href="https://arxiv.org/abs/2503.02246" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Interacting with AI Reasoning Models: Harnessing" Thoughts" for AI-Driven Software Engineering</b></i>, Treude et al., <a href="https://arxiv.org/abs/2503.00483" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Can Frontier LLMs Replace Annotators in Biomedical Text Mining? Analyzing Challenges and Exploring Solutions</b></i>, Zhao et al., <a href="https://arxiv.org/abs/2503.03261" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>An evaluation of DeepSeek Models in Biomedical Natural Language Processing</b></i>, Zhan et al., <a href="https://arxiv.org/abs/2503.00624" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Cognitive-Mental-LLM: Leveraging Reasoning in Large Language Models for Mental Health Prediction via Online Text</b></i>, Patil et al., <a href="https://arxiv.org/abs/2503.10095" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="process-evaluations">1.2.5 Process Evaluations</h4>
</ul>

<b>Deep Reasoning Benchmarks</b>
<ul>
<li><i><b>ROSCOE: A Suite of Metrics for Scoring Step-by-Step Reasoning</b></i>, Golovneva et al., <a href="https://openreview.net/forum?id=xYlJRpzZtsY" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.00-blue" alt="PDF Badge"></a></li>
<li><i><b>Making Language Models Better Reasoners with Step-Aware Verifier</b></i>, Li et al., <a href="https://aclanthology.org/2023.acl-long.291/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.07-blue" alt="PDF Badge"></a></li>
<li><i><b>ReCEval: Evaluating Reasoning Chains via Correctness and Informativeness</b></i>, Prasad et al., <a href="https://aclanthology.org/2023.emnlp-main.622/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Unlocking the Capabilities of Thought: A Reasoning Boundary Framework to Quantify and Optimize Chain-of-Thought</b></i>, Chen et al., <a href="https://openreview.net/forum?id=pC44UMwy2v" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning</b></i>, Lin et al., <a href="https://arxiv.org/abs/2502.01100" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Evaluating Step-by-step Reasoning Traces: A Survey</b></i>, Lee et al., <a href="https://arxiv.org/abs/2502.12289" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Mathematical Reasoning in Large Language Models: Assessing Logical and Arithmetic Errors across Wide Numerical Ranges</b></i>, Shrestha et al., <a href="https://arxiv.org/abs/2502.08680" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models</b></i>, Zhou et al., <a href="https://arxiv.org/abs/2503.22165" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>

<b>Exploration Benchmarks</b>
<ul>
<li><i><b>EVOLvE: Evaluating and Optimizing LLMs For Exploration</b></i>, Nie et al., <a href="https://arxiv.org/abs/2410.06238" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Evaluating the Systematic Reasoning Abilities of Large Language Models through Graph Coloring</b></i>, Heyman et al., <a href="https://arxiv.org/abs/2502.07087" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Inference-Time Computations for LLM Reasoning and Planning: A Benchmark and Insights</b></i>, Parashar et al., <a href="https://arxiv.org/abs/2502.12521" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>

<b>Reflection Benchmarks</b>
<ul>
<li><i><b>Rewardbench: Evaluating reward models for language modeling</b></i>, Lambert et al., <a href="https://arxiv.org/abs/2403.13787" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.03-red" alt="arXiv Badge"></a></li>
<li><i><b>MR-Ben: A Meta-Reasoning Benchmark for Evaluating System-2 Thinking in LLMs</b></i>, Zeng et al., <a href="https://openreview.net/forum?id=GN2qbxZlni" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.06-blue" alt="PDF Badge"></a></li>
<li><i><b>Evaluating LLMs at Detecting Errors in LLM Responses</b></i>, Kamoi et al., <a href="https://openreview.net/forum?id=dnwRScljXr" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.07-blue" alt="PDF Badge"></a></li>
<li><i><b>CriticBench: Benchmarking LLMs for Critique-Correct Reasoning</b></i>, Lin et al., <a href="https://aclanthology.org/2024.findings-acl.91/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Judgebench: A benchmark for evaluating llm-based judges</b></i>, Tan et al., <a href="https://arxiv.org/abs/2410.12784" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Errorradar: Benchmarking complex mathematical reasoning of multimodal large language models via error detection</b></i>, Yan et al., <a href="https://arxiv.org/abs/2410.04509" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Processbench: Identifying process errors in mathematical reasoning</b></i>, Zheng et al., <a href="https://arxiv.org/abs/2412.06559" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Medec: A benchmark for medical error detection and correction in clinical notes</b></i>, Abacha et al., <a href="https://arxiv.org/abs/2412.19260" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models</b></i>, Song et al., <a href="https://arxiv.org/abs/2501.03124" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Multimodal RewardBench: Holistic Evaluation of Reward Models for Vision Language Models</b></i>, Yasunaga et al., <a href="https://arxiv.org/abs/2502.14191" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>CodeCriticBench: A Holistic Code Critique Benchmark for Large Language Models</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2502.16614" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?</b></i>, He et al., <a href="https://arxiv.org/abs/2502.19361" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>FINEREASON: Evaluating and Improving LLMs' Deliberate Reasoning through Reflective Puzzle Solving</b></i>, Chen et al., <a href="https://arxiv.org/abs/2502.20238" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>
<h2 id="deep-reasoning">2. Deep Reasoning</h2>
<ul>
<li><i><b>Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs</b></i>, Wang et al., <a href="https://arxiv.org/abs/2501.18585" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Don't Get Lost in the Trees: Streamlining LLM Reasoning by Overcoming Tree Search Exploration Pitfalls</b></i>, Wang et al., <a href="https://arxiv.org/abs/2502.11183" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>



<h3 id="deep-reasoning-format">2.1 Deep Reasoning Format</h3>
<img src="./assets/images/deep-reasoning-1.jpg" style="width: 580pt">
<h4 id="latent-space-deep-reasoning">2.1.1 Latent Space Deep Reasoning</h4>
</ul>

<ul>
<li><i><b>Guiding language model reasoning with planning tokens</b></i>, Wang et al., <a href="https://arxiv.org/abs/2310.05707" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.10-red" alt="arXiv Badge"></a></li>
<li><i><b>MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning</b></i>, Sprague et al., <a href="https://openreview.net/forum?id=jenyYQzue1" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Quiet-star: Language models can teach themselves to think before speaking</b></i>, Zelikman et al., <a href="https://arxiv.org/abs/2403.09629" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.03-red" alt="arXiv Badge"></a></li>
<li><i><b>From explicit cot to implicit cot: Learning to internalize cot step by step</b></i>, Deng et al., <a href="https://arxiv.org/abs/2405.14838" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.05-red" alt="arXiv Badge"></a></li>
<li><i><b>Training large language models to reason in a continuous latent space</b></i>, Hao et al., <a href="https://arxiv.org/abs/2412.06769" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Efficient Reasoning with Hidden Thinking</b></i>, Shen et al., <a href="https://arxiv.org/abs/2501.19201" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach</b></i>, Geiping et al., <a href="https://arxiv.org/abs/2502.05171" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Reasoning with Latent Thoughts: On the Power of Looped Transformers</b></i>, Saunshi et al., <a href="https://arxiv.org/abs/2502.17416" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Self-Enhanced Reasoning Training: Activating Latent Reasoning in Small Models for Enhanced Reasoning Distillation</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2502.12744" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>LLM Pretraining with Continuous Concepts</b></i>, Tack et al., <a href="https://arxiv.org/abs/2502.08524" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Scalable Language Models with Posterior Inference of Latent Thought Vectors</b></i>, Kong et al., <a href="https://arxiv.org/abs/2502.01567" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking</b></i>, Chen et al., <a href="https://arxiv.org/abs/2502.13842" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Reasoning to Learn from Latent Thoughts</b></i>, Ruan et al., <a href="https://arxiv.org/abs/2503.18866" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="natural-language-deep-reasoning">2.1.2 Natural Language Deep Reasoning</h4>
</ul>

<ul>
<li><i><b>Reflection of thought: Inversely eliciting numerical reasoning in language models via solving linear systems</b></i>, Zhou et al., <a href="https://arxiv.org/abs/2210.05075" target="_blank"><img src="https://img.shields.io/badge/arXiv-2022.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Chain-of-Thought Prompting Elicits Reasoning in Large Language Models</b></i>, Wei et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2022.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Mathprompter: Mathematical reasoning using large language models</b></i>, Imani et al., <a href="https://arxiv.org/abs/2303.05398" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Deductive Verification of Chain-of-Thought Reasoning</b></i>, Ling et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/72393bd47a35f5b3bee4c609e7bba733-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Cross-lingual Prompting: Improving Zero-shot Chain-of-Thought Reasoning across Languages</b></i>, Qin et al., <a href="https://aclanthology.org/2023.emnlp-main.163/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>AutoCAP: Towards Automatic Cross-lingual Alignment Planning for Zero-shot Chain-of-Thought</b></i>, Zhang et al., <a href="https://aclanthology.org/2024.findings-acl.546/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Large language models are not strong abstract reasoners</b></i>, Gendron et al., <a href="https://doi.org/10.24963/ijcai.2024/693" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Planning in Natural Language Improves LLM Search for Code Generation</b></i>, Wang et al., <a href="https://openreview.net/forum?id=B2iSfPNj49" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.10-blue" alt="PDF Badge"></a></li>
<li><i><b>CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction</b></i>, Li et al., <a href="https://arxiv.org/abs/2502.07316" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="structured-language-deep-reasoning">2.1.3 Structured Language Deep Reasoning</h4>
</ul>

<ul>
<li><i><b>Generative language modeling for automated theorem proving</b></i>, Polu et al., <a href="https://arxiv.org/abs/2009.03393" target="_blank"><img src="https://img.shields.io/badge/arXiv-2020.09-red" alt="arXiv Badge"></a></li>
<li><i><b>Multi-step deductive reasoning over natural language: An empirical study on out-of-distribution generalisation</b></i>, Bao et al., <a href="https://arxiv.org/abs/2207.14000" target="_blank"><img src="https://img.shields.io/badge/arXiv-2022.07-red" alt="arXiv Badge"></a></li>
<li><i><b>PAL: Program-aided Language Models</b></i>, Gao et al., <a href="https://proceedings.mlr.press/v202/gao23f.html" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.07-blue" alt="PDF Badge"></a></li>
<li><i><b>Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks</b></i>, Chen et al., <a href="https://openreview.net/forum?id=YfZ4ZPt8zd" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Tinygsm: achieving> 80% on gsm8k with small language models</b></i>, Liu et al., <a href="https://arxiv.org/abs/2312.09241" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.12-red" alt="arXiv Badge"></a></li>
<li><i><b>ChatLogic: Integrating Logic Programming with Large Language Models for Multi-step Reasoning</b></i>, Wang et al., <a href="https://openreview.net/forum?id=AOqGF7Po7Z" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Brain-Inspired Two-Stage Approach: Enhancing Mathematical Reasoning by Imitating Human Thought Processes</b></i>, Chen et al., <a href="https://arxiv.org/abs/2403.00800" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.03-red" alt="arXiv Badge"></a></li>
<li><i><b>MathDivide: Improved mathematical reasoning by large language models</b></i>, Srivastava et al., <a href="https://arxiv.org/abs/2405.13004" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.05-red" alt="arXiv Badge"></a></li>
<li><i><b>Certified Deductive Reasoning with Language Models</b></i>, Poesia et al., <a href="https://openreview.net/forum?id=yXnwrs2Tl6" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.05-blue" alt="PDF Badge"></a></li>
<li><i><b>Interactive Evolution: A Neural-Symbolic Self-Training Framework For Large Language Models</b></i>, Xu et al., <a href="https://arxiv.org/abs/2406.11736" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.06-red" alt="arXiv Badge"></a></li>
<li><i><b>Lean-star: Learning to interleave thinking and proving</b></i>, Lin et al., <a href="https://arxiv.org/abs/2407.10040" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.07-red" alt="arXiv Badge"></a></li>
<li><i><b>Chain of Code: Reasoning with a Language Model-Augmented Code Emulator</b></i>, Li et al., <a href="https://proceedings.mlr.press/v235/li24ar.html" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.07-blue" alt="PDF Badge"></a></li>
<li><i><b>Siam: Self-improving code-assisted mathematical reasoning of large language models</b></i>, Yu et al., <a href="https://arxiv.org/abs/2408.15565" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.08-red" alt="arXiv Badge"></a></li>
<li><i><b>Formal mathematical reasoning: A new frontier in ai</b></i>, Yang et al., <a href="https://arxiv.org/abs/2412.16075" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>SKIntern: Internalizing Symbolic Knowledge for Distilling Better CoT Capabilities into Small Language Models</b></i>, Liao et al., <a href="https://aclanthology.org/2025.coling-main.215.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>CodePlan: Unlocking Reasoning Potential in Large Language Models by Scaling Code-form Planning</b></i>, Wen et al., <a href="https://openreview.net/forum?id=dCPF1wlqj8" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions</b></i>, Ranaldi et al., <a href="https://arxiv.org/abs/2502.12616" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Towards Better Understanding of Program-of-Thought Reasoning in Cross-Lingual and Multilingual Environments</b></i>, Payoungkhamdee et al., <a href="https://arxiv.org/abs/2502.17956" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Beyond Limited Data: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving</b></i>, Dong et al., <a href="https://arxiv.org/abs/2502.00212" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Theorem Prover as a Judge for Synthetic Data Generation</b></i>, Leang et al., <a href="https://arxiv.org/abs/2502.13137" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Code-Driven Inductive Synthesis: Enhancing Reasoning Abilities of Large Language Models with Sequences</b></i>, Chen et al., <a href="https://arxiv.org/abs/2503.13109" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>



<h3 id="deep-reasoning-learning">2.2 Deep Reasoning Learning</h3>
<img src="./assets/images/deep-reasoning-2.png" style="width: 580pt">
</ul>

<ul>
<li><i><b>Instruction tuning for large language models: A survey</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2308.10792" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.08-red" alt="arXiv Badge"></a></li>
<li><i><b>On memorization of large language models in logical reasoning</b></i>, Xie et al., <a href="https://arxiv.org/abs/2410.23123" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs</b></i>, Wang et al., <a href="https://arxiv.org/abs/2501.18585" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning</b></i>, Guo et al., <a href="https://arxiv.org/abs/2501.12948" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Sft memorizes, rl generalizes: A comparative study of foundation model post-training</b></i>, Chu et al., <a href="https://arxiv.org/abs/2501.17161" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Don't Get Lost in the Trees: Streamlining LLM Reasoning by Overcoming Tree Search Exploration Pitfalls</b></i>, Wang et al., <a href="https://arxiv.org/abs/2502.11183" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="deep-reasoning-imitation">2.2.4 Deep Reasoning Imitation</h4>
</ul>

<ul>
<li><i><b>Training verifiers to solve math word problems</b></i>, Cobbe et al., <a href="https://arxiv.org/abs/2110.14168" target="_blank"><img src="https://img.shields.io/badge/arXiv-2021.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Chain of Thought Imitation with Procedure Cloning</b></i>, Yang et al., <a href="https://openreview.net/forum?id=ZJqqSa8FsH9" target="_blank"><img src="https://img.shields.io/badge/PDF-2022.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Large Language Models Are Reasoning Teachers</b></i>, Ho et al., <a href="https://aclanthology.org/2023.acl-long.830/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.07-blue" alt="PDF Badge"></a></li>
<li><i><b>The CoT Collection: Improving Zero-shot and Few-shot Learning of Language Models via Chain-of-Thought Fine-Tuning</b></i>, Kim et al., <a href="https://aclanthology.org/2023.emnlp-main.782/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Brain-Inspired Two-Stage Approach: Enhancing Mathematical Reasoning by Imitating Human Thought Processes</b></i>, Chen et al., <a href="https://arxiv.org/abs/2403.00800" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Exploring Iterative Enhancement for Improving Learnersourced Multiple-Choice Question Explanations with Large Language Models</b></i>, Bao et al., <a href="https://openreview.net/forum?id=a8AE5PYoEi" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.03-blue" alt="PDF Badge"></a></li>
<li><i><b>Qwen2.5-math technical report: Toward mathematical expert model via self-improvement</b></i>, Yang et al., <a href="https://arxiv.org/abs/2409.12122" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.09-red" alt="arXiv Badge"></a></li>
<li><i><b>Enhancing Reasoning Capabilities of LLMs via Principled Synthetic Logic Corpus</b></i>, Morishita et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/8678da90126aa58326b2fc0254b33a8c-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>DART-Math: Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving</b></i>, Tong et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/0ef1afa0daa888d695dcd5e9513bafa3-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>O1 Replication Journey--Part 2: Surpassing O1-preview through Simple Distillation, Big Progress or Bitter Lesson?</b></i>, Huang et al., <a href="https://arxiv.org/abs/2411.16489" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>System-2 Mathematical Reasoning via Enriched Instruction Tuning</b></i>, Cai et al., <a href="https://arxiv.org/abs/2412.16964" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Acemath: Advancing frontier math reasoning with post-training and reward modeling</b></i>, Liu et al., <a href="https://arxiv.org/abs/2412.15084" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems</b></i>, Min et al., <a href="https://arxiv.org/abs/2412.09413" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Openai o1 system card</b></i>, Jaech et al., <a href="https://arxiv.org/abs/2412.16720" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling</b></i>, Bansal et al., <a href="https://openreview.net/forum?id=HuYSURUxs2" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Advancing Math Reasoning in Language Models: The Impact of Problem-Solving Data, Data Synthesis Methods, and Training Stages</b></i>, Chen et al., <a href="https://arxiv.org/abs/2501.14002" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning</b></i>, Guo et al., <a href="https://arxiv.org/abs/2501.12948" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training</b></i>, Yuan et al., <a href="https://arxiv.org/abs/2501.11425" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>s1: Simple test-time scaling</b></i>, Muennighoff et al., <a href="https://arxiv.org/abs/2501.19393" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>RedStar: Does Scaling Long-CoT Data Unlock Better Slow-Reasoning Systems?</b></i>, Xu et al., <a href="https://arxiv.org/abs/2501.11284" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>FastMCTS: A Simple Sampling Strategy for Data Synthesis</b></i>, Li et al., <a href="https://arxiv.org/abs/2502.11476" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>LLMs Can Teach Themselves to Better Predict the Future</b></i>, Turtel et al., <a href="https://arxiv.org/abs/2502.05253" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>SoS1: O1 and R1-Like Reasoning LLMs are Sum-of-Square Solvers</b></i>, Li et al., <a href="https://arxiv.org/abs/2502.20545" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Distillation Scaling Laws</b></i>, Busbridge et al., <a href="https://arxiv.org/abs/2502.08606" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Unveiling the Mechanisms of Explicit CoT Training: How Chain-of-Thought Enhances Reasoning Generalization</b></i>, Yao et al., <a href="https://arxiv.org/abs/2502.04667" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>CoT2Align: Cross-Chain of Thought Distillation via Optimal Transport Alignment for Language Models with Different Tokenizers</b></i>, Le et al., <a href="https://arxiv.org/abs/2502.16806" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning</b></i>, Chen et al., <a href="https://arxiv.org/abs/2502.18001" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision</b></i>, Zhu et al., <a href="https://arxiv.org/abs/2502.20790" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Demystifying Long Chain-of-Thought Reasoning in LLMs</b></i>, Yeo et al., <a href="https://arxiv.org/abs/2502.03373" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>LIMO: Less is More for Reasoning</b></i>, Ye et al., <a href="https://arxiv.org/abs/2502.03387" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>PromptCoT: Synthesizing Olympiad-level Problems for Mathematical Reasoning in Large Language Models</b></i>, Zhao et al., <a href="https://arxiv.org/abs/2503.02324" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Rewarding Graph Reasoning Process makes LLMs more Generalized Reasoners</b></i>, Peng et al., <a href="https://arxiv.org/abs/2503.00845" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>OpenCodeReasoning: Advancing Data Distillation for Competitive Coding</b></i>, Ahmad et al., <a href="https://arxiv.org/abs/2504.01943" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.04-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="deep-reasoning-self-learning">2.2.5 Deep Reasoning Self-Learning</h4>
</ul>

<ul>
<li><i><b>Thinking fast and slow with deep learning and tree search</b></i>, Anthony et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2017.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Star: Bootstrapping reasoning with reasoning</b></i>, Zelikman et al., <a href="https://openreview.net/pdf?id=_3ELRdg2sgI" target="_blank"><img src="https://img.shields.io/badge/PDF-2022.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Synthetic Prompting: Generating Chain-of-Thought Demonstrations for Large Language Models</b></i>, Shao et al., <a href="https://proceedings.mlr.press/v202/shao23a.html" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.07-blue" alt="PDF Badge"></a></li>
<li><i><b>Reinforced self-training (rest) for language modeling</b></i>, Gulcehre et al., <a href="https://arxiv.org/abs/2308.08998" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.08-red" alt="arXiv Badge"></a></li>
<li><i><b>Training Chain-of-Thought via Latent-Variable Inference</b></i>, Hoffman et al., <a href="https://openreview.net/forum?id=a147pIS2Co" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Tree of Thoughts: Deliberate Problem Solving with Large Language Models</b></i>, Yao et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.09-blue" alt="PDF Badge"></a></li>
<li><i><b>RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment</b></i>, Dong et al., <a href="https://openreview.net/forum?id=m7p5O7zblY" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Training large language models for reasoning through reverse curriculum reinforcement learning</b></i>, Xi et al., <a href="https://arxiv.org/abs/2402.05808" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models</b></i>, Singh et al., <a href="https://openreview.net/pdf?id=lNAyUngGFK" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.04-blue" alt="PDF Badge"></a></li>
<li><i><b>V-STaR: Training Verifiers for Self-Taught Reasoners</b></i>, Hosseini et al., <a href="https://openreview.net/forum?id=stmqBSW2dV" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.07-blue" alt="PDF Badge"></a></li>
<li><i><b>ReAct Meets ActRe: Autonomous Annotation of Agent Trajectories for Contrastive Self-Training</b></i>, Yang et al., <a href="https://openreview.net/forum?id=0VLBwQGWpA" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.07-blue" alt="PDF Badge"></a></li>
<li><i><b>Fine-Tuning with Divergent Chains of Thought Boosts Reasoning Through Self-Correction in Language Models</b></i>, Puerto et al., <a href="https://arxiv.org/abs/2407.03181" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.07-red" alt="arXiv Badge"></a></li>
<li><i><b>Direct Large Language Model Alignment Through Self-Rewarding Contrastive Prompt Distillation</b></i>, Liu et al., <a href="https://aclanthology.org/2024.acl-long.523/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Iterative Reasoning Preference Optimization</b></i>, Pang et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/d37c9ad425fe5b65304d500c6edcba00-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Chain of Preference Optimization: Improving Chain-of-Thought Reasoning in LLMs</b></i>, Zhang et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/00d80722b756de0166523a87805dd00f-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>AlphaMath Almost Zero: Process Supervision without Process</b></i>, Chen et al., <a href="https://openreview.net/forum?id=VaXnxQ3UKo" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Cream: Consistency Regularized Self-Rewarding Language Models</b></i>, Wang et al., <a href="https://openreview.net/forum?id=oaWajnM93y" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.10-blue" alt="PDF Badge"></a></li>
<li><i><b>TPO: Aligning Large Language Models with Multi-branch & Multi-step Preference Trees</b></i>, Liao et al., <a href="https://arxiv.org/abs/2410.12854" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Towards Self-Improvement of LLMs via MCTS: Leveraging Stepwise Knowledge with Curriculum Preference Learning</b></i>, Wang et al., <a href="https://arxiv.org/abs/2410.06508" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>On the impact of fine-tuning on chain-of-thought reasoning</b></i>, Lobo et al., <a href="https://arxiv.org/abs/2411.15382" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>Weak-to-Strong Reasoning</b></i>, Yang et al., <a href="https://aclanthology.org/2024.findings-emnlp.490/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Self-Explore: Enhancing Mathematical Reasoning in Language Models with Fine-grained Rewards</b></i>, Hwang et al., <a href="https://aclanthology.org/2024.findings-emnlp.78/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.11-blue" alt="PDF Badge"></a></li>
<li><i><b>OpenRFT: Adapting Reasoning Foundation Model for Domain-specific Tasks with Reinforcement Fine-Tuning</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2412.16849" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Proposing and solving olympiad geometry with guided tree search</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2412.10673" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Enhancing Reasoning through Process Supervision with Monte Carlo Tree Search</b></i>, Li et al., <a href="https://openreview.net/forum?id=OupEEi1341" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Policy Guided Tree Search for Enhanced LLM Reasoning</b></i>, Li et al., <a href="https://arxiv.org/abs/2502.06813" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Self-Improvement Towards Pareto Optimality: Mitigating Preference Conflicts in Multi-Objective Alignment</b></i>, Li et al., <a href="https://arxiv.org/abs/2502.14354" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>BOLT: Bootstrap Long Chain-of-Thought in Language Models without Distillation</b></i>, Pang et al., <a href="https://arxiv.org/abs/2502.03860" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Process-based Self-Rewarding Language Models</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2503.03746" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Entropy-Based Adaptive Weighting for Self-Training</b></i>, Wang et al., <a href="https://arxiv.org/abs/2503.23913" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Entropy-based Exploration Conduction for Multi-step Reasoning</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2503.15848" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>
<h2 id="feasible-reflection">3. Feasible Reflection</h2>


<h3 id="feedback">3.1 Feedback</h3>
<img src="./assets/images/feedback.png" style="width: 580pt">
<ul>
<li><i><b>When is Tree Search Useful for LLM Planning? It Depends on the Discriminator</b></i>, Chen et al., <a href="https://aclanthology.org/2024.acl-long.738/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>From generation to judgment: Opportunities and challenges of llm-as-a-judge</b></i>, Li et al., <a href="https://arxiv.org/abs/2411.16594" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>Search, Verify and Feedback: Towards Next Generation Post-training Paradigm of Foundation Models via Verifier Engineering</b></i>, Guan et al., <a href="https://arxiv.org/abs/2411.11504" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>Llms-as-judges: a comprehensive survey on llm-based evaluation methods</b></i>, Li et al., <a href="https://arxiv.org/abs/2412.05579" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>A Survey on Feedback-based Multi-step Reasoning for Large Language Models on Mathematics</b></i>, Wei et al., <a href="https://arxiv.org/abs/2502.14333" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="hybrid-feedbacks">3.1.1 Hybrid Feedbacks</h4>
</ul>

<ul>
<li><i><b>The lessons of developing process reward models in mathematical reasoning</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2501.07301" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Step-KTO: Optimizing Mathematical Reasoning through Stepwise Binary Feedback</b></i>, Lin et al., <a href="https://arxiv.org/abs/2501.10799" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="overall-feedback">3.1.2 Overall Feedback</h4>
</ul>

<b>Overall Feedback from Outcome Reward Model</b>
<ul>
<li><i><b>Training verifiers to solve math word problems</b></i>, Cobbe et al., <a href="https://arxiv.org/abs/2110.14168" target="_blank"><img src="https://img.shields.io/badge/arXiv-2021.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Towards Mitigating LLM Hallucination via Self Reflection</b></i>, Ji et al., <a href="https://aclanthology.org/2023.findings-emnlp.123/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Deepseekmath: Pushing the limits of mathematical reasoning in open language models</b></i>, Shao et al., <a href="https://arxiv.org/abs/2402.03300" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Generative verifiers: Reward modeling as next-token prediction</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2408.15240" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.08-red" alt="arXiv Badge"></a></li>
<li><i><b>Self-generated critiques boost reward modeling for language models</b></i>, Yu et al., <a href="https://arxiv.org/abs/2411.16646" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>Self-Consistency of the Internal Reward Models Improves Self-Rewarding Language Models</b></i>, Zhou et al., <a href="https://arxiv.org/abs/2502.08922" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>

<b>Overall Feedback from RLLMs</b>
<ul>
<li><i><b>Self-critiquing models for assisting human evaluators</b></i>, Saunders et al., <a href="https://arxiv.org/abs/2206.05802" target="_blank"><img src="https://img.shields.io/badge/arXiv-2022.06-red" alt="arXiv Badge"></a></li>
<li><i><b>Language models (mostly) know what they know</b></i>, Kadavath et al., <a href="https://arxiv.org/abs/2207.05221" target="_blank"><img src="https://img.shields.io/badge/arXiv-2022.07-red" alt="arXiv Badge"></a></li>
<li><i><b>Constitutional AI: Harmlessness from AI Feedback</b></i>, Bai et al., <a href="https://arxiv.org/abs/2212.08073" target="_blank"><img src="https://img.shields.io/badge/arXiv-2022.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Contrastive learning with logic-driven data augmentation for logical reasoning over text</b></i>, Bao et al., <a href="https://arxiv.org/abs/2305.12599" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.05-red" alt="arXiv Badge"></a></li>
<li><i><b>Self-verification improves few-shot clinical information extraction</b></i>, Gero et al., <a href="https://openreview.net/forum?id=SBbJICrglS" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.06-blue" alt="PDF Badge"></a></li>
<li><i><b>Shepherd: A critic for language model generation</b></i>, Wang et al., <a href="https://arxiv.org/abs/2308.04592" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.08-red" alt="arXiv Badge"></a></li>
<li><i><b>Large Language Models are Better Reasoners with Self-Verification</b></i>, Weng et al., <a href="https://aclanthology.org/2023.findings-emnlp.167/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Large Language Models Cannot Self-Correct Reasoning Yet</b></i>, Huang et al., <a href="https://openreview.net/forum?id=IkmD3fKBPQ" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.01-blue" alt="PDF Badge"></a></li>
<li><i><b>SelfCheck: Using LLMs to Zero-Shot Check Their Own Step-by-Step Reasoning</b></i>, Miao et al., <a href="https://openreview.net/forum?id=pTHfApDakA" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Self-reflection in llm agents: Effects on problem-solving performance</b></i>, Renze et al., <a href="https://arxiv.org/abs/2405.06682" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.05-red" alt="arXiv Badge"></a></li>
<li><i><b>Llm critics help catch llm bugs</b></i>, McAleese et al., <a href="https://arxiv.org/abs/2407.00215" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.07-red" alt="arXiv Badge"></a></li>
<li><i><b>LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models</b></i>, Hao et al., <a href="https://openreview.net/forum?id=b0y6fbSUG0" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.07-blue" alt="PDF Badge"></a></li>
<li><i><b>Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution</b></i>, Fernando et al., <a href="https://proceedings.mlr.press/v235/fernando24a.html" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.07-blue" alt="PDF Badge"></a></li>
<li><i><b>Self-Contrast: Better Reflection Through Inconsistent Solving Perspectives</b></i>, Zhang et al., <a href="https://aclanthology.org/2024.acl-long.197/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Critic-cot: Boosting the reasoning abilities of large language model via chain-of-thoughts critic</b></i>, Zheng et al., <a href="https://arxiv.org/abs/2408.16326" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.08-red" alt="arXiv Badge"></a></li>
<li><i><b>Small Language Models Need Strong Verifiers to Self-Correct Reasoning</b></i>, Zhang et al., <a href="https://aclanthology.org/2024.findings-acl.924/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Abstract Meaning Representation-Based Logic-Driven Data Augmentation for Logical Reasoning</b></i>, Bao et al., <a href="https://aclanthology.org/2024.findings-acl.353/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Reversal of Thought: Enhancing Large Language Models with Preference-Guided Reverse Reasoning Warm-up</b></i>, Yuan et al., <a href="https://arxiv.org/abs/2410.12323" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>What Makes Large Language Models Reason in (Multi-Turn) Code Generation?</b></i>, Zheng et al., <a href="https://openreview.net/forum?id=Zk9guOl9NS" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Attention-guided Self-reflection for Zero-shot Hallucination Detection in Large Language Models</b></i>, Liu et al., <a href="https://arxiv.org/abs/2501.09997" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge</b></i>, Saha et al., <a href="https://arxiv.org/abs/2501.18099" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems</b></i>, Ye et al., <a href="https://openreview.net/forum?id=zpDGwcmMV4" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Training an LLM-as-a-Judge Model: Pipeline, Insights, and Practical Lessons</b></i>, Hu et al., <a href="https://arxiv.org/abs/2502.02988" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>A Study on Leveraging Search and Self-Feedback for Agent Reasoning</b></i>, Yuan et al., <a href="https://arxiv.org/abs/2502.12094" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>RefineCoder: Iterative Improving of Large Language Models via Adaptive Critique Refinement for Code Generation</b></i>, Zhou et al., <a href="https://arxiv.org/abs/2502.09183" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>

<b>Overall Feedback from Rule Extraction</b>
<ul>
<li><i><b>Star: Bootstrapping reasoning with reasoning</b></i>, Zelikman et al., <a href="https://openreview.net/pdf?id=_3ELRdg2sgI" target="_blank"><img src="https://img.shields.io/badge/PDF-2022.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Critic: Large language models can self-correct with tool-interactive critiquing</b></i>, Gou et al., <a href="https://arxiv.org/abs/2305.11738" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.05-red" alt="arXiv Badge"></a></li>
<li><i><b>LEVER: Learning to Verify Language-to-Code Generation with Execution</b></i>, Ni et al., <a href="https://proceedings.mlr.press/v202/ni23b.html" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.07-blue" alt="PDF Badge"></a></li>
<li><i><b>Reinforced self-training (rest) for language modeling</b></i>, Gulcehre et al., <a href="https://arxiv.org/abs/2308.08998" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.08-red" alt="arXiv Badge"></a></li>
<li><i><b>Tree of Thoughts: Deliberate Problem Solving with Large Language Models</b></i>, Yao et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Reasoning with Language Model is Planning with World Model</b></i>, Hao et al., <a href="https://aclanthology.org/2023.emnlp-main.507/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-based Self-Verification</b></i>, Zhou et al., <a href="https://openreview.net/forum?id=c8McWs4Av0" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.01-blue" alt="PDF Badge"></a></li>
<li><i><b>VerMCTS: Synthesizing Multi-Step Programs using a Verifier, a Large Language Model, and Tree Search</b></i>, Brandfonbrener et al., <a href="https://arxiv.org/abs/2402.08147" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.02-red" alt="arXiv Badge"></a></li>
<li><i><b>ReFT: Reasoning with Reinforced Fine-Tuning</b></i>, Trung et al., <a href="https://aclanthology.org/2024.acl-long.410/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement</b></i>, Zheng et al., <a href="https://aclanthology.org/2024.findings-acl.762/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>On designing effective rl reward at training time for llm reasoning</b></i>, Gao et al., <a href="https://arxiv.org/abs/2410.15115" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>o1-coder: an o1 replication for coding</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2412.00154" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning</b></i>, Guo et al., <a href="https://arxiv.org/abs/2501.12948" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Dynamic Scaling of Unit Tests for Code Reward Modeling</b></i>, Ma et al., <a href="https://arxiv.org/abs/2501.01054" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning</b></i>, Xie et al., <a href="https://arxiv.org/abs/2502.14768" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>SoRFT: Issue Resolving with Subtask-oriented Reinforced Fine-Tuning</b></i>, Ma et al., <a href="https://arxiv.org/abs/2502.20127" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>ACECODER: Acing Coder RL via Automated Test-Case Synthesis</b></i>, Zeng et al., <a href="https://arxiv.org/abs/2502.01718" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation</b></i>, Baker et al., <a href="https://openai.com/index/chain-of-thought-monitoring/" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.03-blue" alt="PDF Badge"></a></li>
</ul>

<h4 id="process-feedback">3.1.3 Process Feedback</h4>
</ul>

<ul>
<li><i><b>Solving math word problems with process- and outcome-based feedback</b></i>, Uesato et al., <a href="https://arxiv.org/abs/2211.14275" target="_blank"><img src="https://img.shields.io/badge/arXiv-2022.11-red" alt="arXiv Badge"></a></li>
<li><i><b>Vineppo: Unlocking rl potential for llm reasoning through refined credit assignment</b></i>, Kazemnejad et al., <a href="https://arxiv.org/abs/2410.01679" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
</ul>

<b>Process Feedback from Process Rewarded Model</b>
<ul>
<li><i><b>Let's reward step by step: Step-Level reward model as the Navigators for Reasoning</b></i>, Ma et al., <a href="https://arxiv.org/abs/2310.10080" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Let's verify step by step</b></i>, Lightman et al., <a href="https://openreview.net/forum?id=v8L0pN6EOi" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models</b></i>, Hu et al., <a href="https://openreview.net/forum?id=ZWyLjimciT" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.03-blue" alt="PDF Badge"></a></li>
<li><i><b>OVM, Outcome-supervised Value Models for Planning in Mathematical Reasoning</b></i>, Yu et al., <a href="https://aclanthology.org/2024.findings-naacl.55/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.06-blue" alt="PDF Badge"></a></li>
<li><i><b>Token-Supervised Value Models for Enhancing Mathematical Reasoning Capabilities of Large Language Models</b></i>, Lee et al., <a href="https://arxiv.org/abs/2407.12863" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.07-red" alt="arXiv Badge"></a></li>
<li><i><b>Tlcr: Token-level continuous reward for fine-grained reinforcement learning from human feedback</b></i>, Yoon et al., <a href="https://arxiv.org/abs/2407.16574" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.07-red" alt="arXiv Badge"></a></li>
<li><i><b>Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations</b></i>, Wang et al., <a href="https://aclanthology.org/2024.acl-long.510/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Selective Preference Optimization via Token-Level Reward Function Estimation</b></i>, Yang et al., <a href="https://arxiv.org/abs/2408.13518" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.08-red" alt="arXiv Badge"></a></li>
<li><i><b>Step-level Value Preference Optimization for Mathematical Reasoning</b></i>, Chen et al., <a href="https://aclanthology.org/2024.findings-emnlp.463/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Skywork-o1 open series</b></i>, Team et al., <a href="https://huggingface.co/Skywork" target="_blank"><img src="https://img.shields.io/badge/Huggingface-2024.11-yellow" alt="Huggingface Badge"></a></li>
<li><i><b>Entropy-Regularized Process Reward Model</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2412.11006" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Hunyuanprover: A scalable data synthesis framework and guided tree search for automated theorem proving</b></i>, Li et al., <a href="https://arxiv.org/abs/2412.20735" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Acemath: Advancing frontier math reasoning with post-training and reward modeling</b></i>, Liu et al., <a href="https://arxiv.org/abs/2412.15084" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Free process rewards without process labels</b></i>, Yuan et al., <a href="https://arxiv.org/abs/2412.01981" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>AutoPSV: Automated Process-Supervised Verifier</b></i>, Lu et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/9246aa822579d9b29a140ecdac36ad60-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Scaling Autonomous Agents via Automatic Reward Modeling And Planning</b></i>, Chen et al., <a href="https://openreview.net/forum?id=womU9cEwcO" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Advancing LLM Reasoning Generalists with Preference Trees</b></i>, Yuan et al., <a href="https://openreview.net/forum?id=2ea5TNVR0c" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge</b></i>, Saha et al., <a href="https://arxiv.org/abs/2501.18099" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning</b></i>, Setlur et al., <a href="https://openreview.net/forum?id=A6Y7AqlzLW" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models</b></i>, Gu et al., <a href="https://arxiv.org/abs/2502.14272" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Distill Not Only Data but Also Rewards: Can Smaller Language Models Surpass Larger Ones?</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2502.19557" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Adaptivestep: Automatically dividing reasoning step through model confidence</b></i>, Liu et al., <a href="https://arxiv.org/abs/2502.13943" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Process Reward Models for LLM Agents: Practical Framework and Directions</b></i>, Choudhury et al., <a href="https://arxiv.org/abs/2502.10325" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Process reinforcement through implicit rewards</b></i>, Cui et al., <a href="https://arxiv.org/abs/2502.01456" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Full-Step-DPO: Self-Supervised Preference Optimization with Step-wise Rewards for Mathematical Reasoning</b></i>, Xu et al., <a href="https://arxiv.org/abs/2502.14356" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>VersaPRM: Multi-Domain Process Reward Model via Synthetic Reasoning Data</b></i>, Zeng et al., <a href="https://arxiv.org/abs/2502.06737" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Direct Value Optimization: Improving Chain-of-Thought Reasoning in LLMs with Refined Values</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2502.13723" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Teaching Language Models to Critique via Reinforcement Learning</b></i>, Xie et al., <a href="https://arxiv.org/abs/2502.03492" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Uncertainty-Aware Search and Value Models: Mitigating Search Scaling Flaws in LLMs</b></i>, Yu et al., <a href="https://arxiv.org/abs/2502.11155" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>AURORA: Automated Training Framework of Universal Process Reward Models via Ensemble Prompting and Reverse Verification</b></i>, Tan et al., <a href="https://arxiv.org/abs/2502.11520" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Visualprm: An effective process reward model for multimodal reasoning</b></i>, Wang et al., <a href="https://arxiv.org/abs/2503.10291" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>QwQ: Reflect Deeply on the Boundaries of the Unknown</b></i>, Team et al., <a href="https://qwenlm.github.io/blog/qwq-32b-preview/" target="_blank"><img src="https://img.shields.io/badge/Github-2025.11-white" alt="Github Badge"></a></li>
</ul>

<b>Process Feedback from RLLMs</b>
<ul>
<li><i><b>ReAct: Synergizing Reasoning and Acting in Language Models</b></i>, Yao et al., <a href="https://openreview.net/forum?id=WE_vluYUL-X" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.02-blue" alt="PDF Badge"></a></li>
<li><i><b>Reflexion: language agents with verbal reinforcement learning</b></i>, Shinn et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Can We Verify Step by Step for Incorrect Answer Detection?</b></i>, Xu et al., <a href="https://arxiv.org/abs/2402.10528" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Monte carlo tree search boosts reasoning via iterative preference learning</b></i>, Xie et al., <a href="https://arxiv.org/abs/2405.00451" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.05-red" alt="arXiv Badge"></a></li>
<li><i><b>Interactive Evolution: A Neural-Symbolic Self-Training Framework For Large Language Models</b></i>, Xu et al., <a href="https://arxiv.org/abs/2406.11736" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.06-red" alt="arXiv Badge"></a></li>
<li><i><b>Llm critics help catch bugs in mathematics: Towards a better mathematical verifier with natural language feedback</b></i>, Gao et al., <a href="https://arxiv.org/abs/2406.14024" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.06-red" alt="arXiv Badge"></a></li>
<li><i><b>Step-dpo: Step-wise preference optimization for long-chain reasoning of llms</b></i>, Lai et al., <a href="https://arxiv.org/abs/2406.18629" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.06-red" alt="arXiv Badge"></a></li>
<li><i><b>Reasoning in Flux: Enhancing Large Language Models Reasoning through Uncertainty-aware Adaptive Guidance</b></i>, Yin et al., <a href="https://aclanthology.org/2024.acl-long.131/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Advancing Process Verification for Large Language Models via Tree-Based Preference Learning</b></i>, He et al., <a href="https://aclanthology.org/2024.emnlp-main.125/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Outcome-Refining Process Supervision for Code Generation</b></i>, Yu et al., <a href="https://arxiv.org/abs/2412.15118" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Error Classification of Large Language Models on Math Word Problems: A Dynamically Adaptive Framework</b></i>, Sun et al., <a href="https://arxiv.org/abs/2501.15581" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Zero-Shot Verification-guided Chain of Thoughts</b></i>, Chowdhury et al., <a href="https://arxiv.org/abs/2501.13122" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Robotic Programmer: Video Instructed Policy Code Generation for Robotic Manipulation</b></i>, Xie et al., <a href="https://arxiv.org/abs/2501.04268" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Unveiling and Causalizing CoT: A Causal Pespective</b></i>, Fu et al., <a href="https://arxiv.org/abs/2502.18239" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Diverse Inference and Verification for Advanced Reasoning</b></i>, Drori et al., <a href="https://arxiv.org/abs/2502.09955" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Mathematical Reasoning in Large Language Models: Assessing Logical and Arithmetic Errors across Wide Numerical Ranges</b></i>, Shrestha et al., <a href="https://arxiv.org/abs/2502.08680" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Uncertainty-Aware Step-wise Verification with Generative Reward Models</b></i>, Ye et al., <a href="https://arxiv.org/abs/2502.11250" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>JudgeLRM: Large Reasoning Models as a Judge</b></i>, Chen et al., <a href="https://arxiv.org/abs/2504.00050" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.04-red" alt="arXiv Badge"></a></li>
</ul>



<h3 id="refinement">3.2 Refinement</h3>
<img src="./assets/images/refinement.png" style="width: 580pt">
<h4 id="prompt-based-refinement-generation">3.2.4 Prompt-based Refinement Generation</h4>
</ul>

<ul>
<li><i><b>Self-critiquing models for assisting human evaluators</b></i>, Saunders et al., <a href="https://arxiv.org/abs/2206.05802" target="_blank"><img src="https://img.shields.io/badge/arXiv-2022.06-red" alt="arXiv Badge"></a></li>
<li><i><b>Self-Refine: Iterative Refinement with Self-Feedback</b></i>, Madaan et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/91edff07232fb1b55a505a9e9f6c0ff3-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.03-blue" alt="PDF Badge"></a></li>
<li><i><b>Grace: Discriminator-guided chain-of-thought reasoning</b></i>, Khalifa et al., <a href="https://arxiv.org/abs/2305.14934" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.05-red" alt="arXiv Badge"></a></li>
<li><i><b>Automatically correcting large language models: Surveying the landscape of diverse self-correction strategies</b></i>, Pan et al., <a href="https://arxiv.org/abs/2308.03188" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.08-red" alt="arXiv Badge"></a></li>
<li><i><b>Reflexion: language agents with verbal reinforcement learning</b></i>, Shinn et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Towards Mitigating LLM Hallucination via Self Reflection</b></i>, Ji et al., <a href="https://aclanthology.org/2023.findings-emnlp.123/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>SelfCheck: Using LLMs to Zero-Shot Check Their Own Step-by-Step Reasoning</b></i>, Miao et al., <a href="https://openreview.net/forum?id=pTHfApDakA" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Learning to check: Unleashing potentials for self-correction in large language models</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2402.13035" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.02-red" alt="arXiv Badge"></a></li>
<li><i><b>REFINER: Reasoning Feedback on Intermediate Representations</b></i>, Paul et al., <a href="https://aclanthology.org/2024.eacl-long.67/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.03-blue" alt="PDF Badge"></a></li>
<li><i><b>GLoRe: When, Where, and How to Improve LLM Reasoning via Global and Local Refinements</b></i>, Havrilla et al., <a href="https://openreview.net/forum?id=LH6R06NxdB" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.05-blue" alt="PDF Badge"></a></li>
<li><i><b>Enhancing Zero-Shot Chain-of-Thought Reasoning in Large Language Models through Logic</b></i>, Zhao et al., <a href="https://aclanthology.org/2024.lrec-main.543/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.05-blue" alt="PDF Badge"></a></li>
<li><i><b>General purpose verification for chain of thought prompting</b></i>, Vacareanu et al., <a href="https://arxiv.org/abs/2405.00204" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.05-red" alt="arXiv Badge"></a></li>
<li><i><b>Large language models have intrinsic self-correction ability</b></i>, Liu et al., <a href="https://arxiv.org/abs/2406.15673" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.06-red" alt="arXiv Badge"></a></li>
<li><i><b>Progressive-Hint Prompting Improves Reasoning in Large Language Models</b></i>, Zheng et al., <a href="https://openreview.net/forum?id=UkFEs3ciz8" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.06-blue" alt="PDF Badge"></a></li>
<li><i><b>Accessing gpt-4 level mathematical olympiad solutions via monte carlo tree self-refine with llama-3 8b</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2406.07394" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.06-red" alt="arXiv Badge"></a></li>
<li><i><b>Toward Adaptive Reasoning in Large Language Models with Thought Rollback</b></i>, Chen et al., <a href="https://proceedings.mlr.press/v235/chen24y.html" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.07-blue" alt="PDF Badge"></a></li>
<li><i><b>CoT Rerailer: Enhancing the Reliability of Large Language Models in Complex Reasoning Tasks through Error Detection and Correction</b></i>, Wan et al., <a href="https://arxiv.org/abs/2408.13940" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.08-red" alt="arXiv Badge"></a></li>
<li><i><b>Pride and Prejudice: LLM Amplifies Self-Bias in Self-Refinement</b></i>, Xu et al., <a href="https://aclanthology.org/2024.acl-long.826/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>A Theoretical Understanding of Self-Correction through In-context Alignment</b></i>, Wang et al., <a href="https://openreview.net/forum?id=OtvNLTWYww" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search</b></i>, Zhang et al., <a href="https://openreview.net/forum?id=8rcFOqEud5" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Enhancing Mathematical Reasoning in LLMs by Stepwise Correction</b></i>, Wu et al., <a href="https://arxiv.org/abs/2410.12934" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>LLM Self-Correction with DeCRIM: Decompose, Critique, and Refine for Enhanced Following of Instructions with Multiple Constraints</b></i>, Ferraz et al., <a href="https://openreview.net/forum?id=RQ6Ff8lso0" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.10-blue" alt="PDF Badge"></a></li>
<li><i><b>Advancing Large Language Model Attribution through Self-Improving</b></i>, Huang et al., <a href="https://aclanthology.org/2024.emnlp-main.223/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Confidence vs Critique: A Decomposition of Self-Correction Capability for LLMs</b></i>, Yang et al., <a href="https://arxiv.org/abs/2412.19513" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>LLM2: Let Large Language Models Harness System 2 Reasoning</b></i>, Yang et al., <a href="https://arxiv.org/abs/2412.20372" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Understanding the Dark Side of LLMs' Intrinsic Self-Correction</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2412.14959" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Enhancing LLM Reasoning with Multi-Path Collaborative Reactive and Reflection agents</b></i>, He et al., <a href="https://arxiv.org/abs/2501.00430" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>BackMATH: Towards Backward Reasoning for Solving Math Problems Step by Step</b></i>, Zhang et al., <a href="https://aclanthology.org/2025.coling-industry.40/" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>ReARTeR: Retrieval-Augmented Reasoning with Trustworthy Process Rewarding</b></i>, Sun et al., <a href="https://arxiv.org/abs/2501.07861" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models</b></i>, Yang et al., <a href="https://arxiv.org/abs/2502.04404" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Optimizing generative AI by backpropagating language model feedback</b></i>, Yuksekgonul et al., <a href="https://www.nature.com/articles/s41586-025-08661-4" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.03-blue" alt="PDF Badge"></a></li>
<li><i><b>DLPO: Towards a Robust, Efficient, and Generalizable Prompt Optimization Framework from a Deep-Learning Perspective</b></i>, Peng et al., <a href="https://arxiv.org/abs/2503.13413" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Instruct-of-Reflection: Enhancing Large Language Models Iterative Reflection Capabilities via Dynamic-Meta Instruction</b></i>, Liu et al., <a href="https://arxiv.org/abs/2503.00902" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement</b></i>, Yang et al., <a href="https://arxiv.org/abs/2503.16024" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="rl-based-refinement-learning">3.2.5 RL-based Refinement Learning</h4>
</ul>

<ul>
<li><i><b>Training language models to self-correct via reinforcement learning</b></i>, Kumar et al., <a href="https://arxiv.org/abs/2409.12917" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.09-red" alt="arXiv Badge"></a></li>
<li><i><b>Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning</b></i>, Guo et al., <a href="https://arxiv.org/abs/2501.12948" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>7B Model and 8K Examples: Emerging Reasoning with Reinforcement Learning is Both Effective and Efficient</b></i>, Zeng et al., <a href="https://hkust-nlp.notion.site/simplerl-reason" target="_blank"><img src="https://img.shields.io/badge/Notion-2025.01-white" alt="Notion Badge"></a></li>
<li><i><b>S<sup>2</sup>R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning</b></i>, Ma et al., <a href="https://arxiv.org/abs/2502.12853" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates</b></i>, Yang et al., <a href="https://arxiv.org/abs/2502.06772" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification</b></i>, Lee et al., <a href="https://arxiv.org/abs/2502.14565" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>ARIES: Stimulating Self-Refinement of Large Language Models by Iterative Preference Optimization</b></i>, Zeng et al., <a href="https://arxiv.org/abs/2502.05605" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="sft-based-refinement-imitation">3.2.6 SFT-based Refinement Imitation</h4>
</ul>

<ul>
<li><i><b>Learning from mistakes makes llm better reasoner</b></i>, An et al., <a href="https://arxiv.org/abs/2310.20689" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Reflection-tuning: Data recycling improves llm instruction-tuning</b></i>, Li et al., <a href="https://arxiv.org/abs/2310.11716" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Teaching Large Language Models to Self-Debug</b></i>, Chen et al., <a href="https://openreview.net/forum?id=KuPixIqPiq" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Enhancing visual-language modality alignment in large vision language models via self-improvement</b></i>, Wang et al., <a href="https://arxiv.org/abs/2405.15973" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.05-red" alt="arXiv Badge"></a></li>
<li><i><b>Llm critics help catch bugs in mathematics: Towards a better mathematical verifier with natural language feedback</b></i>, Gao et al., <a href="https://arxiv.org/abs/2406.14024" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.06-red" alt="arXiv Badge"></a></li>
<li><i><b>Mutual reasoning makes smaller llms stronger problem-solvers</b></i>, Qi et al., <a href="https://arxiv.org/abs/2408.06195" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.08-red" alt="arXiv Badge"></a></li>
<li><i><b>S <sup>3</sup> c-Math: Spontaneous Step-level Self-correction Makes Large Language Models Better Mathematical Reasoners</b></i>, Yan et al., <a href="https://arxiv.org/abs/2409.01524" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.09-red" alt="arXiv Badge"></a></li>
<li><i><b>Recursive Introspection: Teaching Language Model Agents How to Self-Improve</b></i>, Qu et al., <a href="https://openreview.net/forum?id=DRC9pZwBwR" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>O1 Replication Journey: A Strategic Progress Report--Part 1</b></i>, Qin et al., <a href="https://arxiv.org/abs/2410.18982" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Enhancing llm reasoning via critique models with test-time and training-time supervision</b></i>, Xi et al., <a href="https://arxiv.org/abs/2411.16579" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>Vision-language models can self-improve reasoning via reflection</b></i>, Cheng et al., <a href="https://arxiv.org/abs/2411.00855" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2501.01668" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Critique fine-tuning: Learning to critique is more effective than learning to imitate</b></i>, Wang et al., <a href="https://arxiv.org/abs/2501.17703" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>RealCritic: Towards Effectiveness-Driven Evaluation of Language Model Critiques</b></i>, Tang et al., <a href="https://arxiv.org/abs/2501.14492" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>ProgCo: Program Helps Self-Correction of Large Language Models</b></i>, Song et al., <a href="https://arxiv.org/abs/2501.01264" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>URSA: Understanding and Verifying Chain-of-thought Reasoning in Multimodal Mathematics</b></i>, Luo et al., <a href="https://arxiv.org/abs/2501.04686" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Iterative Deepening Sampling for Large Language Models</b></i>, Chen et al., <a href="https://arxiv.org/abs/2502.05449" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>LLMs Can Easily Learn to Reason from Demonstrations Structure, not content, is what matters!</b></i>, Li et al., <a href="https://arxiv.org/abs/2502.07374" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification</b></i>, Sun et al., <a href="https://arxiv.org/abs/2502.13383" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>
<h2 id="extensive-exploration">4. Extensive Exploration</h2>
<ul>
<li><i><b>Improving Policies via Search in Cooperative Partially Observable Games</b></i>, Lerer et al., <a href="https://ojs.aaai.org/index.php/AAAI/article/view/6208" target="_blank"><img src="https://img.shields.io/badge/PDF-2020.04-blue" alt="PDF Badge"></a></li>
<li><i><b>On The Planning Abilities of OpenAI's o1 Models: Feasibility, Optimality, and Generalizability</b></i>, Wang et al., <a href="https://arxiv.org/abs/2409.19924" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.09-red" alt="arXiv Badge"></a></li>
<li><i><b>LLMs Still Can't Plan; Can LRMs? A Preliminary Evaluation of OpenAI's o1 on PlanBench</b></i>, Valmeekam et al., <a href="https://openreview.net/forum?id=Gcr1Lx4Koz" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.10-blue" alt="PDF Badge"></a></li>
<li><i><b>Scaling of search and learning: A roadmap to reproduce o1 from reinforcement learning perspective</b></i>, Zeng et al., <a href="https://arxiv.org/abs/2412.14135" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning</b></i>, Guo et al., <a href="https://arxiv.org/abs/2501.12948" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
</ul>



<h3 id="exploration-scaling">4.1 Exploration Scaling</h3>
<img src="./assets/images/exploration-scaling.png" style="width: 580pt">
</ul>

<ul>
<li><i><b>Greedy Policy Search: A Simple Baseline for Learnable Test-Time Augmentation</b></i>, Lyzhov et al., <a href="https://proceedings.mlr.press/v124/lyzhov20a.html" target="_blank"><img src="https://img.shields.io/badge/PDF-2020.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Scaling scaling laws with board games</b></i>, Jones et al., <a href="https://arxiv.org/abs/2104.03113" target="_blank"><img src="https://img.shields.io/badge/arXiv-2021.04-red" alt="arXiv Badge"></a></li>
<li><i><b>Large language monkeys: Scaling inference compute with repeated sampling</b></i>, Brown et al., <a href="https://arxiv.org/abs/2407.21787" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.07-red" alt="arXiv Badge"></a></li>
<li><i><b>From Decoding to Meta-Generation: Inference-time Algorithms for Large Language Models</b></i>, Welleck et al., <a href="https://openreview.net/forum?id=eskQMcIbMS" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.11-blue" alt="PDF Badge"></a></li>
<li><i><b>From medprompt to o1: Exploration of run-time strategies for medical challenge problems and beyond</b></i>, Nori et al., <a href="https://arxiv.org/abs/2411.03590" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>ECM: A Unified Electronic Circuit Model for Explaining the Emergence of In-Context Learning and Chain-of-Thought in Large Language Model</b></i>, Chen et al., <a href="https://arxiv.org/abs/2502.03325" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>(Mis) Fitting: A Survey of Scaling Laws</b></i>, Li et al., <a href="https://arxiv.org/abs/2502.18969" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="parallel-scaling">4.1.1 Parallel Scaling</h4>
</ul>

<ul>
<li><i><b>Self-Consistency Improves Chain of Thought Reasoning in Language Models</b></i>, Wang et al., <a href="https://openreview.net/forum?id=1PL1NIMMrw" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.02-blue" alt="PDF Badge"></a></li>
<li><i><b>Large language monkeys: Scaling inference compute with repeated sampling</b></i>, Brown et al., <a href="https://arxiv.org/abs/2407.21787" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.07-red" alt="arXiv Badge"></a></li>
<li><i><b>Inference scaling laws: An empirical analysis of compute-optimal inference for problem-solving with language models</b></i>, Wu et al., <a href="https://arxiv.org/abs/2408.00724" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.08-red" alt="arXiv Badge"></a></li>
<li><i><b>Inference Scaling vs Reasoning: An Empirical Analysis of Compute-Optimal LLM Problem-Solving</b></i>, AbdElhameed et al., <a href="https://arxiv.org/abs/2412.16260" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Metascale: Test-time scaling with evolving meta-thoughts</b></i>, Liu et al., <a href="https://arxiv.org/abs/2503.13447" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>

<b>Sampling Optimization</b>
<ul>
<li><i><b>Cross-lingual Prompting: Improving Zero-shot Chain-of-Thought Reasoning across Languages</b></i>, Qin et al., <a href="https://aclanthology.org/2023.emnlp-main.163/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Scaling llm inference with optimized sample compute allocation</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2410.22480" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Planning in Natural Language Improves LLM Search for Code Generation</b></i>, Wang et al., <a href="https://openreview.net/forum?id=B2iSfPNj49" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.10-blue" alt="PDF Badge"></a></li>
<li><i><b>Python is Not Always the Best Choice: Embracing Multilingual Program of Thoughts</b></i>, Luo et al., <a href="https://aclanthology.org/2024.emnlp-main.408/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective</b></i>, Yu et al., <a href="https://arxiv.org/abs/2501.11110" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities?</b></i>, Zeng et al., <a href="https://arxiv.org/abs/2502.12215" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Optimizing Temperature for Language Models with Multi-Sample Inference</b></i>, Du et al., <a href="https://arxiv.org/abs/2502.05234" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Bag of Tricks for Inference-time Computation of LLM Reasoning</b></i>, Liu et al., <a href="https://arxiv.org/abs/2502.07191" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning</b></i>, Yang et al., <a href="https://arxiv.org/abs/2502.18080" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Is Depth All You Need? An Exploration of Iterative Reasoning in LLMs</b></i>, Wu et al., <a href="https://arxiv.org/abs/2502.10858" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Do We Truly Need So Many Samples? Multi-LLM Repeated Sampling Efficiently Scale Test-Time Compute</b></i>, Chen et al., <a href="https://arxiv.org/abs/2504.00762" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.04-red" alt="arXiv Badge"></a></li>
</ul>

<b>Verification Optimization</b>
<ul>
<li><i><b>Show Your Work: Scratchpads for Intermediate Computation with Language Models</b></i>, Nye et al., <a href="https://openreview.net/forum?id=HBlx2idbkbq" target="_blank"><img src="https://img.shields.io/badge/PDF-2022.03-blue" alt="PDF Badge"></a></li>
<li><i><b>Making Language Models Better Reasoners with Step-Aware Verifier</b></i>, Li et al., <a href="https://aclanthology.org/2023.acl-long.291/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.07-blue" alt="PDF Badge"></a></li>
<li><i><b>Deductive Verification of Chain-of-Thought Reasoning</b></i>, Ling et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/72393bd47a35f5b3bee4c609e7bba733-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Don't Trust: Verify -- Grounding LLM Quantitative Reasoning with Autoformalization</b></i>, Zhou et al., <a href="https://openreview.net/forum?id=V5tdi14ple" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Multi-step problem solving through a verifier: An empirical analysis on model-induced process supervision</b></i>, Wang et al., <a href="https://arxiv.org/abs/2402.02658" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Stepwise self-consistent mathematical reasoning with large language models</b></i>, Zhao et al., <a href="https://arxiv.org/abs/2402.17786" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.02-red" alt="arXiv Badge"></a></li>
<li><i><b>General purpose verification for chain of thought prompting</b></i>, Vacareanu et al., <a href="https://arxiv.org/abs/2405.00204" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.05-red" alt="arXiv Badge"></a></li>
<li><i><b>Improve Mathematical Reasoning in Language Models by Automated Process Supervision</b></i>, Luo et al., <a href="https://arxiv.org/abs/2406.06592" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.06-red" alt="arXiv Badge"></a></li>
<li><i><b>Scaling llm test-time compute optimally can be more effective than scaling model parameters</b></i>, Snell et al., <a href="https://arxiv.org/abs/2408.03314" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.08-red" alt="arXiv Badge"></a></li>
<li><i><b>Inference scaling laws: An empirical analysis of compute-optimal inference for problem-solving with language models</b></i>, Wu et al., <a href="https://arxiv.org/abs/2408.00724" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.08-red" alt="arXiv Badge"></a></li>
<li><i><b>Learning to Reason via Program Generation, Emulation, and Search</b></i>, Weir et al., <a href="https://openreview.net/forum?id=te6VagJf6G" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>What are the essential factors in crafting effective long context multi-hop instruction datasets? insights and best practices</b></i>, Chen et al., <a href="https://arxiv.org/abs/2409.01893" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.09-red" alt="arXiv Badge"></a></li>
<li><i><b>MAgICoRe: Multi-Agent, Iterative, Coarse-to-Fine Refinement for Reasoning</b></i>, Chen et al., <a href="https://arxiv.org/abs/2409.12147" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.09-red" alt="arXiv Badge"></a></li>
<li><i><b>Rlef: Grounding code llms in execution feedback with reinforcement learning</b></i>, Gehring et al., <a href="https://arxiv.org/abs/2410.02089" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Beyond examples: High-level automated reasoning paradigm in in-context learning via mcts</b></i>, Wu et al., <a href="https://arxiv.org/abs/2411.18478" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>Wrong-of-Thought: An Integrated Reasoning Framework with Multi-Perspective Verification and Wrong Information</b></i>, Zhang et al., <a href="https://aclanthology.org/2024.findings-emnlp.388/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.11-blue" alt="PDF Badge"></a></li>
<li><i><b>A simple and provable scaling law for the test-time compute of large language models</b></i>, Chen et al., <a href="https://arxiv.org/abs/2411.19477" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>Lachesis: Predicting LLM Inference Accuracy using Structural Properties of Reasoning Paths</b></i>, Kim et al., <a href="https://arxiv.org/abs/2412.08281" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Seed-cts: Unleashing the power of tree search for superior performance in competitive coding tasks</b></i>, Wang et al., <a href="https://arxiv.org/abs/2412.12544" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>From Drafts to Answers: Unlocking LLM Potential via Aggregation Fine-Tuning</b></i>, Li et al., <a href="https://arxiv.org/abs/2501.11877" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>SETS: Leveraging Self-Verification and Self-Correction for Improved Test-Time Scaling</b></i>, Chen et al., <a href="https://arxiv.org/abs/2501.19306" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Instantiation-based Formalization of Logical Reasoning Tasks using Language Models and Logical Solvers</b></i>, Raza et al., <a href="https://arxiv.org/abs/2501.16961" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>The lessons of developing process reward models in mathematical reasoning</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2501.07301" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>ExACT: Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning</b></i>, Yu et al., <a href="https://openreview.net/forum?id=GBIUbwW9D8" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Scalable Best-of-N Selection for Large Language Models via Self-Certainty</b></i>, Kang et al., <a href="https://arxiv.org/abs/2502.18581" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling</b></i>, Liu et al., <a href="https://arxiv.org/abs/2502.06703" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>ECM: A Unified Electronic Circuit Model for Explaining the Emergence of In-Context Learning and Chain-of-Thought in Large Language Model</b></i>, Chen et al., <a href="https://arxiv.org/abs/2502.03325" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification</b></i>, Zhao et al., <a href="https://arxiv.org/abs/2502.01839" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>TestNUC: Enhancing Test-Time Computing Approaches through Neighboring Unlabeled Data Consistency</b></i>, Zou et al., <a href="https://arxiv.org/abs/2502.19163" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Confidence Improves Self-Consistency in LLMs</b></i>, Taubenfeld et al., <a href="https://arxiv.org/abs/2502.06233" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>S*: Test Time Scaling for Code Generation</b></i>, Li et al., <a href="https://arxiv.org/abs/2502.14382" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Bridging Internal Probability and Self-Consistency for Effective and Efficient LLM Reasoning</b></i>, Zhou et al., <a href="https://arxiv.org/abs/2502.00511" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Multidimensional Consistency Improves Reasoning in Language Models</b></i>, Lai et al., <a href="https://arxiv.org/abs/2503.02670" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Efficient test-time scaling via self-calibration</b></i>, Huang et al., <a href="https://arxiv.org/abs/2503.00031" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="vertical-scaling">4.1.2 Vertical Scaling</h4>
</ul>

<ul>
<li><i><b>Complexity-Based Prompting for Multi-step Reasoning</b></i>, Fu et al., <a href="https://openreview.net/forum?id=yf1icZHC-l9" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.02-blue" alt="PDF Badge"></a></li>
<li><i><b>Openai o1 system card</b></i>, Jaech et al., <a href="https://arxiv.org/abs/2412.16720" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>s1: Simple test-time scaling</b></i>, Muennighoff et al., <a href="https://arxiv.org/abs/2501.19393" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Test-time Computing: from System-1 Thinking to System-2 Thinking</b></i>, Ji et al., <a href="https://arxiv.org/abs/2501.02497" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach</b></i>, Geiping et al., <a href="https://arxiv.org/abs/2502.05171" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking</b></i>, Chen et al., <a href="https://arxiv.org/abs/2502.13842" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling</b></i>, Li et al., <a href="https://arxiv.org/abs/2502.17651" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Reasoning-as-Logic-Units: Scaling Test-Time Reasoning in Large Language Models Through Logic Unit Alignment</b></i>, Li et al., <a href="https://arxiv.org/abs/2502.07803" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Think Twice: Enhancing LLM Reasoning by Scaling Multi-round Test-time Thinking</b></i>, Tian et al., <a href="https://arxiv.org/abs/2503.19855" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>What, How, Where, and How Well? A Survey on Test-Time Scaling in Large Language Models</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2503.24235" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>



<h3 id="external-exploration">4.2 External Exploration</h3>
<img src="./assets/images/external-exploration.png" style="width: 580pt">
<h4 id="human-driven-exploration">4.2.3 Human-driven Exploration</h4>
</ul>

<ul>
<li><i><b>Least-to-Most Prompting Enables Complex Reasoning in Large Language Models</b></i>, Zhou et al., <a href="https://openreview.net/forum?id=WZH7099tgfM" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.02-blue" alt="PDF Badge"></a></li>
<li><i><b>Tree of Thoughts: Deliberate Problem Solving with Large Language Models</b></i>, Yao et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.09-blue" alt="PDF Badge"></a></li>
<li><i><b>PATHFINDER: Guided Search over Multi-Step Reasoning Paths</b></i>, Golovneva et al., <a href="https://openreview.net/forum?id=5TsfEEwRsu" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Demystifying chains, trees, and graphs of thoughts</b></i>, Besta et al., <a href="https://arxiv.org/abs/2401.14295" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Graph of Thoughts: Solving Elaborate Problems with Large Language Models</b></i>, Besta et al., <a href="https://ojs.aaai.org/index.php/AAAI/article/view/29720" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.03-blue" alt="PDF Badge"></a></li>
<li><i><b>Tree of Uncertain Thoughts Reasoning for Large Language Models</b></i>, Mo et al., <a href="https://ieeexplore.ieee.org/document/10448355" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.04-blue" alt="PDF Badge"></a></li>
<li><i><b>GraphReason: Enhancing Reasoning Capabilities of Large Language Models through A Graph-Based Verification Approach</b></i>, Cao et al., <a href="https://aclanthology.org/2024.nlrse-1.1/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing</b></i>, Tian et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/5e5853f35164e434015716a8c2a66543-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>On the diagram of thought</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2409.10038" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.09-red" alt="arXiv Badge"></a></li>
<li><i><b>Understanding When Tree of Thoughts Succeeds: Larger Models Excel in Generation, Not Discrimination</b></i>, Chen et al., <a href="https://arxiv.org/abs/2410.17820" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Treebon: Enhancing inference-time alignment with speculative tree-search and best-of-n sampling</b></i>, Qiu et al., <a href="https://arxiv.org/abs/2410.16033" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Scattered Forest Search: Smarter Code Space Exploration with LLMs</b></i>, Light et al., <a href="https://arxiv.org/abs/2411.05010" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>On the Empirical Complexity of Reasoning and Planning in LLMs</b></i>, Kang et al., <a href="https://aclanthology.org/2024.findings-emnlp.164/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.11-blue" alt="PDF Badge"></a></li>
<li><i><b>CodeTree: Agent-guided Tree Search for Code Generation with Large Language Models</b></i>, Li et al., <a href="https://arxiv.org/abs/2411.04329" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>SPaR: Self-Play with Tree-Search Refinement to Improve Instruction-Following in Large Language Models</b></i>, Cheng et al., <a href="https://arxiv.org/abs/2412.11605" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Forest-of-thought: Scaling test-time compute for enhancing LLM reasoning</b></i>, Bi et al., <a href="https://arxiv.org/abs/2412.09078" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Tree-of-Code: A Tree-Structured Exploring Framework for End-to-End Code Generation and Execution in Complex Task Handling</b></i>, Ni et al., <a href="https://arxiv.org/abs/2412.15305" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>A Roadmap to Guide the Integration of LLMs in Hierarchical Planning</b></i>, Puerta-Merino et al., <a href="https://arxiv.org/abs/2501.08068" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Atom of Thoughts for Markov LLM Test-Time Scaling</b></i>, Teng et al., <a href="https://arxiv.org/abs/2502.12018" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>START: Self-taught Reasoner with Tools</b></i>, Li et al., <a href="https://arxiv.org/abs/2503.04625" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="model-driven-exploration">4.2.4 Model-driven Exploration</h4>
</ul>

<b>Enhancing Exploration Logics</b>
<ul>
<li><i><b>Self-Evaluation Guided Beam Search for Reasoning</b></i>, Xie et al., <a href="https://openreview.net/forum?id=Bw82hwg5Q3" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.09-blue" alt="PDF Badge"></a></li>
<li><i><b>No train still gain. unleash mathematical reasoning of large language models with monte carlo tree search guided by energy function</b></i>, Xu et al., <a href="https://arxiv.org/abs/2309.03224" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.09-red" alt="arXiv Badge"></a></li>
<li><i><b>Mindstar: Enhancing math reasoning in pre-trained llms at inference time</b></i>, Kang et al., <a href="https://arxiv.org/abs/2405.16265" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.05-red" alt="arXiv Badge"></a></li>
<li><i><b>Strategist: Learning Strategic Skills by LLMs via Bi-Level Tree Search</b></i>, Light et al., <a href="https://openreview.net/forum?id=UHWBmZuJPF" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.06-blue" alt="PDF Badge"></a></li>
<li><i><b>Deductive Beam Search: Decoding Deducible Rationale for Chain-of-Thought Reasoning</b></i>, Zhu et al., <a href="https://openreview.net/forum?id=S1XnUsqwr7" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.07-blue" alt="PDF Badge"></a></li>
<li><i><b>Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping</b></i>, Lehnert et al., <a href="https://openreview.net/forum?id=SGoVIC0u0f" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.07-blue" alt="PDF Badge"></a></li>
<li><i><b>Tree search for language model agents</b></i>, Koh et al., <a href="https://arxiv.org/abs/2407.01476" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.07-red" alt="arXiv Badge"></a></li>
<li><i><b>Agent q: Advanced reasoning and learning for autonomous ai agents</b></i>, Putta et al., <a href="https://arxiv.org/abs/2408.07199" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.08-red" alt="arXiv Badge"></a></li>
<li><i><b>RethinkMCTS: Refining Erroneous Thoughts in Monte Carlo Tree Search for Code Generation</b></i>, Li et al., <a href="https://arxiv.org/abs/2409.09584" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.09-red" alt="arXiv Badge"></a></li>
<li><i><b>Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning</b></i>, Zhai et al., <a href="https://openreview.net/forum?id=nBjmMF2IZU" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Aflow: Automating agentic workflow generation</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2410.10762" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Cooperative Strategic Planning Enhances Reasoning Capabilities in Large Language Models</b></i>, Wang et al., <a href="https://arxiv.org/abs/2410.20007" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Deliberate reasoning for llms as structure-aware planning with accurate world model</b></i>, Xiong et al., <a href="https://arxiv.org/abs/2410.03136" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Marco-o1: Towards open reasoning models for open-ended solutions</b></i>, Zhao et al., <a href="https://arxiv.org/abs/2411.14405" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>Technical report: Enhancing llm reasoning with reward-guided tree search</b></i>, Jiang et al., <a href="https://arxiv.org/abs/2411.11694" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>SRA-MCTS: Self-driven Reasoning Aurmentation with Monte Carlo Tree Search for Enhanced Code Generation</b></i>, Xu et al., <a href="https://arxiv.org/abs/2411.11053" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>GPT-Guided Monte Carlo Tree Search for Symbolic Regression in Financial Fraud Detection</b></i>, Kadam et al., <a href="https://arxiv.org/abs/2411.04459" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>MC-NEST--Enhancing Mathematical Reasoning in Large Language Models with a Monte Carlo Nash Equilibrium Self-Refine Tree</b></i>, Rabby et al., <a href="https://arxiv.org/abs/2411.15645" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search</b></i>, Yao et al., <a href="https://arxiv.org/abs/2412.18319" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Towards Intrinsic Self-Correction Enhancement in Monte Carlo Tree Search Boosted Reasoning via Iterative Preference Learning</b></i>, Jiang et al., <a href="https://arxiv.org/abs/2412.17397" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Ensembling Large Language Models with Process Reward-Guided Tree Search for Better Complex Reasoning</b></i>, Park et al., <a href="https://arxiv.org/abs/2412.15797" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design</b></i>, Zheng et al., <a href="https://arxiv.org/abs/2501.08603" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Leveraging Constrained Monte Carlo Tree Search to Generate Reliable Long Chain-of-Thought for Mathematical Reasoning</b></i>, Lin et al., <a href="https://arxiv.org/abs/2502.11169" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Reasoning with Reinforced Functional Token Tuning</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2502.13389" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>CoAT: Chain-of-Associated-Thoughts Framework for Enhancing Large Language Models Reasoning</b></i>, Pan et al., <a href="https://arxiv.org/abs/2502.02390" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning</b></i>, Park et al., <a href="https://arxiv.org/abs/2502.18439" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Better Process Supervision with Bi-directional Rewarding Signals</b></i>, Chen et al., <a href="https://arxiv.org/abs/2503.04618" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>

<b>Exploration-Path Feedback</b>
<ul>
<li><i><b>Don't throw away your value model! Generating more preferable text with Value-Guided Monte-Carlo Tree Search decoding</b></i>, Liu et al., <a href="https://openreview.net/forum?id=kh9Zt2Ldmn" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.07-blue" alt="PDF Badge"></a></li>
<li><i><b>Making PPO even better: Value-Guided Monte-Carlo Tree Search decoding</b></i>, Liu et al., <a href="https://openreview.net/forum?id=QaODpeRaOK" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Llama-berry: Pairwise optimization for o1-like olympiad-level mathematical reasoning</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2410.02884" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>AtomThink: A Slow Thinking Framework for Multimodal Mathematical Reasoning</b></i>, Xiang et al., <a href="https://arxiv.org/abs/2411.11930" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods</b></i>, Puri et al., <a href="https://arxiv.org/abs/2502.01618" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>

<b>Unified Improvements</b>
<ul>
<li><i><b>Enhancing multi-step reasoning abilities of language models through direct q-function optimization</b></i>, Liu et al., <a href="https://arxiv.org/abs/2410.09302" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Process reward model with q-value rankings</b></i>, Li et al., <a href="https://arxiv.org/abs/2410.11287" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking</b></i>, Guan et al., <a href="https://arxiv.org/abs/2501.04519" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Evolving Deeper LLM Thinking</b></i>, Lee et al., <a href="https://arxiv.org/abs/2501.09891" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Hypothesis-Driven Theory-of-Mind Reasoning for Large Language Models</b></i>, Kim et al., <a href="https://arxiv.org/abs/2502.11881" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>SIFT: Grounding LLM Reasoning in Contexts via Stickers</b></i>, Zeng et al., <a href="https://arxiv.org/abs/2502.14922" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>QLASS: Boosting Language Agent Inference via Q-Guided Stepwise Search</b></i>, Lin et al., <a href="https://arxiv.org/abs/2502.02584" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>CritiQ: Mining Data Quality Criteria from Human Preferences</b></i>, Guo et al., <a href="https://arxiv.org/abs/2502.19279" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>



<h3 id="internal-exploration">4.3 Internal Exploration</h3>
<img src="./assets/images/internal-exploration.png" style="width: 580pt">
</ul>

<ul>
<li><i><b>RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold</b></i>, Setlur et al., <a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/4b77d5b896c321a29277524a98a50215-Paper-Conference.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Sft memorizes, rl generalizes: A comparative study of foundation model post-training</b></i>, Chu et al., <a href="https://arxiv.org/abs/2501.17161" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search</b></i>, Shen et al., <a href="https://arxiv.org/abs/2502.02508" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Demystifying Long Chain-of-Thought Reasoning in LLMs</b></i>, Yeo et al., <a href="https://arxiv.org/abs/2502.03373" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>LLM Post-Training: A Deep Dive into Reasoning Large Language Models</b></i>, Kumar et al., <a href="https://arxiv.org/abs/2502.21321" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="rl-strategies">4.3.5 RL Strategies</h4>
</ul>

<ul>
<li><i><b>Policy Gradient Methods for Reinforcement Learning with Function Approximation</b></i>, Sutton et al., <a href="https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-1999.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Proximal policy optimization algorithms</b></i>, Schulman et al., <a href="https://arxiv.org/abs/1707.06347" target="_blank"><img src="https://img.shields.io/badge/arXiv-2017.07-red" alt="arXiv Badge"></a></li>
<li><i><b>Deepseekmath: Pushing the limits of mathematical reasoning in open language models</b></i>, Shao et al., <a href="https://arxiv.org/abs/2402.03300" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.02-red" alt="arXiv Badge"></a></li>
<li><i><b>ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models</b></i>, Li et al., <a href="https://openreview.net/forum?id=Stn8hXkpe6" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.05-blue" alt="PDF Badge"></a></li>
<li><i><b>CPL: Critical Plan Step Learning Boosts LLM Generalization in Reasoning Tasks</b></i>, Wang et al., <a href="https://arxiv.org/abs/2409.08642" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.09-red" alt="arXiv Badge"></a></li>
<li><i><b>Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback</b></i>, Ivison et al., <a href="https://openreview.net/forum?id=JMBWTlazjW" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>A Small Step Towards Reproducing OpenAI o1: Progress Report on the Steiner Open Source Models</b></i>, Ji et al., <a href="https://medium.com/@peakji/b9a756a00855" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.10-blue" alt="PDF Badge"></a></li>
<li><i><b>A Comprehensive Survey of Direct Preference Optimization: Datasets, Theories, Variants, and Applications</b></i>, Xiao et al., <a href="https://arxiv.org/abs/2410.15595" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Critical Tokens Matter: Token-Level Contrastive Estimation Enhence LLM's Reasoning Capability</b></i>, Lin et al., <a href="https://arxiv.org/abs/2411.19943" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>Improving Multi-Step Reasoning Abilities of Large Language Models with Direct Advantage Policy Optimization</b></i>, Liu et al., <a href="https://arxiv.org/abs/2412.18279" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Offline Reinforcement Learning for LLM Multi-Step Reasoning</b></i>, Wang et al., <a href="https://arxiv.org/abs/2412.16145" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models</b></i>, Hu et al., <a href="https://arxiv.org/abs/2501.03262" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Diverse Preference Optimization</b></i>, Lanchantin et al., <a href="https://arxiv.org/abs/2501.18101" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>COS (M+ O) S: Curiosity and RL-Enhanced MCTS for Exploring Story Space via Language Models</b></i>, Materzok et al., <a href="https://arxiv.org/abs/2501.17104" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>7B Model and 8K Examples: Emerging Reasoning with Reinforcement Learning is Both Effective and Efficient</b></i>, Zeng et al., <a href="https://hkust-nlp.notion.site/simplerl-reason" target="_blank"><img src="https://img.shields.io/badge/Notion-2025.01-white" alt="Notion Badge"></a></li>
<li><i><b>LIMR: Less is More for RL Scaling</b></i>, Li et al., <a href="https://arxiv.org/abs/2502.11886" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Ignore the KL Penalty! Boosting Exploration on Critical Tokens to Enhance RL Fine-Tuning</b></i>, Vassoyan et al., <a href="https://arxiv.org/abs/2502.06533" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Lean and Mean: Decoupled Value Policy Optimization with Global Value Guidance</b></i>, Huang et al., <a href="https://arxiv.org/abs/2502.16944" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Process reinforcement through implicit rewards</b></i>, Cui et al., <a href="https://arxiv.org/abs/2502.01456" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>SPPD: Self-training with Process Preference Learning Using Dynamic Value Margin</b></i>, Yi et al., <a href="https://arxiv.org/abs/2502.13516" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Reward-aware Preference Optimization: A Unified Mathematical Framework for Model Alignment</b></i>, Sun et al., <a href="https://arxiv.org/abs/2502.00203" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Focused-DPO: Enhancing Code Generation Through Focused Preference Optimization on Error-Prone Points</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2502.11475" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Reasoning with Reinforced Functional Token Tuning</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2502.13389" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b></sup>Qsharp</sup>: Provably Optimal Distributional RL for LLM Post-Training</b></i>, Zhou et al., <a href="https://arxiv.org/abs/2502.20548" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Thinking Preference Optimization</b></i>, Yang et al., <a href="https://arxiv.org/abs/2502.13173" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training</b></i>, Bartoldson et al., <a href="https://arxiv.org/abs/2503.18929" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Dapo: An open-source llm reinforcement learning system at scale</b></i>, Yu et al., <a href="https://arxiv.org/abs/2503.14476" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model</b></i>, Hu et al., <a href="https://arxiv.org/abs/2503.24290" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Optimizing Test-Time Compute via Meta Reinforcement Finetuning</b></i>, Qu et al., <a href="https://openreview.net/forum?id=WGz4ytjo1h" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.03-blue" alt="PDF Badge"></a></li>
<li><i><b>Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't</b></i>, Dang et al., <a href="https://arxiv.org/abs/2503.16219" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks</b></i>, Zhou et al., <a href="https://arxiv.org/abs/2503.15478" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks</b></i>, YuYue et al., <a href="https://arxiv.org/abs/2504.05118" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.04-red" alt="arXiv Badge"></a></li>
</ul>

<h4 id="reward-strategies">4.3.6 Reward Strategies</h4>
</ul>

<b>Model-rewarded RL</b>
<ul>
<li><i><b>Training verifiers to solve math word problems</b></i>, Cobbe et al., <a href="https://arxiv.org/abs/2110.14168" target="_blank"><img src="https://img.shields.io/badge/arXiv-2021.10-red" alt="arXiv Badge"></a></li>
<li><i><b>AlphaZero-Like Tree-Search can Guide Large Language Model Decoding and Training</b></i>, Wan et al., <a href="https://openreview.net/forum?id=C4OpREezgj" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.05-blue" alt="PDF Badge"></a></li>
<li><i><b>ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search</b></i>, Zhang et al., <a href="https://openreview.net/forum?id=8rcFOqEud5" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling</b></i>, Hou et al., <a href="https://arxiv.org/abs/2501.11651" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Kimi k1. 5: Scaling reinforcement learning with llms</b></i>, Team et al., <a href="https://arxiv.org/abs/2501.12599" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL</b></i>, Luo et al., <a href="https://github.com/agentica-project/rllm" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.02-blue" alt="PDF Badge"></a></li>
<li><i><b>STeCa: Step-level Trajectory Calibration for LLM Agent Learning</b></i>, Wang et al., <a href="https://arxiv.org/abs/2502.14276" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Expanding RL with Verifiable Rewards Across Diverse Domains</b></i>, Su et al., <a href="https://arxiv.org/abs/2503.23829" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>R-PRM: Reasoning-Driven Process Reward Modeling</b></i>, She et al., <a href="https://arxiv.org/abs/2503.21295" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
</ul>

<b>Rule-rewarded RL</b>
<ul>
<li><i><b>Stepcoder: Improve code generation with reinforcement learning from compiler feedback</b></i>, Dou et al., <a href="https://arxiv.org/abs/2402.01391" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models</b></i>, Singh et al., <a href="https://openreview.net/pdf?id=lNAyUngGFK" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.04-blue" alt="PDF Badge"></a></li>
<li><i><b>Building math agents with multi-turn iterative preference learning</b></i>, Xiong et al., <a href="https://arxiv.org/abs/2409.02392" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.09-red" alt="arXiv Badge"></a></li>
<li><i><b>Unlocking the Capabilities of Thought: A Reasoning Boundary Framework to Quantify and Optimize Chain-of-Thought</b></i>, Chen et al., <a href="https://openreview.net/forum?id=pC44UMwy2v" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>o1-coder: an o1 replication for coding</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2412.00154" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning</b></i>, Guo et al., <a href="https://arxiv.org/abs/2501.12948" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Kimi k1. 5: Scaling reinforcement learning with llms</b></i>, Team et al., <a href="https://arxiv.org/abs/2501.12599" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning</b></i>, Xie et al., <a href="https://arxiv.org/abs/2502.14768" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning</b></i>, Yang et al., <a href="https://arxiv.org/abs/2502.18080" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Training Language Models to Reason Efficiently</b></i>, Arora et al., <a href="https://arxiv.org/abs/2502.04463" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Exploring the Limit of Outcome Reward for Learning Mathematical Reasoning</b></i>, Lyu et al., <a href="https://arxiv.org/abs/2502.06781" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Competitive Programming with Large Reasoning Models</b></i>, El-Kishky et al., <a href="https://arxiv.org/abs/2502.06807" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution</b></i>, Wei et al., <a href="https://arxiv.org/abs/2502.18449" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Inference-Time Computations for LLM Reasoning and Planning: A Benchmark and Insights</b></i>, Parashar et al., <a href="https://arxiv.org/abs/2502.12521" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Metastable Dynamics of Chain-of-Thought Reasoning: Provable Benefits of Search, RL and Distillation</b></i>, Kim et al., <a href="https://arxiv.org/abs/2502.01694" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>On the Emergence of Thinking in LLMs I: Searching for the Right Intuition</b></i>, Ye et al., <a href="https://arxiv.org/abs/2502.06773" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks</b></i>, Cuadron et al., <a href="https://arxiv.org/abs/2502.08235" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Z1: Efficient Test-time Scaling with Code</b></i>, Yu et al., <a href="https://arxiv.org/abs/2504.00810" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.04-red" alt="arXiv Badge"></a></li>
</ul>
<h2 id="future-and-frontiers">5. Future and Frontiers</h2>

<img src="./assets/images/future.jpg" style="width: 580pt">



<h3 id="agentic-embodied-long-cot">5.1 Agentic & Embodied Long CoT</h3>
<ul>
<li><i><b>Solving Math Word Problems via Cooperative Reasoning induced Language Models</b></i>, Zhu et al., <a href="https://aclanthology.org/2023.acl-long.245/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.07-blue" alt="PDF Badge"></a></li>
<li><i><b>Reasoning with Language Model is Planning with World Model</b></i>, Hao et al., <a href="https://aclanthology.org/2023.emnlp-main.507/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Large language models as commonsense knowledge for large-scale task planning</b></i>, Zhao et al., <a href="https://openreview.net/pdf?id=tED747HURfX" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Robotic Control via Embodied Chain-of-Thought Reasoning</b></i>, Zawalski et al., <a href="https://openreview.net/forum?id=S70MgnIA0v" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.00-blue" alt="PDF Badge"></a></li>
<li><i><b>Tree-Planner: Efficient Close-loop Task Planning with Large Language Models</b></i>, Hu et al., <a href="https://openreview.net/forum?id=Glcsog6zOe" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models</b></i>, Zhou et al., <a href="https://openreview.net/forum?id=njwv9BsGHF" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.05-blue" alt="PDF Badge"></a></li>
<li><i><b>Strategist: Learning Strategic Skills by LLMs via Bi-Level Tree Search</b></i>, Light et al., <a href="https://openreview.net/forum?id=UHWBmZuJPF" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.06-blue" alt="PDF Badge"></a></li>
<li><i><b>Mixture-of-agents enhances large language model capabilities</b></i>, Wang et al., <a href="https://arxiv.org/abs/2406.04692" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.06-red" alt="arXiv Badge"></a></li>
<li><i><b>ADaPT: As-Needed Decomposition and Planning with Language Models</b></i>, Prasad et al., <a href="https://aclanthology.org/2024.findings-naacl.264/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.06-blue" alt="PDF Badge"></a></li>
<li><i><b>Tree search for language model agents</b></i>, Koh et al., <a href="https://arxiv.org/abs/2407.01476" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.07-red" alt="arXiv Badge"></a></li>
<li><i><b>Hiagent: Hierarchical working memory management for solving long-horizon agent tasks with large language model</b></i>, Hu et al., <a href="https://arxiv.org/abs/2408.09559" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.08-red" alt="arXiv Badge"></a></li>
<li><i><b>S3 agent: Unlocking the power of VLLM for zero-shot multi-modal sarcasm detection</b></i>, Wang et al., <a href="https://dl.acm.org/doi/10.1145/3690642" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>MACM: Utilizing a Multi-Agent System for Condition Mining in Solving Complex Mathematical Problems</b></i>, Lei et al., <a href="https://openreview.net/forum?id=VR2RdSxtzs" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning</b></i>, Zhai et al., <a href="https://openreview.net/forum?id=nBjmMF2IZU" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>EVOLvE: Evaluating and Optimizing LLMs For Exploration</b></i>, Nie et al., <a href="https://arxiv.org/abs/2410.06238" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Agents Thinking Fast and Slow: A Talker-Reasoner Architecture</b></i>, Christakopoulou et al., <a href="https://openreview.net/forum?id=xPhcP6rbI4" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.10-blue" alt="PDF Badge"></a></li>
<li><i><b>Robotic Programmer: Video Instructed Policy Code Generation for Robotic Manipulation</b></i>, Xie et al., <a href="https://arxiv.org/abs/2501.04268" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Titans: Learning to memorize at test time</b></i>, Behrouz et al., <a href="https://arxiv.org/abs/2501.00663" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success</b></i>, Kim et al., <a href="https://arxiv.org/abs/2502.19645" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning</b></i>, Wang et al., <a href="https://arxiv.org/abs/2503.10480" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Embodied-Reasoner: Synergizing Visual Search, Reasoning, and Action for Embodied Interactive Tasks</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2503.21696" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Cosmos-reason1: From physical common sense to embodied reasoning</b></i>, Azzolini et al., <a href="https://arxiv.org/abs/2503.15558" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Improving Retrospective Language Agents via Joint Policy Gradient Optimization</b></i>, Feng et al., <a href="https://arxiv.org/abs/2503.01490" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Haste Makes Waste: Evaluating Planning Abilities of LLMs for Efficient and Feasible Multitasking with Time Constraints Between Actions</b></i>, Wu et al., <a href="https://arxiv.org/abs/2503.02238" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents</b></i>, Zhu et al., <a href="https://arxiv.org/abs/2503.01935" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>ReMA: Learning to Meta-think for LLMs with Multi-Agent Reinforcement Learning</b></i>, Wan et al., <a href="https://arxiv.org/abs/2503.09501" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>MAS-GPT: Training LLMs To Build LLM-Based Multi-Agent Systems</b></i>, Ye et al., <a href="https://openreview.net/forum?id=TqHoQIlumy" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.03-blue" alt="PDF Badge"></a></li>
<li><i><b>Advances and Challenges in Foundation Agents: From Brain-Inspired Intelligence to Evolutionary, Collaborative, and Safe Systems</b></i>, Liu et al., <a href="https://arxiv.org/abs/2504.01990" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.04-red" alt="arXiv Badge"></a></li>
</ul>



<h3 id="efficient-long-cot">5.2 Efficient Long CoT</h3>
</ul>

<ul>
<li><i><b>Guiding language model reasoning with planning tokens</b></i>, Wang et al., <a href="https://arxiv.org/abs/2310.05707" target="_blank"><img src="https://img.shields.io/badge/arXiv-2023.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Synergy-of-thoughts: Eliciting efficient reasoning in hybrid language models</b></i>, Shang et al., <a href="https://arxiv.org/abs/2402.02563" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Distilling system 2 into system 1</b></i>, Yu et al., <a href="https://arxiv.org/abs/2407.06023" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.07-red" alt="arXiv Badge"></a></li>
<li><i><b>Concise thoughts: Impact of output length on llm reasoning and cost</b></i>, Nayab et al., <a href="https://arxiv.org/abs/2407.19825" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.07-red" alt="arXiv Badge"></a></li>
<li><i><b>Litesearch: Efficacious tree search for llm</b></i>, Wang et al., <a href="https://arxiv.org/abs/2407.00320" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.07-red" alt="arXiv Badge"></a></li>
<li><i><b>Uncertainty-Guided Optimization on Large Language Model Search Trees</b></i>, Grosse et al., <a href="https://arxiv.org/abs/2407.03951" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.07-red" alt="arXiv Badge"></a></li>
<li><i><b>CPL: Critical Plan Step Learning Boosts LLM Generalization in Reasoning Tasks</b></i>, Wang et al., <a href="https://arxiv.org/abs/2409.08642" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.09-red" alt="arXiv Badge"></a></li>
<li><i><b>Unlocking the Capabilities of Thought: A Reasoning Boundary Framework to Quantify and Optimize Chain-of-Thought</b></i>, Chen et al., <a href="https://openreview.net/forum?id=pC44UMwy2v" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Kvsharer: Efficient inference via layer-wise dissimilar KV cache sharing</b></i>, Yang et al., <a href="https://arxiv.org/abs/2410.18517" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Interpretable contrastive monte carlo tree search reasoning</b></i>, Gao et al., <a href="https://arxiv.org/abs/2410.01707" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Dualformer: Controllable fast and slow thinking by learning with randomized reasoning traces</b></i>, Su et al., <a href="https://arxiv.org/abs/2410.09918" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>DynaThink: Fast or Slow? A Dynamic Decision-Making Framework for Large Language Models</b></i>, Pan et al., <a href="https://aclanthology.org/2024.emnlp-main.814/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Language models are hidden reasoners: Unlocking latent reasoning capabilities via self-rewarding</b></i>, Chen et al., <a href="https://arxiv.org/abs/2411.04282" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>Token-budget-aware llm reasoning</b></i>, Han et al., <a href="https://arxiv.org/abs/2412.18547" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>B-STaR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners</b></i>, Zeng et al., <a href="https://arxiv.org/abs/2412.17256" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>C3oT: Generating Shorter Chain-of-Thought without Compromising Effectiveness</b></i>, Kang et al., <a href="https://arxiv.org/abs/2412.11664" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Training large language models to reason in a continuous latent space</b></i>, Hao et al., <a href="https://arxiv.org/abs/2412.06769" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>CoMT: A Novel Benchmark for Chain of Multi-modal Thought on Large Vision-Language Models</b></i>, Cheng et al., <a href="https://arxiv.org/abs/2412.12932" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Kimi k1. 5: Scaling reinforcement learning with llms</b></i>, Team et al., <a href="https://arxiv.org/abs/2501.12599" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning</b></i>, Luo et al., <a href="https://arxiv.org/abs/2501.12570" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Reward-Guided Speculative Decoding for Efficient LLM Reasoning</b></i>, Liao et al., <a href="https://arxiv.org/abs/2501.19324" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Think Smarter not Harder: Adaptive Reasoning with Inference Aware Optimization</b></i>, Yu et al., <a href="https://arxiv.org/abs/2501.17974" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Efficient Reasoning with Hidden Thinking</b></i>, Shen et al., <a href="https://arxiv.org/abs/2501.19201" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>On the Query Complexity of Verifier-Assisted Language Generation</b></i>, Botta et al., <a href="https://arxiv.org/abs/2502.12123" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>TokenSkip: Controllable Chain-of-Thought Compression in LLMs</b></i>, Xia et al., <a href="https://arxiv.org/abs/2502.12067" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Boost, Disentangle, and Customize: A Robust System2-to-System1 Pipeline for Code Generation</b></i>, Du et al., <a href="https://arxiv.org/abs/2502.12492" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Jakiro: Boosting Speculative Decoding with Decoupled Multi-Head via MoE</b></i>, Huang et al., <a href="https://arxiv.org/abs/2502.06282" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Towards Reasoning Ability of Small Language Models</b></i>, Srivastava et al., <a href="https://arxiv.org/abs/2502.11569" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs</b></i>, Ji et al., <a href="https://arxiv.org/abs/2502.14837" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Portable Reward Tuning: Towards Reusable Fine-Tuning across Different Pretrained Models</b></i>, Chijiwa et al., <a href="https://arxiv.org/abs/2502.12776" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification</b></i>, Sun et al., <a href="https://arxiv.org/abs/2502.13383" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Language Models Can Predict Their Own Behavior</b></i>, Ashok et al., <a href="https://arxiv.org/abs/2502.13329" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>On the Convergence Rate of MCTS for the Optimal Value Estimation in Markov Decision Processes</b></i>, Chang et al., <a href="https://ieeexplore.ieee.org/document/10870057" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.02-blue" alt="PDF Badge"></a></li>
<li><i><b>CoT-Valve: Length-Compressible Chain-of-Thought Tuning</b></i>, Ma et al., <a href="https://arxiv.org/abs/2502.09601" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Training Language Models to Reason Efficiently</b></i>, Arora et al., <a href="https://arxiv.org/abs/2502.04463" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Chain of Draft: Thinking Faster by Writing Less</b></i>, Xu et al., <a href="https://arxiv.org/abs/2502.18600" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Learning to Stop Overthinking at Test Time</b></i>, Bao et al., <a href="https://arxiv.org/abs/2502.10954" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Self-Training Elicits Concise Reasoning in Large Language Models</b></i>, Munkhbat et al., <a href="https://arxiv.org/abs/2502.20122" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Length-Controlled Margin-Based Preference Optimization without Reference Model</b></i>, Li et al., <a href="https://arxiv.org/abs/2502.14643" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Reasoning on a Spectrum: Aligning LLMs to System 1 and System 2 Thinking</b></i>, Ziabari et al., <a href="https://arxiv.org/abs/2502.12470" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Dynamic Parallel Tree Search for Efficient LLM Reasoning</b></i>, Ding et al., <a href="https://arxiv.org/abs/2502.16235" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Stepwise Perplexity-Guided Refinement for Efficient Chain-of-Thought Reasoning in Large Language Models</b></i>, Cui et al., <a href="https://arxiv.org/abs/2502.13260" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Dynamic Chain-of-Thought: Towards Adaptive Deep Reasoning</b></i>, Wang et al., <a href="https://arxiv.org/abs/2502.10428" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs</b></i>, Xu et al., <a href="https://arxiv.org/abs/2502.12134" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>LightThinker: Thinking Step-by-Step Compression</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2502.15589" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning</b></i>, Yan et al., <a href="https://arxiv.org/abs/2502.02871" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Stepwise Informativeness Search for Improving LLM Reasoning</b></i>, Wang et al., <a href="https://arxiv.org/abs/2502.15335" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Adaptive Group Policy Optimization: Towards Stable Training and Token-Efficient Reasoning</b></i>, Li et al., <a href="https://arxiv.org/abs/2503.15952" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Innate Reasoning is Not Enough: In-Context Learning Enhances Reasoning Large Language Models with Less Overthinking</b></i>, Ge et al., <a href="https://arxiv.org/abs/2503.19602" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Understanding r1-zero-like training: A critical perspective</b></i>, Liu et al., <a href="https://arxiv.org/abs/2503.20783" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>The First Few Tokens Are All You Need: An Efficient and Effective Unsupervised Prefix Fine-Tuning Method for Reasoning Models</b></i>, Ji et al., <a href="https://arxiv.org/abs/2503.02875" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning</b></i>, Aggarwal et al., <a href="https://arxiv.org/abs/2503.04697" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>DAST: Difficulty-Adaptive Slow-Thinking for Large Reasoning Models</b></i>, Shen et al., <a href="https://arxiv.org/abs/2503.04472" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning</b></i>, Hou et al., <a href="https://arxiv.org/abs/2504.01296" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.04-red" alt="arXiv Badge"></a></li>
</ul>



<h3 id="knowledge-augmented-long-cot">5.3 Knowledge-Augmented Long CoT</h3>
</ul>

<ul>
<li><i><b>Best of Both Worlds: Harmonizing LLM Capabilities in Decision-Making and Question-Answering for Treatment Regimes</b></i>, Liu et al., <a href="https://openreview.net/forum?id=afu9qhp7md" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.00-blue" alt="PDF Badge"></a></li>
<li><i><b>Understanding Reasoning Ability of Language Models From the Perspective of Reasoning Paths Aggregation</b></i>, Wang et al., <a href="https://proceedings.mlr.press/v235/wang24a.html" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.07-blue" alt="PDF Badge"></a></li>
<li><i><b>Stream of search (sos): Learning to search in language</b></i>, Gandhi et al., <a href="https://openreview.net/pdf?id=2cop2jmQVL" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.07-blue" alt="PDF Badge"></a></li>
<li><i><b>CoPS: Empowering LLM Agents with Provable Cross-Task Experience Sharing</b></i>, Yang et al., <a href="https://arxiv.org/abs/2410.16670" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Disentangling memory and reasoning ability in large language models</b></i>, Jin et al., <a href="https://arxiv.org/abs/2411.13504" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>Huatuogpt-o1, towards medical complex reasoning with llms</b></i>, Chen et al., <a href="https://arxiv.org/abs/2412.18925" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>RAG-Star: Enhancing Deliberative Reasoning with Retrieval Augmented Verification and Refinement</b></i>, Jiang et al., <a href="https://arxiv.org/abs/2412.12881" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>O1 Replication Journey--Part 3: Inference-time Scaling for Medical Reasoning</b></i>, Huang et al., <a href="https://arxiv.org/abs/2501.06458" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>MedS <sup>3</sup>: Towards Medical Small Language Models with Self-Evolved Slow Thinking</b></i>, Jiang et al., <a href="https://arxiv.org/abs/2501.12051" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Search-o1: Agentic search-enhanced large reasoning models</b></i>, Li et al., <a href="https://arxiv.org/abs/2501.05366" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Chain-of-Retrieval Augmented Generation</b></i>, Wang et al., <a href="https://arxiv.org/abs/2501.14342" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Evaluating Large Language Models through Role-Guide and Self-Reflection: A Comparative Study</b></i>, Zhao et al., <a href="https://openreview.net/forum?id=E36NHwe7Zc" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Citrus: Leveraging Expert Cognitive Pathways in a Medical Language Model for Advanced Medical Decision Support</b></i>, Wang et al., <a href="https://arxiv.org/abs/2502.18274" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>ECM: A Unified Electronic Circuit Model for Explaining the Emergence of In-Context Learning and Chain-of-Thought in Large Language Model</b></i>, Chen et al., <a href="https://arxiv.org/abs/2502.03325" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Large Language Models for Recommendation with Deliberative User Preference Alignment</b></i>, Fang et al., <a href="https://arxiv.org/abs/2502.02061" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>ChineseEcomQA: A Scalable E-commerce Concept Evaluation Benchmark for Large Language Models</b></i>, Chen et al., <a href="https://arxiv.org/abs/2502.20196" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>DeepRAG: Thinking to Retrieval Step by Step for Large Language Models</b></i>, Guan et al., <a href="https://arxiv.org/abs/2502.01142" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Open Deep Research</b></i>, Team et al., <a href="https://github.com/nickscamara/open-deep-research" target="_blank"><img src="https://img.shields.io/badge/Github-2025.02-white" alt="Github Badge"></a></li>
<li><i><b>HopRAG: Multi-Hop Reasoning for Logic-Aware Retrieval-Augmented Generation</b></i>, Liu et al., <a href="https://arxiv.org/abs/2502.12442" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>O1 Embedder: Let Retrievers Think Before Action</b></i>, Yan et al., <a href="https://arxiv.org/abs/2502.07555" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models (VLMs) via Reinforcement Learning</b></i>, Pan et al., <a href="https://arxiv.org/abs/2502.19634" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Towards Robust Legal Reasoning: Harnessing Logical LLMs in Law</b></i>, Kant et al., <a href="https://arxiv.org/abs/2502.17638" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning</b></i>, Lu et al., <a href="https://arxiv.org/abs/2502.11271" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning</b></i>, Song et al., <a href="https://arxiv.org/abs/2503.05592" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>RARE: Retrieval-Augmented Reasoning Modeling</b></i>, Wang et al., <a href="https://arxiv.org/abs/2503.23513" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Graph-Augmented Reasoning: Evolving Step-by-Step Knowledge Graph Retrieval for LLM Reasoning</b></i>, Wu et al., <a href="https://arxiv.org/abs/2503.01642" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Learning to Reason with Search for LLMs via Reinforcement Learning</b></i>, Chen et al., <a href="https://arxiv.org/abs/2503.19470" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning</b></i>, Liu et al., <a href="https://arxiv.org/abs/2503.16252" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>m1: Unleash the Potential of Test-Time Scaling for Medical Reasoning with Large Language Models</b></i>, Huang et al., <a href="https://arxiv.org/abs/2504.00869" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.04-red" alt="arXiv Badge"></a></li>
</ul>



<h3 id="multilingual-long-cot">5.4 Multilingual Long CoT</h3>
</ul>

<ul>
<li><i><b>Cross-lingual Prompting: Improving Zero-shot Chain-of-Thought Reasoning across Languages</b></i>, Qin et al., <a href="https://aclanthology.org/2023.emnlp-main.163/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>Not All Languages Are Created Equal in LLMs: Improving Multilingual Capability by Cross-Lingual-Thought Prompting</b></i>, Huang et al., <a href="https://aclanthology.org/2023.findings-emnlp.826/" target="_blank"><img src="https://img.shields.io/badge/PDF-2023.12-blue" alt="PDF Badge"></a></li>
<li><i><b>xcot: Cross-lingual instruction tuning for cross-lingual chain-of-thought reasoning</b></i>, Chai et al., <a href="https://arxiv.org/abs/2401.07037" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Multilingual large language model: A survey of resources, taxonomy and frontiers</b></i>, Qin et al., <a href="https://arxiv.org/abs/2404.04925" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.04-red" alt="arXiv Badge"></a></li>
<li><i><b>A Tree-of-Thoughts to Broaden Multi-step Reasoning across Languages</b></i>, Ranaldi et al., <a href="https://aclanthology.org/2024.findings-naacl.78/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.06-blue" alt="PDF Badge"></a></li>
<li><i><b>AutoCAP: Towards Automatic Cross-lingual Alignment Planning for Zero-shot Chain-of-Thought</b></i>, Zhang et al., <a href="https://aclanthology.org/2024.findings-acl.546/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Enhancing Advanced Visual Reasoning Ability of Large Language Models</b></i>, Li et al., <a href="https://aclanthology.org/2024.emnlp-main.114/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.11-blue" alt="PDF Badge"></a></li>
<li><i><b>DRT-o1: Optimized Deep Reasoning Translation via Long Chain-of-Thought</b></i>, Wang et al., <a href="https://arxiv.org/abs/2412.17498" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>A survey of multilingual large language models</b></i>, Qin et al., <a href="https://www.cell.com/patterns/fulltext/S2666-3899(24)00290-3" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Demystifying Multilingual Chain-of-Thought in Process Reward Modeling</b></i>, Wang et al., <a href="https://arxiv.org/abs/2502.12663" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>The Multilingual Mind: A Survey of Multilingual Reasoning in Language Models</b></i>, Ghosh et al., <a href="https://arxiv.org/abs/2502.09457" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
</ul>



<h3 id="multimodal-long-cot">5.5 Multimodal Long CoT</h3>
</ul>

<ul>
<li><i><b>Large Language Models Can Self-Correct with Minimal Effort</b></i>, Wu et al., <a href="https://openreview.net/forum?id=mmZLMs4l3d" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.05-blue" alt="PDF Badge"></a></li>
<li><i><b>Multimodal Chain-of-Thought Reasoning in Language Models</b></i>, Zhang et al., <a href="https://openreview.net/forum?id=y1pPWFVfvR" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.06-blue" alt="PDF Badge"></a></li>
<li><i><b>Q*: Improving multi-step reasoning for llms with deliberative planning</b></i>, Wang et al., <a href="https://arxiv.org/abs/2406.14283" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.06-red" alt="arXiv Badge"></a></li>
<li><i><b>M<sup>3</sup>CoT: A Novel Benchmark for Multi-Domain Multi-step Multi-modal Chain-of-Thought</b></i>, Chen et al., <a href="https://aclanthology.org/2024.acl-long.446/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>A survey on evaluation of multimodal large language models</b></i>, Huang et al., <a href="https://arxiv.org/abs/2408.15769" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.08-red" alt="arXiv Badge"></a></li>
<li><i><b>Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning</b></i>, Zhai et al., <a href="https://openreview.net/forum?id=nBjmMF2IZU" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>What factors affect multi-modal in-context learning? an in-depth exploration</b></i>, Qin et al., <a href="https://arxiv.org/abs/2410.20482" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>Enhancing Advanced Visual Reasoning Ability of Large Language Models</b></i>, Li et al., <a href="https://aclanthology.org/2024.emnlp-main.114/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Insight-v: Exploring long-chain visual reasoning with multimodal large language models</b></i>, Dong et al., <a href="https://arxiv.org/abs/2411.14432" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>Llava-o1: Let vision language models reason step-by-step</b></i>, Xu et al., <a href="https://arxiv.org/abs/2411.10440" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>AtomThink: A Slow Thinking Framework for Multimodal Mathematical Reasoning</b></i>, Xiang et al., <a href="https://arxiv.org/abs/2411.11930" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>ARES: Alternating Reinforcement Learning and Supervised Fine-Tuning for Enhanced Multi-Modal Chain-of-Thought Reasoning Through Diverse AI Feedback</b></i>, Byun et al., <a href="https://aclanthology.org/2024.emnlp-main.252/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.11-blue" alt="PDF Badge"></a></li>
<li><i><b>Enhancing the reasoning ability of multimodal large language models via mixed preference optimization</b></i>, Wang et al., <a href="https://arxiv.org/abs/2411.10442" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.11-red" alt="arXiv Badge"></a></li>
<li><i><b>Slow Perception: Let's Perceive Geometric Figures Step-by-step</b></i>, Wei et al., <a href="https://arxiv.org/abs/2412.20631" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Diving into Self-Evolving Training for Multimodal Reasoning</b></i>, Liu et al., <a href="https://arxiv.org/abs/2412.17451" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Scaling inference-time search with vision value model for improved visual comprehension</b></i>, Xiyao et al., <a href="https://arxiv.org/abs/2412.03704" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>CoMT: A Novel Benchmark for Chain of Multi-modal Thought on Large Vision-Language Models</b></i>, Cheng et al., <a href="https://arxiv.org/abs/2412.12932" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>Inference Retrieval-Augmented Multi-Modal Chain-of-Thoughts Reasoning for Language Models</b></i>, He et al., <a href="https://openreview.net/pdf/9a7e7a9787d14ac8302215f8e4ef959606b78a94.pdf" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.00-blue" alt="PDF Badge"></a></li>
<li><i><b>Audio-CoT: Exploring Chain-of-Thought Reasoning in Large Audio Language Model</b></i>, Ma et al., <a href="https://arxiv.org/abs/2501.07246" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>BoostStep: Boosting mathematical capability of Large Language Models via improved single-step reasoning</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2501.03226" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model</b></i>, Zang et al., <a href="https://arxiv.org/abs/2501.12368" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Can MLLMs Reason in Multimodality? EMMA: An Enhanced MultiModal ReAsoning Benchmark</b></i>, Hao et al., <a href="https://arxiv.org/abs/2501.05444" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Visual Agents as Fast and Slow Thinkers</b></i>, Sun et al., <a href="https://openreview.net/forum?id=ncCuiD3KJQ" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.01-blue" alt="PDF Badge"></a></li>
<li><i><b>Virgo: A Preliminary Exploration on Reproducing o1-like MLLM</b></i>, Du et al., <a href="https://arxiv.org/abs/2501.01904" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Llamav-o1: Rethinking step-by-step visual reasoning in llms</b></i>, Thawakar et al., <a href="https://arxiv.org/abs/2501.06186" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Inference-time scaling for diffusion models beyond scaling denoising steps</b></i>, Ma et al., <a href="https://arxiv.org/abs/2501.09732" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step</b></i>, Guo et al., <a href="https://arxiv.org/abs/2501.13926" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Imagine while Reasoning in Space: Multimodal Visualization-of-Thought</b></i>, Li et al., <a href="https://arxiv.org/abs/2501.07542" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Monte Carlo Tree Diffusion for System 2 Planning</b></i>, Yoon et al., <a href="https://arxiv.org/abs/2502.07202" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Boosting Multimodal Reasoning with MCTS-Automated Structured Thinking</b></i>, Wu et al., <a href="https://arxiv.org/abs/2502.02339" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Audio-Reasoner: Improving Reasoning Capability in Large Audio Language Models</b></i>, Xie et al., <a href="https://arxiv.org/abs/2503.02318" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Visual-RFT: Visual Reinforcement Fine-Tuning</b></i>, Liu et al., <a href="https://arxiv.org/abs/2503.01785" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Qwen2. 5-Omni Technical Report</b></i>, Xu et al., <a href="https://arxiv.org/abs/2503.20215" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Vision-r1: Incentivizing reasoning capability in multimodal large language models</b></i>, Huang et al., <a href="https://arxiv.org/abs/2503.06749" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl</b></i>, Peng et al., <a href="https://arxiv.org/abs/2503.07536" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Reason-RFT: Reinforcement Fine-Tuning for Visual Reasoning</b></i>, Tan et al., <a href="https://arxiv.org/abs/2503.20752" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>OThink-MR1: Stimulating multimodal generalized reasoning capabilities through dynamic reinforcement learning</b></i>, Liu et al., <a href="https://arxiv.org/abs/2503.16081" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Grounded Chain-of-Thought for Multimodal Large Language Models</b></i>, Wu et al., <a href="https://arxiv.org/abs/2503.12799" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Test-Time View Selection for Multi-Modal Decision Making</b></i>, Jain et al., <a href="https://openreview.net/forum?id=aNmZ9s6BZV" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.03-blue" alt="PDF Badge"></a></li>
<li><i><b>Rethinking RL Scaling for Vision Language Models: A Transparent, From-Scratch Framework and Comprehensive Evaluation Scheme</b></i>, Ma et al., <a href="https://arxiv.org/abs/2504.02587" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.04-red" alt="arXiv Badge"></a></li>
</ul>



<h3 id="safety-and-stability-for-long-cot">5.6 Safety and Stability for Long CoT</h3>
</ul>

<ul>
<li><i><b>Larger and more instructable language models become less reliable</b></i>, Zhou et al., <a href="https://www.nature.com/articles/s41586-024-07930-y" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.00-blue" alt="PDF Badge"></a></li>
<li><i><b>On the Hardness of Faithful Chain-of-Thought Reasoning in Large Language Models</b></i>, Tanneru et al., <a href="https://arxiv.org/abs/2406.10625" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.06-red" alt="arXiv Badge"></a></li>
<li><i><b>The Impact of Reasoning Step Length on Large Language Models</b></i>, Jin et al., <a href="https://aclanthology.org/2024.findings-acl.108/" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.08-blue" alt="PDF Badge"></a></li>
<li><i><b>Unlocking the Capabilities of Thought: A Reasoning Boundary Framework to Quantify and Optimize Chain-of-Thought</b></i>, Chen et al., <a href="https://openreview.net/forum?id=pC44UMwy2v" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.09-blue" alt="PDF Badge"></a></li>
<li><i><b>Can Large Language Models Understand You Better? An MBTI Personality Detection Dataset Aligned with Population Traits</b></i>, Li et al., <a href="https://arxiv.org/abs/2412.12510" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.12-red" alt="arXiv Badge"></a></li>
<li><i><b>o3-mini vs DeepSeek-R1: Which One is Safer?</b></i>, Arrieta et al., <a href="https://arxiv.org/abs/2501.18438" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Efficient Reasoning with Hidden Thinking</b></i>, Shen et al., <a href="https://arxiv.org/abs/2501.19201" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Think More, Hallucinate Less: Mitigating Hallucinations via Dual Process of Fast and Slow Thinking</b></i>, Cheng et al., <a href="https://arxiv.org/abs/2501.01306" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection</b></i>, Zhao et al., <a href="https://arxiv.org/abs/2501.02295" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Challenges in Ensuring AI Safety in DeepSeek-R1 Models: The Shortcomings of Reinforcement Learning Strategies</b></i>, Parmar et al., <a href="https://arxiv.org/abs/2501.17030" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>Early External Safety Testing of OpenAI's o3-mini: Insights from the Pre-Deployment Evaluation</b></i>, Arrieta et al., <a href="https://arxiv.org/abs/2501.17749" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>International AI Safety Report</b></i>, Bengio et al., <a href="https://arxiv.org/abs/2501.17805" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>GuardReasoner: Towards Reasoning-based LLM Safeguards</b></i>, Liu et al., <a href="https://arxiv.org/abs/2501.18492" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.01-red" alt="arXiv Badge"></a></li>
<li><i><b>OVERTHINKING: Slowdown Attacks on Reasoning LLMs</b></i>, Kumar et al., <a href="https://arxiv.org/abs/2502.02542" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>A Mousetrap: Fooling Large Reasoning Models for Jailbreak with Chain of Iterative Chaos</b></i>, Yao et al., <a href="https://arxiv.org/abs/2502.15806" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>MetaSC: Test-Time Safety Specification Optimization for Language Models</b></i>, Gallego et al., <a href="https://arxiv.org/abs/2502.07985" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Leveraging Reasoning with Guidelines to Elicit and Utilize Knowledge for Enhancing Safety Alignment</b></i>, Wang et al., <a href="https://arxiv.org/abs/2502.04040" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1</b></i>, Zhou et al., <a href="https://arxiv.org/abs/2502.12659" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Reasoning and the Trusting Behavior of DeepSeek and GPT: An Experiment Revealing Hidden Fault Lines in Large Language Models</b></i>, Lu et al., <a href="https://arxiv.org/abs/2502.12825" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Superintelligent Agents Pose Catastrophic Risks: Can Scientist AI Offer a Safer Path?</b></i>, Bengio et al., <a href="https://arxiv.org/abs/2502.15657" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Emergent Response Planning in LLM</b></i>, Dong et al., <a href="https://arxiv.org/abs/2502.06258" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Investigating the Impact of Quantization Methods on the Safety and Reliability of Large Language Models</b></i>, Kharinaev et al., <a href="https://arxiv.org/abs/2502.15799" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Safety Evaluation of DeepSeek Models in Chinese Contexts</b></i>, Zhang et al., <a href="https://arxiv.org/abs/2502.11137" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Reasoning Does Not Necessarily Improve Role-Playing Ability</b></i>, Feng et al., <a href="https://arxiv.org/abs/2502.16940" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>H-CoT: Hijacking the Chain-of-Thought Safety Reasoning Mechanism to Jailbreak Large Reasoning Models, Including OpenAI o1/o3, DeepSeek-R1, and Gemini 2.0 Flash Thinking</b></i>, Kuo et al., <a href="https://arxiv.org/abs/2502.12893" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>BoT: Breaking Long Thought Processes of o1-like Large Language Models through Backdoor Attack</b></i>, Zhu et al., <a href="https://arxiv.org/abs/2502.12202" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>" Nuclear Deployed!": Analyzing Catastrophic Risks in Decision-making of Autonomous LLM Agents</b></i>, Xu et al., <a href="https://arxiv.org/abs/2502.11355" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>SafeChain: Safety of Language Models with Long Chain-of-Thought Reasoning Capabilities</b></i>, Jiang et al., <a href="https://arxiv.org/abs/2502.12025" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language Models from Jailbreaking</b></i>, Zhu et al., <a href="https://arxiv.org/abs/2502.12970" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>CER: Confidence Enhanced Reasoning in LLMs</b></i>, Razghandi et al., <a href="https://arxiv.org/abs/2502.14634" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps</b></i>, Tutek et al., <a href="https://arxiv.org/abs/2502.14829" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>The Hidden Dimensions of LLM Alignment: A Multi-Dimensional Safety Analysis</b></i>, Pan et al., <a href="https://arxiv.org/abs/2502.09674" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>Policy Frameworks for Transparent Chain-of-Thought Reasoning in Large Language Models</b></i>, Chen et al., <a href="https://arxiv.org/abs/2503.14521" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Do Chains-of-Thoughts of Large Language Models Suffer from Hallucinations, Cognitive Biases, or Phobias in Bayesian Reasoning?</b></i>, Araya et al., <a href="https://arxiv.org/abs/2503.15268" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Process or Result? Manipulated Ending Tokens Can Mislead Reasoning LLMs to Ignore the Correct Reasoning Steps</b></i>, Cui et al., <a href="https://arxiv.org/abs/2503.19326" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Safety Tax: Safety Alignment Makes Your Large Reasoning Models Less Reasonable</b></i>, Huang et al., <a href="https://arxiv.org/abs/2503.00555" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>Recitation over Reasoning: How Cutting-Edge Language Models Can Fail on Elementary School-Level Reasoning Problems?</b></i>, Yan et al., <a href="https://arxiv.org/abs/2504.00509" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.04-red" alt="arXiv Badge"></a></li>
<li><i><b>Reasoning Models Don’t Always Say What They Think</b></i>, Chen et al., <a href="https://www.anthropic.com/research/reasoning-models-dont-say-think" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.04-blue" alt="PDF Badge"></a></li>
</ul>
<h2 id="resources">6. Resources</h2>


<h3 id="open-sourced-training-framework">6.1 Open-Sourced Training Framework</h3>
<ul>
<li><i><b>OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework</b></i>, Hu et al., <a href="https://arxiv.org/abs/2405.11143" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.05-red" alt="arXiv Badge"></a></li>
<li><i><b>LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models</b></i>, Hao et al., <a href="https://openreview.net/forum?id=b0y6fbSUG0" target="_blank"><img src="https://img.shields.io/badge/PDF-2024.07-blue" alt="PDF Badge"></a></li>
<li><i><b>OpenR: An Open Source Framework for Advanced Reasoning with Large Language Models</b></i>, Wang et al., <a href="https://arxiv.org/abs/2410.09671" target="_blank"><img src="https://img.shields.io/badge/arXiv-2024.10-red" alt="arXiv Badge"></a></li>
<li><i><b>TinyZero</b></i>, Pan et al., <a href="https://github.com/Jiayi-Pan/TinyZero" target="_blank"><img src="https://img.shields.io/badge/Github-2025.00-white" alt="Github Badge"></a></li>
<li><i><b>R1-V: Reinforcing Super Generalization Ability in Vision-Language Models with Less Than </sup>3</b></i>, Chen et al., <a href="https://github.com/Deep-Agent/R1-V" target="_blank"><img src="https://img.shields.io/badge/Github-2025.00-white" alt="Github Badge"></a></li>
<li><i><b>VL-Thinking: An R1-Derived Visual Instruction Tuning Dataset for Thinkable LVLMs</b></i>, Chen et al., <a href="https://github.com/UCSC-VLAA/VL-Thinking" target="_blank"><img src="https://img.shields.io/badge/Github-2025.00-white" alt="Github Badge"></a></li>
<li><i><b>VLM-R1: A stable and generalizable R1-style Large Vision-Language Model</b></i>, Shen et al., <a href="https://github.com/om-ai-lab/VLM-R1" target="_blank"><img src="https://img.shields.io/badge/Github-2025.00-white" alt="Github Badge"></a></li>
<li><i><b>7B Model and 8K Examples: Emerging Reasoning with Reinforcement Learning is Both Effective and Efficient</b></i>, Zeng et al., <a href="https://hkust-nlp.notion.site/simplerl-reason" target="_blank"><img src="https://img.shields.io/badge/Notion-2025.01-white" alt="Notion Badge"></a></li>
<li><i><b>Open R1</b></i>, Team et al., <a href="https://github.com/huggingface/open-r1" target="_blank"><img src="https://img.shields.io/badge/Github-2025.01-white" alt="Github Badge"></a></li>
<li><i><b>DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL</b></i>, Luo et al., <a href="https://github.com/agentica-project/rllm" target="_blank"><img src="https://img.shields.io/badge/PDF-2025.02-blue" alt="PDF Badge"></a></li>
<li><i><b>X-R1</b></i>, Team et al., <a href="https://github.com/dhcode-cpp/X-R1" target="_blank"><img src="https://img.shields.io/badge/Github-2025.02-white" alt="Github Badge"></a></li>
<li><i><b>Open-Reasoner-Zero: An Open Source Approach to Scaling Reinforcement Learning on the Base Model</b></i>, Jingcheng Hu and Yinmin Zhang and Qi Han and Daxin Jiang and Xiangyu Zhang et al., <a href="https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero" target="_blank"><img src="https://img.shields.io/badge/Github-2025.02-white" alt="Github Badge"></a></li>
<li><i><b>Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning</b></i>, Xie et al., <a href="https://arxiv.org/abs/2502.14768" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.02-red" alt="arXiv Badge"></a></li>
<li><i><b>R1-Multimodal-Journey</b></i>, Shao et al., <a href="https://github.com/FanqingM/R1-Multimodal-Journey" target="_blank"><img src="https://img.shields.io/badge/Github-2025.02-white" alt="Github Badge"></a></li>
<li><i><b>Open-R1-Multimodal</b></i>, Lab et al., <a href="https://github.com/EvolvingLMMs-Lab/open-r1-multimodal" target="_blank"><img src="https://img.shields.io/badge/Github-2025.02-white" alt="Github Badge"></a></li>
<li><i><b>Video-R1</b></i>, Team et al., <a href="https://github.com/tulerfeng/Video-R1" target="_blank"><img src="https://img.shields.io/badge/Github-2025.02-white" alt="Github Badge"></a></li>
<li><i><b>Dapo: An open-source llm reinforcement learning system at scale</b></i>, Yu et al., <a href="https://arxiv.org/abs/2503.14476" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.03-red" alt="arXiv Badge"></a></li>
<li><i><b>VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks</b></i>, YuYue et al., <a href="https://arxiv.org/abs/2504.05118" target="_blank"><img src="https://img.shields.io/badge/arXiv-2025.04-red" alt="arXiv Badge"></a></li>
</ul>



## 🎁 Citation
If you find this work useful, welcome to cite us.
```bib
@misc{chen2025reasoning,
      title={Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models}, 
      author={Qiguang Chen and Libo Qin and Jinhao Liu and Dengyun Peng and Jiannan Guan and Peng Wang and Mengkang Hu and Yuhang Zhou and Te Gao and Wanxiang Che},
      year={2025},
      eprint={2503.09567},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2503.09567}, 
}
```

<!-- omit in toc -->
## ⭐ Star History

<a href="https://star-history.com/#LightChen233/Awesome-Long-Chain-of-Thought-Reasoning&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=LightChen233/Awesome-Long-Chain-of-Thought-Reasoning&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=LightChen233/Awesome-Long-Chain-of-Thought-Reasoning&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=LightChen233/Awesome-Long-Chain-of-Thought-Reasoning&type=Date" />
 </picture>
</a>
