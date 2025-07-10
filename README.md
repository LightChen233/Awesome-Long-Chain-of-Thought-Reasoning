# <img src="assets/images/icon.jpg" alt="SVG Image" width="40px"> Awesome-Long-Chain-of-Thought-Reasoning


[![arXiv](https://img.shields.io/badge/arXiv-Long_Chain_of_Thought-b31b1b.svg)](https://arxiv.org/pdf/2503.09567) 
[![Paper](https://img.shields.io/badge/Paper-908-green.svg)](https://github.com//LightChen233/Awesome-Long-Chain-of-Thought-Reasoning)
[![Last Commit](https://img.shields.io/github/last-commit/LightChen233/Awesome-Long-Chain-of-Thought-Reasoning)](https://github.com/LightChen233/Awesome-Long-Chain-of-Thought-Reasoning)
[![Contribution Welcome](https://img.shields.io/badge/Contributions-welcome-blue)]()

\[[English Tutorial](README.md)\] / \[[中文教程](README-zh.md)\] 


![image](./assets/images/overall.png)


<!-- omit in toc -->
# 🔥 News
- **2025.07**: 🎉🎉🎉 We have updated the number of reviewed papers to over 1000. Additionally, we have added bilingual supports and updated our repository more friendly for Long-CoT beginner.
- **2025.04**: 🎉🎉🎉 We have updated the number of reviewed papers to over 900. Additionally, we have enhanced the presentation with more engaging teaser figure.
- **2025.03**: 🎉🎉🎉 We have published a survey paper titled "[Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models](https://arxiv.org/pdf/2503.09567)". Please feel free to cite or open pull requests for your awesome studies.

<!-- omit in toc -->
# 🌟 Introduction

Welcome to the repository associated with our survey paper, "Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models". This repository contains **resources and updates** related to our ongoing Long CoT research. For a detailed introduction, please refer to [our survey paper](https://arxiv.org/pdf/2503.09567).

Recent advancements in reasoning with large language models (RLLMs), such as OpenAI-O1 and DeepSeek-R1, have demonstrated their impressive capabilities in complex domains like mathematics and coding. A central factor in their success lies in the application of long chain-of-thought (Long CoT) characteristics, which enhance reasoning abilities and enable the solution of intricate problems.

![image](./assets/images/develop.png)

However, despite these developments, a comprehensive survey on Long CoT is still lacking, limiting our understanding of its distinctions from traditional short chain-of-thought (Short CoT) and complicating ongoing debates on issues like "overthinking" and "test-time scaling." This survey seeks to fill this gap by offering a unified perspective on Long CoT.
1. We first distinguish Long CoT from Short CoT and introduce a novel taxonomy to categorize current reasoning paradigms.
2. Next, we explore the key characteristics of Long CoT: deep reasoning, extensive exploration, and feasible reflection, which enable models to handle more complex tasks and produce more efficient, coherent outcomes compared to the shallower Short CoT.
3. We then investigate key phenomena such as the emergence of Long CoT with these characteristics, including overthinking, and test-time scaling, offering insights into how these processes manifest in practice.
4. Finally, we identify significant research gaps and highlight promising future directions, including the integration of multi-modal reasoning, efficiency improvements, and enhanced knowledge frameworks. 

By providing a structured overview, this survey aims to inspire future research and further the development of logical reasoning in artificial intelligence.

![image](./assets/images/intro.jpg)

<!-- omit in toc -->


# 🕹️ Content
## 0. How to Learn \& About Us
We aim to help newcomers quickly establish domain knowledge, so our design concept is as follows: briefly introduce the main technologies involved in reasoning large models and Long CoT, allowing everyone to understand which problems different technologies can address, so that when they wish to delve deeper into the field in the future, they will have a clear starting point.

We are a team of beginners in reasoning large models, and we hope that through our own learning experiences, we can offer some assistance to future learners, accelerating the popularization and application of reasoning large models. We welcome more friends to join our project, and we are also open to friendship and academic collaboration. For any inquiries, please feel free to contact us via email at [charleschen2333@gmail.com](mailto:charleschen2333@gmail.com).

**Daily Knowledge Resources**
- **Social Media:**
  - Recommended WeChat Public Accounts: JIQIZHIXIN, Paper Weekly, MLNLP...
  - Recommended Twitter Accounts: [AK](https://x.com/_akhaliq), [elvis](https://x.com/omarsar0), [Philipp Schmid](https://x.com/_philschmid), ...
- **Cutting-edge Courses:** [CS336](https://stanford-cs336.github.io/spring2025/)
- **Community Sharing:** [MLNLP](https://space.bilibili.com/168887299), [JIQIZHIXIN](https://space.bilibili.com/73414544), [BAAI](https://hub.baai.ac.cn/), [NICE Academic](https://space.bilibili.com/507524288)

## 1. Classical Reasoning Model
- [OpenAI-o1 / o3 / o4](https://platform.openai.com/docs/models/#o3): The earliest reasoning large language models exploring Long CoT, developed by OpenAI’s first-tier models.
- [Gemini](https://github.com/google-gemini): First-tier reasoning large language models developed by Google.
- [Deepseek-r1](https://github.com/deepseek-ai/DeepSeek-R1): The first open-source reasoning large language model with Long CoT.
- [QwQ](https://qwenlm.github.io/zh/blog/qwq-32b-preview/): The first open-source large-scale reasoning large language model with Long CoT.
- [Qwen3](https://github.com/QwenLM/Qwen3): The most commonly used open-source reasoning large language models for Long CoT developed by Alibaba.
- [Seed-Thinking-v1.5](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/blob/main/seed-thinking-v1.5.pdf): ByteDance’s open-source reasoning model for Long CoT.
- [Kimi-k1.5](https://github.com/MoonshotAI/Kimi-k1.5): The earliest multimodal reasoning model for Long CoT developed by Moonshot.
- [MiniMax-m1](https://github.com/MiniMax-AI/MiniMax-M1): The open-source reasoning model for Long CoT developed by MiniMax.


## 2. Introduction to Long-CoT Capabilities
In this chapter, we will provide the most representative technologies for each capability, along with the latest developments. A detailed list of papers can be found in the [complete list](pages/paper.md).


![image](./assets/images/contents.jpg)


### 2.1 Deep Reasoning

The core of deep reasoning ability lies in the need for sufficient logical depth to manage a large number of reasoning nodes. Without this capability, the performance of reasoning large language models (RLLMs) significantly degrades. Current methods for enhancing deep reasoning can be categorized into two main approaches: Deep Reasoning Format and Deep Reasoning Learning.

<img src="./assets/images/deep-reasoning-2.png" style="width: 580pt">

#### 2.1.1 Deep Reasoning Format
Since reasoning models heavily depend on the format of reasoning, they tend to achieve the deepest reasoning paths in the forms they excel at. As a result, some works have begun exploring better reasoning formats for deeper reasoning.



<img src="./assets/images/deep-reasoning-1.jpg" style="width: 580pt">

**Natural Language Deep Reasoning**

- **Core Idea:** Aims to express deep reasoning through natural language formats.
- **Representative Works:**
  - [Natural Program](https://proceedings.neurips.cc/paper_files/paper/2023/file/72393bd47a35f5b3bee4c609e7bba733-Paper-Conference.pdf): ensures more structured and rigorous logical analysis.
  - [Code I/O](https://arxiv.org/abs/2502.07316): restructures code-based reasoning patterns into natural language forms, further unleashing the reasoning potential of RLLMs.

---

**Structured Language Deep Reasoning**

- **Core Idea:** Aims to enhance deep reasoning through programmatic or symbolic language formats. Current research primarily focuses on using code to improve mathematical reasoning capabilities.
- **Representative Works:**
  - [Program-of-Thought](https://openreview.net/forum?id=YfZ4ZPt8zd): enables models to think using code language, thereby enhancing their reasoning capabilities.
  - [DeepSeek-Prover](https://arxiv.org/abs/2405.14333): converts natural language questions into formal statements, filters out low-quality statements, and generates proofs to create synthetic data, enhancing LLM’s theorem proving ability.
  - [RBF](https://proceedings.neurips.cc/paper_files/paper/2024/hash/62ab1c2cb4b03e717005479efb211841-Abstract-Conference.html): demonstrates why structured language is more effective than natural language in scenarios that require strong planning.



---

**Latent Space Deep Reasoning**

- **Core Idea:** Enhances LLM reasoning ability through continuous latent space operations.
- **Representative Works:**
  1. **Token-driven:** Early studies introduced implicit "planning tokens" or "thinking tokens" to guide the reasoning process in latent space.
     - [Coconut (Chain of Continuous Thought)](https://arxiv.org/abs/2412.06769): further expands this method by maintaining multiple parallel reasoning paths, enhancing complexity while ensuring efficiency.
     - [Heima](https://arxiv.org/abs/2501.19201): performs efficient reasoning through latent hidden spaces, innovatively compressing the entire Long CoT process into a single token, resulting in significant computational resource savings.
  2. **Vector-driven:** Inserts an additional vector to guide the reasoning process in latent space.
     - [LTMs](https://arxiv.org/abs/2502.01567): innovatively abstracts each layer of the LLM into "thinking blocks" and introduces the concept of a "thinking vector" for each layer. Through iterative deep computation in latent space, the model dynamically scales the computational load during testing.
  3. **Manager-driven:** Proposes a continuous manager mechanism to manage the latent space states.
     - [Recurrent Block](https://arxiv.org/abs/2502.05171): uses iterative control over trained "recurrent blocks" as recursive "thinking blocks" to integrate deeper model layers during reasoning, enhancing performance without the need for specialized training data.
     - [Implicit Thought Transformer (ITT)](https://arxiv.org/abs/2502.13842): leverages raw Transformer layers as recursive "thinking blocks," using adaptive token routing to select key tokens and residual thinking connections to control reasoning depth, thereby achieving efficient processing of key tokens.
- **Relevant Repositories:**
  - [Awesome-Latent-CoT](https://github.com/EIT-NLP/Awesome-Latent-CoT): provides an overview of various thought chain representations in latent space, capturing complex non-linguistic thoughts that cannot be expressed by language alone.

---

#### 2.1.2 Deep Reasoning Learning
The deficiency of deep reasoning abilities in RLLMs can significantly reduce model performance. As a result, the academic focus has shifted towards enhancing reasoning capabilities through training. Supervised fine-tuning (SFT), as a memory process, can stabilize model output, while reinforcement learning (RL) facilitates generalization and self-learning.

---

**Deep Reasoning Imitation**

- **Core Idea:** By imitating advanced reasoning systems, deep reasoning in RLLMs can be effectively achieved, enabling models to learn complex reasoning patterns and generalize across tasks.
- **Representative Works:**
  1. **Imitation from Human**
     - [GSM8K/GPT-Verifier](https://arxiv.org/abs/2110.14168): Introduces early imitative learning based on human-annotated deep reasoning samples.
     - [ALT](https://proceedings.neurips.cc/paper_files/paper/2024/file/8678da90126aa58326b2fc0254b33a8c-Paper-Conference.pdf): Enhances deep reasoning in RLLMs by generating a large-scale dataset of human-annotated logical templates.
  2. **Imitation from Advanced RLLMs**
     - [AceMath](https://arxiv.org/abs/2412.15084): Uses few-shot prompting to distill Long CoT samples from advanced LLMs, improving performance through multi-stage quality-guided SFT.
     - [DART-Math](https://proceedings.neurips.cc/paper_files/paper/2024/file/0ef1afa0daa888d695dcd5e9513bafa3-Paper-Conference.pdf): Effectively distills difficulty-dependent deep reasoning samples through rejection sampling in the synthesis stage.
     - [OpenThoughts](https://arxiv.org/abs/2506.04178) / [OpenCodeReasoning](https://arxiv.org/abs/2504.01943) / [NaturalThoughts](https://arxiv.org/abs/2507.01921): Extends this paradigm to mathematics, code, and general scenarios.
  3. **Imitation from Scaling-augmented RLLMs**
     - [Bansal et al.](https://openreview.net/forum?id=HuYSURUxs2): Find that expanding the sampling scale and length improves data quality.
     - [Qwen-Math](https://arxiv.org/abs/2409.12122) / [PromptCoT](https://arxiv.org/abs/2503.02324): Further combine large-scale sampling with reward model sample selection to generate Olympic-level difficulty deep reasoning samples.
     - [FastMCTS](https://arxiv.org/abs/2502.11476): Utilizes Monte Carlo Tree Search (MCTS) to identify optimal deep reasoning paths.
- **Latest Developments:**
     - [Journey P2](https://arxiv.org/abs/2411.16489): Knowledge distilled from advanced RLLM APIs such as o1, R1 significantly boosts small LLMs' performance, with supervised fine-tuning methods surpassing teacher models in complex mathematical reasoning tasks.
     - [s1](https://arxiv.org/abs/2501.19393) / [LIMO](https://arxiv.org/abs/2502.03387): A small number of high-quality samples is sufficient to activate deep reasoning abilities in base LLMs.

---

**Deep Reasoning Self-Learning**

- **Core Idea:** Although simple imitation yields excellent performance, current models still heavily rely on human annotations or outputs from advanced models during imitation and distillation. To break through this limitation, research focuses on self-learning techniques to achieve more advanced reasoning capabilities.
- **Representative Works:**
  1. **Self-Learning from Direct Sampling**
     - [STaR](https://arxiv.org/abs/2203.14465): Uses in-context learning (ICL) to sample deep reasoning results and treats the correctness of the final answer as an implicit reward for self-learning.
     - [Reinforced Self-Training (ReST)](https://arxiv.org/abs/2308.08998): Proposes the "Grow-Improve" paradigm, where the self-generated reasoning process is rewarded, enhancing with offline reinforcement learning.
     - [ReST$^{EM}$](https://arxiv.org/abs/2312.06585): Generates rewards and iteratively optimizes LLMs to achieve peak performance on validation sets, significantly improving robustness.
     - [TOPS](https://arxiv.org/abs/2502.18080): Finds that self-learning with deep reasoning samples at an appropriate reasoning depth is the most efficient.
  2. **Self-Learning from Tree Search**
     - [PGTS](https://arxiv.org/abs/2502.06813): Uses policy-guided tree search, combining reinforcement learning with structured tree exploration.
     - [ReST-MCTS*](https://arxiv.org/abs/2406.03816): Optimizes MCTS behavior through progressive trajectory extraction and curriculum preference learning, significantly improving LLMs' reasoning ability.
- **Latest Developments:** Introduce an error-correction adaptive mechanism by training a verifier or using entropy to filter and optimize the reward process, thus enhancing the quality of self-learning.
   - [UnCert-CoT](https://arxiv.org/abs/2503.15341): Dynamically schedules thought chains based on entropy-aware uncertainty, activating multi-path reasoning only under high-entropy situations, significantly improving code generation accuracy and efficiency.
   - [Wang et al.](https://arxiv.org/abs/2506.01939): Analyzes the impact of verifiable reward-based reinforcement learning on large language models' reasoning capabilities from the token entropy perspective, where high-entropy "branching" tokens dominate adjustments to multi-path reasoning strategies. Policy gradient optimization is applied exclusively to these high-entropy tokens.
   - [CoT-Valve](https://arxiv.org/abs/2502.09601): Dynamically adjusts to reduce reasoning path length based on task difficulty, thus reducing computational overhead.



---

### 2.2 Feasible Reflection

#### 2.2.1 Feedback

Feedback mechanisms provide multi-granularity evaluation signals for Long CoT, ranging from Overall Feedback, which evaluates the final outcome, to Process Feedback, which supervises individual steps of the reasoning process, and Hybrid Feedback, which combines both types. These mechanisms not only support reward modeling and path optimization but also lay the foundation for subsequent self-correction, serving as a crucial bridge to move RLLMs from static generation to dynamic evaluation.

<img src="./assets/images/feedback.png" style="width: 580pt">

---

**Overall Feedback**

- **Core Idea:** Overall feedback evaluates the complete reasoning process and the final result from a global perspective, commonly used to guide large language models in improving reasoning quality during reinforcement learning or self-optimization. Feedback forms include numerical rewards, rule checks, and natural language evaluations.
- **Representative Works:**
  1. **Outcome Reward Models(ORM):** Provide numeric reward signals to optimize output quality, suitable for tasks where accuracy is hard to assess directly.
     - [Gen-Verifier](https://arxiv.org/abs/2110.14168): introduces the first generation verification framework based on reasoning accuracy;
     - [Critic-RM](https://arxiv.org/abs/2411.16646): combines natural language criticism and reward prediction, significantly optimizing feedback quality;
     - [Self-Rewarding LMs (SRLMs)](https://arxiv.org/abs/2502.08922): introduce consistency mechanisms, achieving self-supervised rewards without human annotation.
  2. **Rule Extraction:** Uses in-task rules to verify and correct answers, enhancing the stability of feedback.
     - [STaR](https://arxiv.org/abs/2203.14465) / [ReST](https://arxiv.org/abs/2308.08998): show that rule-based feedback based on final answers outperforms ORM in mathematical tasks;
     - [OpenCodeInterpreter](https://arxiv.org/abs/2402.14658) / [AceCoder](https://arxiv.org/abs/2502.01718): generate program-level feedback using automated test cases in coding tasks.
  3. **RLLMs Feedback (LLM-as-a-Judge):**
      The model self-critiques and evaluates in natural language, enhancing reflection and error-correction capabilities.
     - [EvalPlanner](https://arxiv.org/abs/2501.18099): distinguishes feedback between planning and reasoning;
     - [RoT](https://arxiv.org/abs/2410.12323): combines reverse reasoning and reflection to assist models in discovering knowledge gaps;
     - [AutoRace](https://arxiv.org/abs/2404.05221): provides task-specific evaluation criteria to improve feedback relevance.
- **Relevant Repositories:**
  - [RewardBench](https://github.com/allenai/reward-bench)For system evaluation of ORM methods.

**Process Feedback**

- **Core Idea:**
   Process feedback evaluates each step in the reasoning chain progressively, often in conjunction with reinforcement learning or tree search, guiding the model to fine-tune without relying on human annotations. The feedback sources primarily include process reward models and language models driven by natural language.
- **Representative Works:**
  1. **Process Feedback from Process Reward Models (PRMs):**
      Use automatically constructed or minimally annotated data to train stepwise reward functions, the mainstream approach in Long CoT.
     - [PRM800K](https://arxiv.org/abs/2305.20050): Pioneers using human-annotated stepwise supervision to enhance reward stability;
     - [Math-Shepherd](https://arxiv.org/abs/2312.08935): Automatically generates stepwise feedback using tree search to enhance PRM generalization;
     - [Full-Step-DPO](https://arxiv.org/abs/2502.14356): Rewards the entire reasoning chain, encouraging holistic optimization;
     - [AdaptiveStep](https://arxiv.org/abs/2502.13943): Dynamically segments reasoning steps based on confidence, enabling token-level fine-grained feedback.
  2. **Process Feedback from RLLMs:** Leverages the model’s own generation of natural language feedback to simulate reward signals, improving the flexibility and scalability of process supervision.
     - [React](https://arxiv.org/abs/2210.03629) / [Reflexion](https://arxiv.org/abs/2303.11366): Generate language feedback after each action, enhancing decision-making rationality;
     - [Step-DPO](https://arxiv.org/abs/2405.18629): Introduces a self-validation mechanism to construct positive-negative contrastive samples, optimizing the training process;
     - [CACE](https://arxiv.org/abs/1907.07165): Proposes causal impact metrics between reasoning steps to make the entire chain more interpretable;
     - [ORPS](https://arxiv.org/html/2412.15118v1): Automatically optimizes reasoning strategies with program execution feedback, reducing human reliance.
- **Relevant Repositories:**
  - [ProcessBench](https://github.com/QwenLM/ProcessBench): Evaluates stepwise reasoning and reward model performance;
  - [PRMBench](https://github.com/ssmisya/PRMBench): Focuses on comparative analysis of PRM methods in mathematical tasks.

**Hybrid Feedback**

- **Core Idea:** Hybrid feedback mechanisms combine the strengths of both overall and process feedback, assessing final outputs while focusing on intermediate reasoning steps. This unified multi-granularity evaluation system enhances the overall reasoning quality and error-correction capabilities of language models.
- **Representative Works:**
  - [Consensus Filtering](https://arxiv.org/abs/2501.07301): Combines Monte Carlo estimation with LLM-as-Judge to integrate overall and stepwise feedback, enhancing reasoning consistency and accuracy;
  - [Step-KTO](https://arxiv.org/abs/2501.10799): Merges PRM and ORM binary feedback mechanisms, emphasizing reflection-driven error correction, guiding the model to form more coherent Long CoT structures.

#### 2.2.2 Refinement

The Refinement mechanism focuses on self-correction capabilities based on feedback information, serving as a key step in achieving closed-loop optimization in Long CoT. Through Prompt-based Refinement, spontaneous reflection is achieved; SFT-based Refinement facilitates imitation learning; and RL-based Refinement strengthens self-correction strategies. As a result, the model gradually develops the ability of "self-diagnosis—self-updating," making the reasoning chain more robust and controllable.


<img src="./assets/images/refinement.png" style="width: 580pt">

---

**Prompt-based Refinement**

- **Core Idea:** By guiding the model to generate initial responses via prompts, and allowing for self-feedback and multi-round corrections in subsequent rounds, this method improves reasoning accuracy, reduces hallucinations, and supports stronger automated reflection capabilities.
- **Representative Works:**
  - [ReAct](https://github.com/ysymyth/ReAct) / [Reflexion](https://github.com/noahshinn/reflexion): A typical implementation of the multi-round reflection and self-correction mechanism;
  - [Self-Backtracking](https://arxiv.org/abs/2502.04404) / [Refiner](https://aclanthology.org/2024.eacl-long.67/) / [BackMath](https://aclanthology.org/2025.coling-industry.40/): Supports the model in autonomously backtracking and modifying during the reasoning process, streamlining decision paths;
  - [MCTSr](https://arxiv.org/abs/2406.07394) / [ReST-MCTS](https://arxiv.org/abs/2406.03816): Combines tree search with confidence updates to enable multi-round dynamic reflection;
  - [LLM2](https://arxiv.org/abs/2412.20372)/ [ReARTeR](https://arxiv.org/abs/2501.07861): Promotes the automatic evolution and stable convergence of refinement strategies in Long CoT tasks.
  

**SFT-based Refinement**

- **Core Idea:** By utilizing high-quality reflection data for supervised fine-tuning, the model imitates the self-correction behaviors of more advanced models, enhancing its step-by-step error correction and reflective capabilities. This is suitable for small model capability transfer and fine-grained training.
- **Representative Works:**
  - [rStar](https://arxiv.org/abs/2408.06195): Enhances small model self-improvement capabilities through self-play methods;
  - [Math-Minos](https://arxiv.org/abs/2405.14024): Trains the model using step-by-step rationale labels for fine-grained reasoning;
  - [Journey Learning](https://arxiv.org/abs/2411.16489): Combines MCTS backtracking to generate supervision signals;
  - [MM-Verify](https://arxiv.org/abs/2502.13383): Expands the refinement mechanism to multimodal image-text reasoning.

**RL-based Refinement**

- **Core Idea:** Through reinforcement learning mechanisms, the model is guided to self-reflect and correct during testing or reasoning processes, emphasizing self-refinement capabilities under reward guidance, thus reducing dependence on manual supervision.
- **Representative Works:**
  - [SCoRe](https://arxiv.org/abs/2409.12917): Enhances the model's self-refinement ability during testing through self-generated correction trajectories and regularization;
  - [DeepSeek-R1](https://arxiv.org/abs/2501.12948): Uses result-level reinforcement learning to activate the model's natural feedback and "aha" moments for corrections;
  - [S$^2$R](https://arxiv.org/abs/2502.12853): Combines process-level reinforcement learning to achieve dynamic refinement in reasoning;
  - [ReVISE](https://arxiv.org/abs/2502.14565): Introduces an internal verifier to decide when to trigger RL-guided reflective behaviors.

### 2.3 Extensive Exploration
Extensive Exploration enables reasoning large language models (RLLMs) to explore multiple reasoning paths more deeply and comprehensively when dealing with complex problems, thereby improving problem-solving accuracy and robustness. From the perspective of exploration types, extensive exploration techniques can be divided into three categories: Exploration Scaling, Internal Exploration, and External Exploration.

#### 2.3.1 Exploration Scaling
Exploration Scaling aims to enhance the model's ability to solve more complex problems by increasing the number or length of reasoning paths. This approach is typically suitable when the reasoning task is more complex, and a single reasoning path may not effectively lead to the correct answer.

---

**Sequential Scaling**

- **Core Idea:** By extending the reasoning chain of a single path, the model gradually deepens its thinking, thereby improving its understanding and handling of complex problems. This is especially applicable to tasks with Long CoT that require multi-step reasoning to draw conclusions, such as mathematical proofs, logical deductions, and multi-step planning.
- **Representative Works:**
  - [OpenAI-o1](https://arxiv.org/abs/2412.16720) / [Deepseek-R1](https://arxiv.org/abs/2501.12948): 
  Extends the reasoning chain to provide detailed multi-step reasoning processes, effectively improving the ability to solve complex problems in mathematics, coding, and other areas.
  - [ITT(Inner Thinking Transformer:)](https://arxiv.org/abs/2502.13842): redefines the layer computation in Transformer as "thinking steps," dynamically allocating computation resources to deeper reasoning on key tokens without increasing the total number of model parameters.

---

**Parallel Scaling**

- **Core Idea:** By generating multiple reasoning paths in parallel and combining, voting, or verifying their results, the model can effectively avoid the issue of a single path getting trapped in local optima or errors, thus improving robustness and accuracy in situations with high ambiguity, multiple possible solutions, or unclear outcomes.
- **Representative Works:**
  - [Self-Consistency](https://arxiv.org/abs/2203.11171): Proposes generating multiple reasoning paths and selecting the most frequent answer from the results, effectively improving the stability and accuracy of the final answer.
  - [ECM(Electronic Circuit Model)](https://arxiv.org/abs/2502.03325): Borrowing the concepts of parallel and series circuits in electronics, combines reasoning paths in parallel or series, considering various possibilities and improving decision quality.

---

#### 2.3.2 Internal Exploration
Internal Exploration primarily refers to large reasoning models (RLLMs) actively exploring and optimizing reasoning paths through their internal mechanisms (usually reinforcement learning strategies and reward mechanisms), allowing for more efficient and deeper solutions to complex reasoning problems. This method enables the model to autonomously adjust its reasoning strategy, reducing reliance on external guiding data.

---

**RL Strategies**
- **Core Idea:** Leverage reinforcement learning (RL) algorithms to guide models in actively learning and exploring diverse reasoning paths. This approach overcomes the limitations of overly uniform patterns in reasoning processes, enhancing model performance in tasks that involve high uncertainty or heavily depend on autonomous decision-making.
- **Representative Works:**
  - [PPO(Proximal Policy Optimization)](https://arxiv.org/abs/1707.06347): A classic RL algorithm that efficiently optimizes a model's internal decision-making mechanism through a policy-gradient-based approach, suitable for path exploration and optimization in complex environments.
  - [DivPO(Diverse Preference Optimization)](https://arxiv.org/abs/2501.18101): Encourages models to explore a greater variety of reasoning paths to maintain decision diversity, preventing convergence to local optima.
  - [GRPO(Guided Reward Policy Optimization)](https://arxiv.org/pdf/2402.03300): Designs a guided reward mechanism that enables models to more effectively explore within complex logical reasoning spaces.

---

**Reward Strategies**

- **Core Idea:** Directly guide models to explore and optimize effective reasoning paths through carefully designed reward functions, which are particularly useful in scenarios where there are explicit optimization goals or specific reasoning bottlenecks to address.
- **Representative Works:**
  - [Deepseek-R1](https://arxiv.org/abs/2501.12948): Proposes a specially designed reward function to incentivize models to optimize intermediate reasoning steps, aiding the model in building a high-quality internal reasoning process.
  - [ReST-MCTS*](https://arxiv.org/abs/2406.03816): Combines Monte Carlo Tree Search (MCTS) with reward strategies, guiding the tree search algorithm through process rewards for more accurate exploration of effective reasoning paths, improving overall reasoning quality.

---

#### 2.3.3 External Exploration
External exploration refers to the assistance of external tools, human knowledge, or other models in guiding the model to more effectively explore diverse reasoning paths and improve its ability to solve complex problems. This approach is often used in scenarios where fine-grained guidance or external knowledge is essential for effective problem-solving. External exploration can be subdivided into two types: Human-driven Exploration and Model-driven Exploration.

---

**Human-driven Exploration**
- **Core Idea:** Utilizes human intuition, experience, or feedback to guide the model in selecting and adjusting reasoning paths, especially in situations where the model's autonomous exploration ability is limited or the reasoning task is complex and requires decomposition into multiple sub-tasks.
- **Representative Works:**
  - [Least-to-Most](https://arxiv.org/abs/2205.10625): Breaks down complex problems into simpler subproblems, solving each and using prior answers as inputs for subsequent steps, ultimately synthesizing a solution for the overall problem. This method was proposed to address the "difficulty generalization" bottleneck in traditional Chain-of-Thought approaches.
  - [ToT (Tree-of-Thought)](https://arxiv.org/abs/2305.10601): Expands the traditional "left-to-right" token generation reasoning process into a "tree structure exploration," where each node represents a thought unit. This supports multi-path attempts, backtracking, forward reasoning, and self-evaluation within the reasoning process.

---

**Model-driven Exploration**

- **Core Idea:** Uses auxiliary models or algorithms to automatically guide the current model's reasoning process, reducing the need for human intervention and allowing for the efficient search and optimization of numerous complex reasoning paths, thus improving automation and overall efficiency.
- **Representative Works:**
  - [PPO-MCTS](https://arxiv.org/abs/2309.15028): Integrates MCTS (Monte Carlo Tree Search) with PPO-based training to enhance reasoning. The key is to retain the value network obtained during PPO training and use it during the reasoning phase to guide the MCTS in selecting more desirable output sequences, thereby improving the quality and consistency of the generated text.
  - [MindStar](https://arxiv.org/abs/2405.16265):  Reformulates complex reasoning problems (particularly mathematical ones) as search problems, where structured searches are performed over different reasoning paths to select the optimal one.
  - [rStar-Math](https://arxiv.org/abs/2501.04519):  Develops a strong mathematical reasoning system through MCTS + small model reward mechanisms + self-evolution processes, enabling small models to outperform o1-preview in mathematical capabilities.

---

## 3. Key Phenomena and Related Principles

<img src="./assets/images/analysis.jpg" style="width: 580pt">

### 3.1 Reasoning Emergence Phenomenon

Long CoT abilities naturally emerge after training, demonstrated by the model's ability to generate multi-step, coherent reasoning processes by internalizing logical structures and contextual examples from pretraining data, even in the absence of direct supervision. Related studies have described this phenomenon as follows:

- [Wang et al.](https://aclanthology.org/2023.acl-long.153/) found that a small number of high-quality contextual examples can effectively guide RL-based language models (RLLMs) to generate clear, logically consistent reasoning chains, indicating that the model has internalized basic reasoning patterns during pretraining.
- [Madaan et al.](https://aclanthology.org/2023.findings-emnlp.0/) demonstrated that even without specific problem entities, the model can still generate reasonable reasoning chains by retaining only the logical structure information, showcasing its inductive and transfer abilities regarding structural information.
- [Stechly et al.](https://openreview.net/forum?id=kPBEAZU5Nm) pointed out that by adjusting decoding strategies or constructing specialized prompts, latent  CoT abilities within the model can be explicitly activated, resulting in multi-step reasoning in complex tasks.
- [Guo et al.](https://arxiv.org/abs/2501.12948) showed that rule-based RL strategies can directly induce models to form coherent reasoning chains during pretraining, significantly improving performance in multi-step tasks.

### 3.2 Reasoning Boundary Phenomenon

Large language models exhibit clear performance boundaries in Long CoT: when the depth or complexity of reasoning exceeds a certain threshold, model performance significantly degrades, sometimes even resulting in logical collapse. This phenomenon suggests that current models have a "reasoning boundary," which is the upper limit of reasoning complexity that can be supported by their parameter space and computational resources. Existing research has systematically explored this phenomenon from both theoretical modeling and empirical analysis:

- [Chen et al.](https://openreview.net/forum?id=pC44UMwy2v.) formally introduced the "reasoning boundary" concept, experimentally quantifying the performance critical points of models under different task complexities, and indicating that accuracy sharply declines when the reasoning task exceeds the model's capacity.
- [Bi et al.](https://ojs.aaai.org/index.php/AAAI/article/view/29721) observed that performance deteriorates drastically when the model attempts to mimic overly complex CoT examples in code generation tasks, indicating that beyond a certain complexity, Long CoT examples become counterproductive.
- [Feng et al.](https://arxiv.org/abs/2305.15408) proposed a mathematical model showing that models with fixed parameter sizes cannot perform numerical calculations exceeding a certain complexity, revealing a hard limit in accuracy.
- [Zhou et al.](https://arxiv.org/abs/2502.05252) constructed the GSM-Infinite dataset and demonstrated through experiments that the upper limits of reasoning abilities vary significantly across different tasks, further emphasizing that reasoning boundaries are related to task structure.

### 3.3 Overthinking Phenomenon

In Long CoT, extending the reasoning chain does not always lead to performance improvement. Studies have shown that once the reasoning length exceeds the model’s capacity, accuracy decreases, a phenomenon known as "overthinking," which reflects the non-linear marginal benefits of reasoning and error accumulation in the process.

- [Chen et al.](https://openreview.net/forum?id=pC44UMwy2v.) found that when the number of reasoning steps exceeds the model’s boundary, reasoning accuracy significantly drops, indicating that there is an optimal depth range for reasoning.
- [Wolf et al.](https://arxiv.org/abs/2409.18028) emphasized that the fundamental reason for performance degradation is the amplification of errors in intermediate reasoning steps, which affects the final judgment.
- [Xie et al.](https://arxiv.org/html/2502.14768v1) experimentally showed that reasoning length does not have a monotonic relationship with accuracy, challenging the intuition that "longer CoT leads to better reasoning."
- [Wu et al.](https://arxiv.org/abs/2502.07266) established a mathematical model defining the "optimal reasoning length" interval under different models and task conditions, suggesting that performance reverses once the length exceeds the optimal range.
- [Chen et al.](https://arxiv.org/abs/2502.03325) introduced the "reasoning chain Ohm’s law," analogizing the non-linear relationship between reasoning length and performance to information flow resistance in the model.

### 3.4 Inference Test-Time Scaling Phenomenon

The inference test-time scaling phenomenon refers to the increase in reasoning performance by extending the computational process (e.g., reasoning chain length or sample number) during inference. This phenomenon reveals the "dynamic amplification" potential of the model, but also comes with a trade-off between exploration depth and computational cost.

- [Brown et al.](https://arxiv.org/abs/2407.21787) observed that by repeating multiple rounds of inference attempts, even if the initial attempt fails, the correct answer can be found within a certain number of trials, introducing the "language monkey" phenomenon.
- [o1](https://arxiv.org/abs/2412.16720) demonstrated that simply increasing the reasoning chain length improves accuracy, particularly in complex mathematical tasks.
- [Jin et al.](https://aclanthology.org/volumes/2024.findings-acl/) pointed out that while increasing the reasoning chain length initially leads to performance improvement, beyond a certain threshold, performance deteriorates, resulting in a typical nonlinear growth curve.
- [Wu et al.](https://arxiv.org/abs/2408.00724) found that there is a logarithmic relationship between the number of inference samples and the lower bound of error, suggesting an asymptotic relationship between computational complexity (FLOPs) and inference performance.
- [Chen et al.](https://arxiv.org/abs/2502.03325) established the theoretical upper-bound of parallel inference, indicating that no matter how the sample size is increased, the model’s verification performance cannot exceed its internal reasoning ceiling.

### 3.5 PRM and ORM Selction Phenomenon

In reinforcement learning optimization, Long CoT tasks involve supervision of the model generation process. Researchers distinguish between two main strategies: Process Reward Model (PRM), which focuses on the reasoning process itself, and Outcome Reward Model (ORM), which only concerns whether the final output is correct. The two strategies differ significantly in terms of generalization ability, learning stability, and supervision cost.

- [Lampinen et al.](https://aclanthology.org/2022.findings-emnlp.38) validated the causal relationship between intermediate steps and final answers in qualitative experiments, providing theoretical support for the rationale behind process supervision.
- [Jia et al.](https://arxiv.org/abs/2502.10581) theoretically proved that under sufficiently diverse data, ORM is not harder to optimize than PRM, with the two only differing by polynomial factors in terms of sample complexity.
- [Guo et al.](https://arxiv.org/abs/2501.12948) showed that rule-based PRM reinforcement learning significantly improves the model's Long CoT capabilities in complex tasks but also faces the risk of reward hacking.
- [Tan](https://aclanthology.org/2023.blackboxnlp-1.12.) emphasized the importance of reward distribution in intermediate reasoning steps for complex reasoning paths, which ORM cannot provide.
- [Jiang et al.](https://arxiv.org/abs/2501.03124) pointed out that PRM is more costly in terms of data collection, as it requires labeling each reasoning step, limiting its large-scale application.

### 3.6 Aha Moment Phenomenon

The Aha Moment refers to the sudden integration of information during the reasoning process, leading to a key turning point in judgment, resembling human reflection and self-correction. This phenomenon highlights the model's dynamic cognitive adjustment abilities, but its occurrence depends on the collaboration between external stimuli and internal mechanisms.

- [Guo et al.](https://arxiv.org/abs/2501.12948) first triggered the Aha Moment behavior under unsupervised conditions through rule-based rewards, with models reflecting on intermediate reasoning and self-correcting.
- [Xie et al.](https://arxiv.org/abs/2502.14768) further demonstrated through experiments that this behavior can be replicated across multiple models, verifying that it is not an偶然 event but rather an inducible strategy.
- [Zhou et al.](https://arxiv.org/abs/2503.05132) extended the Aha Moment phenomenon to multimodal tasks, showing that it is not specific to text-based tasks but reflects the model's broader cognitive abilities.
- [Liu et al.](https://oatllm.notion.site/oat-zero) pointed out that in certain reinforcement learning frameworks (e.g., R1-Zero), Aha behavior may not genuinely exist, with lengthening the generation more likely a result of reward optimization rather than actual reflection.
- [Yang et al.](https://arxiv.org/abs/2504.02956) found that the Aha behavior often involves human-like language enhancement and dynamic uncertainty regulation, with the model more inclined to use expressions like "I think" under high-pressure tasks, reflecting its coping mechanisms for task stress.




## 4. Algorithms

### 4.1 Supervised Fine-Tuning (SFT)

In advancing large models to possess powerful Long CoT reasoning abilities, Supervised Fine-Tuning (SFT) plays a crucial role, bridging pre-training with more advanced alignment methods such as Reinforcement Learning from Human Feedback (RLHF). The core goal of SFT is to teach models how to follow instructions and initially master the ability to generate structured, step-by-step reasoning chains, thus laying the foundation for more complex reasoning tasks.



- **In the context of deep reasoning, SFT is especially critical.**
Although the lack of sufficient reasoning depth in RLLMs significantly reduces performance, SFT stabilizes the model’s output format through a memorization process, allowing it to learn reasoning from human-labeled or distilled data. In contrast to reinforcement learning (RL), which focuses more on generalization and self-learning, SFT plays a vital role in deep reasoning imitation. It allows RLLMs to learn complex reasoning patterns by mimicking high-quality reasoning examples generated by humans, advanced RLLMs, or enhanced RLLMs, and generalizing them to new tasks. SFT not only significantly improves the model’s reasoning performance but, in some cases, enables even a small number of high-quality samples to activate the underlying LLM's deep reasoning capabilities, allowing it to predict events outside the model's knowledge base. This makes SFT one of the key technologies for enhancing reasoning levels and generalization abilities in RLLMs.

- **For feasible reflection, SFT primarily focuses on optimization-based imitation (Refinement Imitation).**
In reflection-based LLM reasoning, SFT is a key mechanism for enabling self-optimization and error correction in the model. Through SFT, the model can directly learn the error-correction processes of advanced LLMs, significantly enhancing its reflective abilities, such as performing self-play reasoning, iterative feedback error correction, and even justifying and reflecting on the reasoning process through incremental natural language feedback. Additionally, SFT can integrate visual and textual reasoning in multimodal scenarios, improving the model’s critical thinking and self-correction abilities. SFT enhances the reasoning accuracy of LLMs through iterative feedback and self-correction strategies, which is especially beneficial for smaller models.


#### 4.1.1 Core Technology

SFT consists of two core concepts: **Instruction Tuning** and **Parameter-Efficient Fine-Tuning, PEFT**。

**Instruction Tuning**
-   **Core Idea:** By fine-tuning the model on a large number of instructions covering various tasks, the model’s zero-shot generalization ability on unseen tasks can be significantly enhanced. This enables the model to learn the skill of "following instructions."
-   **Representative Works:**
    -   [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652): Google’s pioneering work demonstrating that multi-task instruction fine-tuning unlocks zero-shot capabilities in LLMs for unseen tasks.
    -   [Instruction Tuning for Large Language Models: A Survey](https://arxiv.org/abs/2308.10792): A comprehensive survey systematically introducing methods, datasets, challenges, and future directions in instruction tuning.



**参数高效微调(PEFT)**

- **Core Idea:** Given the high cost of full fine-tuning (Full Fine-tuning) for LLMs, PEFT methods have emerged. These methods achieve near-full fine-tuning effects by updating only a small subset of the model’s parameters, greatly reducing hardware requirements.
-   **Representative Works:**
    -   [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685): A revolutionary LoRA technique proposed to efficiently fine-tune models by injecting low-rank adaptation matrices, currently one of the most widely used PEFT methods.
    -   [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314): A further optimization of LoRA, combining 4-bit quantization, double-weight quantization, and page optimizers, making it possible to fine-tune massive models on a single consumer-grade GPU.
    -   [Adapter Tuning](https://arxiv.org/abs/1902.00751): Inserts small neural network modules (adapters) between layers of the Transformer, updating only the parameters of these adapters during training.
    -   [Prompt Tuning](https://arxiv.org/abs/2104.08691) / [P-Tuning](https://aclanthology.org/2022.acl-short.8/): Instead of modifying the model’s weights, it learns one or more trainable virtual tokens (soft prompts) at the input end, guiding the model to perform downstream tasks more effectively.

**Technical Comparison**

| Technology Type                            | Core Idea                                                                                                     | Advantages                                                                                                    | Disadvantages                                                                                                             |
| :----------------------------------------- | :------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------ |
| **Full Fine-tuning**                       | Update all model weights.                                                                                     | Highest performance ceiling, can fully adapt to new data.                                                     | Extremely high training cost (memory, time), prone to catastrophic forgetting, requires storing the entire model.         |
| **Parameter-Efficient Fine-tuning (PEFT)** | Freeze most of the original parameters, only update a small set of additional parameters or specific subsets. | Very low training cost, fast, resistant to forgetting, small fine-tuning products (Adapters), easy to deploy. | Performance may be slightly inferior to full fine-tuning, and its adaptation to extremely complex tasks might be limited. |

#### 4.1.2 Learning Resources

| Resource Name                 | Speaker/Author    | Features                                                                                                                                                          | Link                                                                                |
| :---------------------------- | :---------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------- |
| Let's build GPT: from scratch | Andrej Karpathy   | A hands-on guide to building GPT from scratch, deeply understanding the fundamentals of Transformer and training processes; a prerequisite for understanding SFT. | [YouTube](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) |
| Hugging Face SFT Course       | Hugging Face      | Official SFT series tutorial, using the Hugging Face TRL codebase for SFT code practice.                                                                          | [Course Link](https://huggingface.co/learn/llm-course/chapter11/1)                  |
| Hugging Face SFT Trainer Doc  | Hugging Face      | Advanced documentation for Hugging Face SFTTrainer.                                                                                                               | [Documentation Link](https://huggingface.co/docs/trl/sft_trainer)                   |
| Hugging Face PEFT Course      | Hugging Face      | Official PEFT series tutorial, explaining the theory and code practices of various efficient fine-tuning techniques like LoRA.                                    | [Course Link](https://huggingface.co/docs/peft/index)                               |
| LLMs-from-scratch             | Sebastian Raschka | Tutorial code for the official book, "Build a Large Language Model (From Scratch)."                                                                               | [Course Link](https://github.com/rasbt/LLMs-from-scratch)                           |

#### 4.1.3 Development Frameworks

| Framework            | Features                                                                                                                                                               | Main Use Case                                                                                            | Resource Link                                      |
| :------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------- | :------------------------------------------------- |
| **Hugging Face TRL** | Official Hugging Face library, integrating various training methods like SFT, RLHF, DPO, seamlessly connecting with ecosystems (`transformers`, `peft`, `accelerate`). | Provides the standardized SFT trainer `SFTTrainer`, simplifying the training process.                    | [GitHub](https://github.com/huggingface/trl)       |
| **LLaMA-Factory**    | One-stop LLM fine-tuning platform with a Web UI, enabling users with no coding experience to easily perform SFT, PEFT, and model evaluation.                           | Highly user-friendly, supports massive models and datasets, suitable for beginners and quick validation. | [GitHub](https://github.com/hiyouga/LLaMA-Factory) |



#### 4.1.4 Best Practices and Common Pitfalls

1.  **Data Quality is Far More Important than Quantity:**
    -   **Core Principle**: It is better to use 1,000 high-quality, diverse data points than 10,000 low-quality, homogeneous ones. Low-quality data can teach the model incorrect patterns.
    -   **Format Consistency**: Ensure that all training data follows a unified dialogue template (e.g.,[ChatML](https://huggingface.co/docs/transformers/main/en/chat_templating)), which is crucial for training the model to recognize roles and dialogue boundaries.
2.  **Choose the Right Fine-Tuning Strategy**:
    -   For most applications with limited resources, **QLoRA** should be prioritized as it strikes the best balance between efficiency and effectiveness.
    -   If seeking optimal performance and with sufficient resources, **full fine-tuning** may be considered, though care should be taken to avoid the risk of overfitting.
3.  **Tuning Key Hyperparameters**:
    -   **Learning Rate**: SFT typically uses smaller learning rates than pretraining, usually ranging from `1e-5` to `5e-5`.
    -   **Epochs**: Usually, 1 to 3 epochs are sufficient. Too many epochs can lead to overfitting on small datasets, causing the model to "forget" the general knowledge learned during pretraining.
    -   **Batch Size**: Within the memory limits, increasing the batch size appropriately helps stabilize training.
4.  **Evaluation and Iteration**:
    -   **Comprehensive Evaluation**: Do not rely solely on the loss function. Combine it with **objective evaluation** benchmarks (such as MMLU) and subjective **human evaluation** for a more thorough assessment of model performance.
    -   **Iterative Optimization**: SFT is an ongoing iterative process. Based on evaluation results, continuously clean the data, adjust hyperparameters, and optimize the model.

#### 4.1.5 Relevant Paper Repositories
- [LLM4NLP](https://github.com/LightChen233/Awesome-LLM-for-NLP)

### 4.2 Reinforcement Learning

#### 4.2.1 Reinforcement Learning

- [West Lake University’s "Mathematical Principles of Reinforcement Learning"](https://www.bilibili.com/video/BV1sd4y167NS/)
  - Features: Starts with MDP and the Bellman Equation, derived using the policy gradient theorem.
  - Prerequisite Knowledge: Linear algebra, probability theory.
  - Focus: The mathematical essence of value iteration and policy optimization.
- [Book-Mathematical-Foundation-of-Reinforcement-Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning) (Beginner-friendly)

#### 4.2.2 Core Algorithms of Reinforcement Learning

**Authoritative Courses**

| Course                                        | Lecturer      | Features                                                                 | Resources                                                                      |
| --------------------------------------------- | ------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Foundations of Deep RL                        | Pieter Abbeel | 6 concise lectures (Q-learning → PPO)                                    | [YouTube](https://youtube.com/playlist?list=PLkFD6_40KJIwhWpGazJ9VSj9CFMkb79A) |
| UC Berkeley CS285                             | Sergey Levine | Includes SAC/Inverse Reinforcement Learning and other advanced topics    | [Course Website](http://rail.eecs.berkeley.edu/deeprlcourse/)                  |
| Reinforcement Learning by Hung-yi Lee |  Hung-yi Lee     | In Chinese + Practical Exercises with EasyRL                             | [Bilibili](https://www.bilibili.com/video/BV1UE411G78S)                        |
| Reinforcement Learning: An Overview           | Kevin Murphy  | Continuously updated resources on Deep Reinforcement Learning algorithms | [Arxiv](https://arxiv.org/abs/2412.05265)                                      |

**Essential Basic Algorithms**

- **Basic Reinforcement Learning Algorithms**

  - **DQN**: The beginning of Deep Reinforcement Learning
  - **PPO**: A key method in policy optimization, widely used in industrial applications
  - **SAC**: Incorporates exploration entropy, robust for continuous action spaces
  - **TD3**: Improves off-policy reinforcement learning using double delayed networks

- **Model-Based Reinforcement Learning Algorithms**

  - **[dreamer](https://github.com/google-research/dreamer)**: A model-based reinforcement learning algorithm
  - **[tdmpc2](https://github.com/nicklashansen/tdmpc2)**: Significant advancement in model-based reinforcement learning algorithms

- **Offline Reinforcement Learning Algorithms**

  - **[CQL](https://github.com/aviralkumar2907/CQL)**: Introduces conservative constraints, foundational work in offline reinforcement learning
  - **[decision-transformer](https://github.com/kzl/decision-transformer)**: Introduces autoregressive models into offline reinforcement learning

- **Large-Scale Model Reinforcement Learning Algorithms**

  - **PPO**: Applying classic PPO to large language models
  - **DPO**: Preference optimization without rewards, an offline reinforcement learning algorithm for large models
  - **GRPO**: Group Relative Policy Optimization, core algorithm in DeepSeek-R1

**Cutting-Edge Algorithms for Large-Scale Model Reinforcement Learning**

- **[DAPO](https://github.com/BytedTsinghua-SIA/DAPO)**: Four improvements on GRPO
- **[LUFFY](https://github.com/ElliottYan/LUFFY)**: Off-policy version of GRPO, introduces high-quality external trajectories
- **[Absolute-Zero-Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner)**: A large-scale reinforcement learning algorithm requiring no annotations
- **[One-Shot-RLVR](https://github.com/ypwang61/One-Shot-RLVR)**: One-shot optimization for large model inference
- **[SPIRAL](https://github.com/spiral-rl/spiral)**: Reinforcement learning in self-play game environments, successfully enhancing mathematical reasoning abilities
- **[High-Entropy Minority Tokens Drive Effective RLVR](https://shenzhi-wang.github.io/high-entropy-minority-tokens-rlvr/)**: Reinforcement learning driven by high-entropy tokens (20%)
- **[Spurious\_Rewards](https://github.com/ruixin31/Spurious_Rewards)**: Random rewards can also enhance LLM reasoning abilities
- **[SwS](https://github.com/MasterVito/SwS)**: Reasoning reinforcement learning driven by self-perceived weaknesses

#### 4.2.3 Development Frameworks for Reinforcement Learning

**Basic Reinforcement Learning Frameworks**

- **[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)** (Quick experimentation, with well-established and stable baselines)
- **[legged\_gym](https://github.com/leggedrobotics/legged_gym)** (Quadruped robot control)

**Large-Scale Model Reinforcement Learning Frameworks**

- **[verl](https://github.com/volcengine/verl)**: A high-performance and user-friendly open-source reinforcement learning training library based on Ray, vLLM, ZeRO-3, and HuggingFace Transformers, with features like efficient resource utilization, scalability, and production readiness. (Complex structure, highly reusable, excellent performance)
- **[OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF)**: An open-source RLHF framework released by teams such as NVIDIA, based on Ray, vLLM, ZeRO-3, and HuggingFace Transformers. It supports algorithms like PPO, GRPO, and REINFORCE++, and provides dynamic sampling and asynchronous agent mechanisms to accelerate training.
- **[AReaL](https://github.com/inclusionAI/AReaL)**: Asynchronous reinforcement learning framework
- **[ROLL](https://github.com/alibaba/ROLL)**: Supports training large models with 600+ billion parameters
- **[Hugging Face TRL](https://github.com/huggingface/trl)**: An RLHF full-stack library maintained by Hugging Face, integrating SFT, GRPO, DPO, Reward Modeling, and other modules. It supports multiple model architectures and distributed scaling, making it one of the most active RLHF tools in the community. (User-friendly, quick start, active community)
- **[RL4LMs](https://github.com/allenai/RL4LMs)**: An open-source RLHF library for language models, providing end-to-end tools for reward model construction and policy network training, helping researchers quickly build custom RLHF pipelines.

Additionally, there are some interesting extension repositories:

- **[Sachin19/trlp](https://github.com/Sachin19/trlp)**: An end-to-end RLHF library based on the TRL stack, supporting not only language models but also extending to Stable Diffusion models. It includes steps like SFT, reward modeling, and PPO, with example code for experimentation.
- **[OpenRLHF-M](https://github.com/OpenRLHF/OpenRLHF-M)**: An extension of OpenRLHF, optimized for multimodal models. It leverages DeepSpeed and HuggingFace Transformers to achieve higher throughput and richer training scenarios.
- **[HumanSignal-RLHF](https://github.com/HumanSignal/RLHF)**: An archived resource repository, gathering links and tutorials on RLHF data collection, system construction, and best practices, suitable for beginners to quickly understand the full RLHF pipeline.
- **[MichaelEinhorn/trl-textworld](https://github.com/MichaelEinhorn/trl-textworld)**: A derivative version of TRL focused on performing RLHF experiments in the TextWorld environment, demonstrating how to train models like GPT2 using PPO to generate text that meets specific feedback requirements.


### 4.2.4 Test Environments

**Classical RL Tests**

- **OpenAI Gym**: Classic Control

| Environment ID   | Task Description                       | Features                                                                             |
| ---------------- | -------------------------------------- | ------------------------------------------------------------------------------------ |
| `CartPole-v1`    | Balance an inverted pendulum           | 4-dimensional state/discrete actions, termination if pole tilts > 12° or steps ≥ 500 |
| `MountainCar-v0` | Swing car to the top                   | 2-dimensional state/discrete actions, requires potential energy swing                |
| `Pendulum-v1`    | Control pendulum to stay vertical      | 3-dimensional state/continuous actions, no physical termination condition            |
| `Acrobot-v1`     | Swing double-link to touch target line | 6-dimensional state/discrete actions, termination when target line is touched        |

- **Atari 2600**: Games

| Environment ID     | Game Type      | Challenges                                                 |
| ------------------ | -------------- | ---------------------------------------------------------- |
| `Pong-v5`          | Ping Pong      | 210×160 RGB input, requires image preprocessing            |
| `Breakout-v5`      | Breakout       | Dense rewards, suitable for DQN training                   |
| `SpaceInvaders-v5` | Space Invaders | Multiple enemies coordinated attack, complex reward system |

- **Box2D**: Physics Simulation

| Environment ID     | Physics System | Core Challenges                                                        |
| ------------------ | -------------- | ---------------------------------------------------------------------- |
| `LunarLander-v2`   | Lunar Lander   | 8-dimensional state/discrete actions, fuel control and precise landing |
| `BipedalWalker-v3` | Bipedal Walker | 24-dimensional state/continuous actions, balancing on complex terrain  |
| `CarRacing-v2`     | Car Racing     | 96×96 RGB input, vision + continuous control combined                  |

- **MuJoCo**: Robotic Control

| Environment ID   | Robot Model    | Task Type                                          |
| ---------------- | -------------- | -------------------------------------------------- |
| `HalfCheetah-v4` | Cheetah Robot  | High-speed running control (17-dimensional state)  |
| `Ant-v4`         | Ant Robot      | Complex terrain navigation (111-dimensional state) |
| `Humanoid-v4`    | Humanoid Robot | Bipedal balance walking (376-dimensional state)    |

- **Other Special Environments**

| Category      | Example Environment | Application Area                    |
| ------------- | ------------------- | ----------------------------------- |
| Text Game     | `TextFlappyBird-v0` | RL based on character interfaces    |
| Multi-agent   | `PistonBall-v6`     | Multi-agent cooperation/competition |
| 3D Navigation | `AntMaze-v4`        | Complex maze path planning          |

**Extended Resources**:

- **Safe RL**: `Safety-Gymnasium` (task with constraints)
- **Autonomous Driving**: `CARLA`/`AirSim` (high fidelity simulation)
- **Multi-agent**: `PettingZoo` (compatible with Gymnasium API)

> 💡 Full environment list can be found at:
> [Gymnasium Documentation](https://gymnasium.farama.org/) | [OpenAI Gym Wiki](https://github.com/openai/gym/wiki/Table-of-environments)

**Large Model RL Tests**

| Environment   | Purpose                                      |
| ------------- | -------------------------------------------- |
| Math-500      | Mathematical reasoning                       |
| AIME2024/2025 | Mathematical competition                     |
| AMC           | Mathematical competition                     |
| GPQA          | PhD-level biophysics and chemistry reasoning |

### 4.3 Agent

The ability of LLM Agents to solve complex problems fundamentally relies on their reasoning and planning capabilities. The core mechanism of this ability is Long CoT, which breaks down complex tasks into smaller, logical steps. The characteristics of Long CoT, particularly its depth of inference, extensive exploration, and feasibility reflection, are not just additional features but the foundation for realizing these abilities. If an agent cannot "think longer" and engage in a "thinking-critique-improvement" cycle, its ability to make independent decisions and adapt in unfamiliar scenarios will be severely limited, causing it to revert to "predefined pipelines" or "iterative interactions with humans." Models such as o1 and DeepSeek-R1 have made breakthroughs in using Long CoT to solve complex tasks, directly proving this causal relationship: enhanced reasoning depth directly leads to an improvement in agent capabilities (autonomy in complex tasks). Therefore, the future development of AI agents will be closely linked to breakthroughs in Long CoT.

**AI Agent Online Courses and Resources**

- [Andrew Ng's "How to Build, Evaluate, and Iterate LLM Agents"](https://www.bilibili.com/video/BV1Ew4m1R7ju/?vd_source=a39056a294c1d415f3413ef933024e2b): A seminar by LlamaIndex and TruEra team experts (March 2024), explaining how to build LLM agents using tool frameworks like LlamaIndex and evaluate agent performance, detect hallucinations and biases using observability tools like TruLens. The video provides both Chinese and English subtitles, making it suitable for learning about agent development and evaluation methods in production environments.
- [Coursera AI Agent Developer Specialization (Vanderbilt University)](https://www.coursera.org/specializations/ai-agents): A series of 6 courses for beginners with Python experience, focusing on building and deploying intelligent AI agents using Python, tools, memory, and reasoning. Topics include creating custom GPTs, applying prompt engineering, designing reliable AI systems, and implementing multi-agent collaboration systems.
- [Hugging Face Agent Course](https://huggingface.co/learn/agents-course/unit0/introduction): A free online course introducing agents.

**Open Source Frameworks for Building LLM AI Agents**

- [LangChain](https://github.com/langchain-ai/langchain): The most widely used framework for LLM agent development, offering a modular and extensible architecture, unified LLM interfaces, pre-built agent toolkits (for CSV, JSON, SQL), Python and Pandas integration, and vector storage capabilities. It supports React-style agents and provides a memory module to maintain context.
- [CrewAI](https://github.com/crewAIInc/crewAI): An open-source framework for orchestrating role-playing AI agents, emphasizing multi-agent collaboration through defined roles and shared goals. It is independent, streamlined, and offers deep customization, supporting "Crew" (team) and "Flow" (event-driven workflows).
- [Dify](https://github.com/langgenius/dify): An open-source framework for LLM applications with a visual prompt orchestration interface, long context integration, API-based development, multi-model support, and RAG pipelines.
- [OpenAI Agent Demo](https://github.com/openai/openai-cs-agents-demo): OpenAI's official platform for setting up Agent client services (visual platform, no additional code required).

For more frameworks, refer to [Awesome LLM Agent Frameworks](https://github.com/kaushikb11/awesome-llm-agents/blob/main/README.md).

**End-to-End RL Learning for Complex Agent Trajectories**

- [Agent-R1](https://github.com/0russwest0/Agent-R1): An open-source framework aimed at accelerating research and development at the intersection of RL and agents. It uses end-to-end reinforcement learning to train agents in specific environments, allowing developers to define domain-specific tools and reward functions without complex process engineering. It supports multi-round tool calls and multi-tool coordination.
- [RAGEN](https://github.com/RAGEN-AI/RAGEN): A framework for training LLM reasoning agents with RL in interactive, stochastic, and multi-round environments. It introduces the StarPO (State-Think-Act-Reward Policy Optimization) framework, which features staggered rollout and update phases for trajectory-level optimization.

**RL-enhanced Tool Use and Search Capabilities**

- [ReCall](https://github.com/Agent-RL/ReCall): A novel framework that trains LLMs for tool invocation reasoning with RL, without the need for supervised data on tool usage trajectories or reasoning steps. It is designed to enable LLMs to use and combine any user-defined tools in an agent-like manner.
- [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): An extension of the OpenManus framework, specifically focused on enhancing AI agents via RL techniques like GRPO, to enable training across multiple environments and performance tuning for specific tasks.
- [R1-Searcher](https://github.com/RUCAIBox/R1-Searcher), [Search-R1](github.com/PeterGriffinJin/Search-R1): Research exploring the use of RL to enhance the search capabilities of LLMs.


**Awesome Blog**

- [Neptune.ai blog](https://neptune.ai/blog/building-llm-agents-with-autogen): Provides a detailed step-by-step guide, such as "How to Build LLM Agents with AutoGen," covering components, RAG pipelines, planning, tools, and memory integration.
- [n8n.io blog](https://blog.n8n.io/llm-agents/): Offers insights into the capabilities of LLM agents (such as strategic planning, memory, and tool integration) and includes a practical tutorial on building agents.
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/an-easy-introduction-to-llm-reasoning-ai-agents-and-test-time-scaling/): Provides an introductory article on LLM reasoning and AI agents.
- [Botpress blog](https://botpress.com/blog/chain-of-thought): Explains chain-of-thought prompting and discusses various AI agent frameworks.
- [SuperAnnotate blog](https://www.superannotate.com/blog/llm-agents): Offers a comprehensive overview of LLM agents, their capabilities, and their future.
- [Smythos blog](https://smythos.com/developers/agent-development/llm-agents/): Discusses how LLM agents are revolutionizing task automation and AI integration.
- [Unite.ai](https://www.unite.ai/reinforcement-learning-meets-chain-of-thought-transforming-llms-into-autonomous-reasoning-agents/): Provides a detailed discussion on how reinforcement learning, combined with chain-of-thought, transforms LLMs into autonomous reasoning agents.
- [Holistic AI blog](https://www.holisticai.com/blog/llm-agents-use-cases-risks): Delves into the architecture of LLM agents, including multimodal enhancement, tool usage, and memory.
- [ProjectPro](https://www.projectpro.io/article/agentic-ai-design-patterns/1126) and [Lightrains blog](https://lightrains.com/blogs/ai-agent-design-patterns-cxo/): Discuss various AI agent design patterns, including reflection, tool usage, and planning patterns.

**Awesome GitHub Repositories**

- [Awesome-LLM-Agents](https://github.com/kaushikb11/awesome-llm-agents/blob/main/README.md): A curated list of various LLM agent frameworks, serving as a valuable starting point for exploring the ecosystem.
- [Awesome-LLM-Agents-Scientific-Discovery](https://github.com/zhoujieli/Awesome-LLM-Agents-Scientific-Discovery): A curated list of papers focused on LLM-driven AI agents' applications in biomedical research and broader scientific discovery.
- [Awesome-Agent-RL](https://github.com/0russwest0/Awesome-Agent-RL): A specialized collection of papers and resources focused on unleashing the potential of AI agents through reinforcement learning.
- [Awesome-LLM-APPs](https://github.com/Shubhamsaboo/awesome-llm-apps): A curated collection of excellent LLM applications built using RAG, AI agents, multi-agent teams, MCP, speech agents, and more.
<!-- ## 4. Behavior Analysis \& Rationale -->


## 5. Datasets

### 5.1 Benchmarks

#### 5.1.1 Evaluation Frameworks

- **LLM Evaluation Frameworks:**

  - [OpenCompass](https://github.com/open-compass/opencompass) is a comprehensive evaluation platform for large language models (LLMs) that supports assessments of a wide range of open and closed models across over 100 datasets. It covers multiple dimensions such as language understanding, reasoning, and code generation, and supports various evaluation modes, including zero-shot, few-shot, and Chain-of-Thought (CoT), as well as distributed evaluation capabilities.
  - [DeepEval](https://github.com/confident-ai/deepeval) is an easy-to-use, open-source LLM evaluation framework designed for evaluating and testing large language model systems. It aims to help developers efficiently assess the quality of model-generated content based on key metrics such as relevance, factual consistency, bias, and toxicity. Its usage is similar to the Python unit testing framework, Pytest.

- **MLLM Evaluation Frameworks:**

  - [VLMEvalKit](https://github.com/open-compass/vlmevalkit) is an open-source toolkit launched by OpenCompass specifically designed for the evaluation of large vision-language models. It supports one-click evaluations for over 220 vision-language models across more than 80 benchmark tests, covering tasks such as image question answering, image-text matching, and visual reasoning. It provides evaluation results based on both exact matching and LLM-based answer extraction.
  - [EvalScope](https://github.com/modelscope/evalscope) is a model evaluation framework introduced by the MoTower community, supporting performance benchmarking for various types of models, including large language models, multimodal language models, embedding models, and AIGC models.

- **CoT Evaluation Frameworks:**

  - [ROSCOE](https://github.com/facebookresearch/ParlAI/tree/main/projects/roscoe) aims to provide a set of automated metrics to evaluate the reasoning quality of models without requiring reference answers.
  - [ReCEval](https://github.com/archiki/ReCEval) is a reasoning chain evaluation framework proposed by Archiki Prasad and colleagues. It aims to provide a detailed analysis of the multi-step reasoning process generated by large language models through two dimensions: "correctness" and "informativeness."

#### 5.1.2 Outcome Benchmarks

This section focuses on evaluating the final performance of Long CoT reasoning from a holistic perspective, emphasizing whether the reasoning chain is ultimately sound and accurate.

- **Complex Mathematics**
  
|      Name     | Number of Problems | Release Date |                                     Authors                                     |                                                                                                                                                                                               Description                                                                                                                                                                                              |                                                          Relevant Links                                                         |
| :-----------: | :----------------: | :----------: | :-----------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------: |
|     GSM8K     |       \~8,500      |     2021     |                                      OpenAI                                     |                                                                                                     A dataset of K-12 math word problems provided by OpenAI, each with detailed solution steps. The problems cover basic arithmetic, word problems, etc., requiring multi-step reasoning to solve.                                                                                                     |                                    🤗[dataset](https://huggingface.co/datasets/openai/gsm8k)                                    |
|      MATH     |       12,500       |     2021     |                          Hendrycks et al. (UC Berkeley)                         |                                                                           A dataset of challenging math problems from math competitions, each accompanied by a complete step-by-step solution. It includes topics such as algebra, geometry, and probability, designed to evaluate models' mathematical reasoning abilities.                                                                           |                                        🌐[repository](https://github.com/hendrycks/math)                                        |
|   AIME 2024   |         30         |     2024     |                               AI-MO Project Group                               |                                                                                     American Invitational Mathematics Examination 2024, a high-level high school math competition dataset, including all questions from AIME I and II of 2024. The problems focus on integer solutions and combinatorial reasoning.                                                                                    |                             🤗[dataset](https://huggingface.co/datasets/AI-MO/aimo-validation-aime)                             |
|   AIME 2025   |         30         |     2025     |                                   OpenCompass                                   |                                                                                                                     A collection of problems from AIME 2025 I & II. The difficulty is similar to AIME 2024, assessing high school students' complex math problem-solving abilities.                                                                                                                    |                                🤗[dataset](https://huggingface.co/datasets/opencompass/AIME2025)                                |
|    AMC 2023   |         83         |     2024     |                               AI-MO Project Group                               |                                                                                               American Mathematics Competitions 2023, a validation set consisting of 83 problems from the AMC12 competition. It includes questions from the 2022-2023 AMC12 covering topics such as algebra and geometry.                                                                                              |                              🤗[dataset](https://huggingface.co/datasets/AI-MO/aimo-validation-amc)                             |
|   USAMO 2025  |          6         |     2025     |                          Balunović et al. (ETH Zurich)                          |                                                                                             A dataset of problems from the USA Mathematical Olympiad 2025. These are final exam questions from the USAMO, typically difficult proof-based problems that test deep mathematical reasoning and proof skills.                                                                                             |                   🌐[website](https://matharena.ai/) <br> 🌐[repository](https://github.com/eth-sri/matharena)                  |
| OlympiadBench |        8,476       |     2024     |                     He Chaohui et al. (Tsinghua University)                     |                                                    A bilingual multimodal scientific problem dataset at the Olympiad level. It includes 8,476 problems from competitions in subjects like mathematics and physics, each with expert step-by-step solutions, used to comprehensively evaluate the model's cross-disciplinary deep reasoning ability.                                                    | 🤗[dataset](https://huggingface.co/datasets/Hothan/OlympiadBench) <br> 🌐[repository](https://github.com/OpenBMB/OlympiadBench) |
|  OlympicArena |       11,163       |     2024     | Huang Zhen et al. (Shanghai Jiao Tong University & Shanghai Research Institute) |                                    Also known as OlympiadArena, this comprehensive benchmark covers 62 types of “Olympiad” challenges across 7 categories such as mathematics, physics, chemistry, and biology. It contains 11,163 Olympiad-level problems, categorized by subject and problem type, designed to promote general artificial intelligence reasoning.                                    |   🤗[dataset](https://huggingface.co/datasets/GAIR/OlympicArena) <br> 🌐[repository](https://gair-nlp.github.io/OlympicArena)   |
|  Putnam-AXIOM |      236 + 52      |     2024     |                       Gulati et al. (Stanford University)                       |                                                                      A dataset from the Putnam Mathematics Competition, including 236 problems from the Putnam competition and 52 cross-problems from Putnam AIME. Each problem comes with detailed solution steps and is used to assess models' mathematical reasoning abilities.                                                                     |                                      📄[paper](https://openreview.net/forum?id=t1mAXb4Cop)                                      |
|  FrontierMath |          -         |     2024     |                             Glazer et al. (Epoch AI)                            | A collection of frontier mathematical problems collaboratively created by dozens of mathematicians. It covers major branches of modern mathematics, from number theory and real analysis to algebraic geometry. The problems require hours or even days to solve manually. Hundreds of original high-difficulty problems are included, all of which have not been published to avoid training leakage. |                                           📄[paper](https://arxiv.org/pdf/2411.04872)                                           |
|   ThinkBench  |        2,912       |     2025     |      Huang Shulin et al. (University of Science and Technology of Shanghai)     |                                                   A dynamic challenge set designed to evaluate the robust reasoning abilities of large language models (LLMs). It contains 2,912 reasoning tasks generated by applying out-of-distribution perturbations to existing problems, aiming to test the model's reasoning accuracy in unfamiliar contexts.                                                   |                                           📄[paper](https://arxiv.org/pdf/2502.16268)                                           |
|  MATH-Perturb |      279 \* 2      |     2025     |                    Huang Kaixuan et al. (Princeton & Google)                    |                    A perturbation set for the most difficult problems in the MATH dataset. It selects 279 of the hardest Level 5 problems from MATH and generates 279 variants for each through "simple perturbations" and "difficult perturbations." Model performance on these perturbed problems significantly declines, reflecting its real mathematical generalization ability.                   |                                      📄[paper](https://openreview.net/forum?id=IkmD3fKBPQ)                                      |

- **Complex Coding**


|      Name     | Number of Problems | Release Date |                                     Authors                                     |                                                                                                                                                                                               Description                                                                                                                                                                                              |                                                          Relevant Links                                                         |
|:-------:|:-----------:|:--------:|:------------------------:|:------------------------------------------------------------------------------------------:|:---------:|
| SWE-bench     | 2,294                       | 2024         | Chen Tianle et al. (Princeton NLP)                              | Software Engineering Bench, a dataset extracted from real software project issues-patch pairs on GitHub. It collects 2,294 issues and their corresponding Pull Request fixes from 12 popular Python libraries. The dataset is used to evaluate models' ability to automatically resolve real code bugs.                                                                                          | 🤗[dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench) <br> 🌐[repository](https://github.com/SWE-bench/SWE-bench)              |
| CodeContests  | \~10,000                    | 2022         | Li et al. (DeepMind)                                            | A competitive programming dataset proposed by DeepMind for training AlphaCode. It aggregates a vast number of problems and test cases from platforms such as Codeforces and AtCoder. The dataset contains around 10,000 multilingual programming problems, useful for code generation model training and evaluation.                                                                             | 🤗[dataset](https://huggingface.co/datasets/deepmind/code_contests)                                                                       |
| LiveCodeBench | \~400 (increasing annually) | 2024         | Jain et al. (UC Berkeley & MIT)                                 | A "live" benchmark for code. Continuously collects the latest publicly available problems from LeetCode, AtCoder, and Codeforces, totaling around 400 high-quality programming problems. In addition to code generation, it also evaluates models' abilities in code debugging, self-repair, and unit test generation.                                                                           | 🤗[dataset](https://huggingface.co/livecodebench) <br> 🌐[repository](https://github.com/LiveCodeBench/LiveCodeBench)                     |
| MHPP          | 210                         | 2025         | Dai Jianbo et al.                                               | Mostly Hard Python Problems, a human-designed collection of difficult Python programming tasks. The dataset contains 210 problems across seven challenge categories, each requiring multi-step reasoning or complex algorithms to solve. It is used to assess the limits of LLMs in code reasoning efficiency and accuracy.                                                                      | 📄[paper](https://openreview.net/forum?id=TVFVx8TUbN)                                                                                     |
| ProBench      | -                           | 2025         | Yang Lei et al. (Shanghai University of Science and Technology) | A benchmark designed specifically for competitive programming. It collects contest problems from Codeforces, Luogu, and Nowcoder platforms in the second half of 2024, with unified difficulty and algorithm tags. The dataset contains several hundred problems, filling the gap in advanced code reasoning evaluation.                                                                         | 🤗[dataset](https://huggingface.co/datasets/yl-9/probench) <br> 🌐[repository](https://github.com/YL-9/probench)                          |
| HumanEval Pro | 164                         | 2024         | Yu Zhaojian et al. (Microsoft AI Research)                      | An enhanced version of the OpenAI HumanEval dataset. For the original 164 programming problems, an additional "sub-question" is added, requiring the model to first solve a simpler sub-problem before using the result to solve the more complex problem. Compared to the original HumanEval, the Pro version reduces model accuracy by about 20%.                                              | 🤗[dataset](https://huggingface.co/datasets/CodeEval-Pro/humaneval-pro) <br> 🌐[repository](https://github.com/CodeEval-Pro/CodeEval-Pro) |
| MBPP Pro      | 378                         | 2024         | Yu Zhaojian et al. (Microsoft AI Research)                      | An advanced version of the Google MBPP programming problem dataset. It selects 378 problems from the MBPP test set and constructs additional questions similar to those in HumanEval Pro, making the problems more hierarchical and comprehensive. It is used for a more stringent evaluation of models' multi-step reasoning abilities in basic programming tasks.                              | 🤗[dataset](https://huggingface.co/datasets/CodeEval-Pro/mbpp-pro) <br> 🌐[repository](https://github.com/CodeEval-Pro/CodeEval-Pro)      |
| EquiBench     | 2,400                       | 2025         | Wei Anjiang et al. (Stanford & NYU)                             | A code semantic understanding benchmark. It evaluates LLMs' understanding of program execution semantics through equivalence verification tasks. The dataset provides 2,400 pairs of functionally equivalent/inequivalent programs in four programming languages. Models are required to determine if the outputs of two programs are identical, testing their understanding of deep code logic. | 🤗[dataset](https://huggingface.co/datasets/anjiangwei/EquiBench-Datasets) <br> 🌐[repository](https://github.com/Anjiang-Wei/EquiBench)  |


- **Commonsense Puzzles**

Here is the translation of the table into academic English:

|      Name     | Number of Problems | Release Date |                                     Authors                                     |                                                                                                                                                                                               Description                                                                                                                                                                                              |                                                          Relevant Links                                                         |
| :------------------: | :-----------------------------: | :----------: | :-----------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       LiveBench      |          Dynamic Update         |     2025     |   White et al. (NYU & Meta AI)  |                                                                             An online updating comprehensive evaluation framework for LLMs. New tasks are added monthly to ensure the test set is not contaminated by the model's training data. Tasks cover areas like mathematics, logic, programming, and common sense QA. It uses automated scoring and verifiable standard answers to ensure unbiased and objective evaluation.                                                                             | 🤗[dataset](https://huggingface.co/collections/livebench/livebench-67eaef9bb68b45b17a197a98) <br> 🌐[repository](https://github.com/livebench/livebench) <br> 🌐[website](https://livebench.ai/) |
| BIG-Bench Hard (BBH) | 23 Tasks (Over 2,000 Questions) |     2023     | Suzgun et al. (Google Research) |                                        A collection of 23 of the most challenging tasks selected from the BIG-Bench large-scale general benchmark. These tasks show much lower performance on models like GPT-3 compared to human average levels and cover areas like boolean expression evaluation, causal reasoning, date understanding, and complex common sense/logic problems. It is commonly used as a benchmark for chain-of-thought (CoT) enhancement experiments.                                       |                               🤗[dataset](https://huggingface.co/datasets/maveriq/bigbenchhard) <br> 🌐[repository](https://github.com/suzgunmirac/BIG-Bench-Hard)                               |
|      ZebraLogic      |                -                |     2024     |        Lin et al. (HKUST)       |                                                                                                        A logic reasoning dataset inspired by “zebra puzzles.” It contains a set of complex deductive reasoning problems, often involving non-monotonic reasoning scenarios, generated by models and manually verified. It is used to test the model’s consistency in reasoning under purely logical cues.                                                                                                        |  🤗[dataset](https://huggingface.co/datasets/WildEval/ZebraLogic) <br> 🌐[repository](https://github.com/WildEval/ZeroEval) <br> 🌐[website](https://huggingface.co/spaces/WildEval/ZebraLogic)  |
|          ARC         |              10,377             |     2018     |        Clark et al. (AI2)       |                                   AI2 Reasoning Challenge, a multiple-choice dataset for natural common sense and science questions. The questions are sourced from US K-12 science exams and are divided into easy and hard sections. It includes 7,787 training questions and 2,590 challenge questions. GPT-4 still struggles to surpass elimination-round performance on the ARC challenge set, making it a common benchmark for general common sense intelligence testing.                                  |                                                                   🤗[dataset](https://huggingface.co/datasets/allenai/ai2_arc)                                                                   |
|       JustLogic      |              4,900              |     2024     |    Michael Chen et al. (USYD)   |                            A pure deductive logic reasoning benchmark. It includes 4,900 propositional logic reasoning problems automatically generated by a synthetic algorithm, which do not rely on any common sense knowledge, focusing solely on testing the model’s ability to perform formal logical deductions. Each task provides a set of premises and a proposition conclusion, and the model must determine the truth value of the conclusion: true, false, or uncertain.                            |                                🤗[dataset](https://huggingface.co/datasets/WildEval/ZebraLogic) <br> 🌐[repository](https://github.com/michaelchen-lab/JustLogic)                                |
|      QuestBench      |              \~600              |     2025     |       Li et al. (DeepMind)      | Information retrieval reasoning evaluation released by DeepMind. It contains four types of "incomplete problems": logic, planning, mathematics (GSM), and formula problems, where each question is missing one key condition. The model must identify the most critical clarifying question to ask and use that information to answer the original question. It includes around 600 such common sense/reasoning problems, designed to evaluate the ability of LLMs to identify and ask for critical information. |                                                                   🌐[repository](https://github.com/google-deepmind/questbench)                                                                  |


- **Scientific Reasoning**

| Name | Number of Problems | Release Date | Authors | Description | Relevant Links |
|:-------:|:-----------:|:--------:|:------------------------:|:------------------------------------------------------------------------------------------:|:---------:|
| GPQA Diamond               | 198                 | 2024         | Rein et al. (NYU)                   | A highly difficult subset of Graduate-level Physics/Biology/Chemistry Q\&A. The GPQA dataset filters out 198 questions that are answered correctly by experts but incorrectly by laypersons. These "diamond-level" problems are almost at the graduate level and require models to possess cross-disciplinary deep reasoning abilities.                                                                      | 🤗[dataset](https://huggingface.co/datasets/Idavidrein/gpqa) <br> 🌐[repository](https://github.com/idavidrein/gpqa)                                       |
| MMLU-Pro                   | \~12,000            | 2024         | Wang Yubo et al.                    | An enhanced version of the original MMLU benchmark. It includes 12,000 high-quality academic exam questions from 14 major fields (with the number of answer options expanded from 4 to 10), focusing on comprehensive knowledge and complex reasoning. Compared to the original MMLU, the Pro version significantly increases the difficulty, with the model's accuracy dropping by an average of about 20%. | 🤗[dataset](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) <br> 🌐[repository](https://github.com/TIGER-AI-Lab/MMLU-Pro)                              |
| SuperGPQA                  | 26,529              | 2025         | Doubao (Seed) Team                  | A large-scale graduate-level knowledge reasoning benchmark. Covering 285 academic disciplines, it contains 26,529 high-difficulty professional exam questions. Over 42% of the questions require mathematical calculations or formal reasoning, aiming to test the model's reasoning limits in long-tail disciplines.                                                                                        | 🤗[dataset](https://huggingface.co/datasets/m-a-p/SuperGPQA) <br> 🌐[repository](https://github.com/SuperGPQA/SuperGPQA)                                   |
| Humanity’s Last Exam (HLE) | 2,500               | 2025         | CAIS & Scale AI                     | "Humanity's Last Exam," designed as the final closed-book test of human knowledge. It includes 2,500 multiple-choice or short-answer questions across dozens of fields such as mathematics, natural sciences, and humanities. Created collaboratively by global experts, it exceeds the difficulty of all previous benchmarks and is considered the most difficult comprehensive exam AI currently faces.    | 🤗[dataset](https://huggingface.co/datasets/cais/hle) <br> 🌐[repository](https://github.com/centerforaisafety/hle) <br> 🌐[website](https://lastexam.ai/) |
| TPBench                    | -                   | 2024         | Daniel J.H. Chung et al. (DeepMind) | A Theoretical Physics Benchmark designed to assess models' ability to solve advanced theoretical physics problems. Proposed by Chung et al., this benchmark collects a set of theoretical physics problems requiring advanced knowledge and complex derivations, testing the model's limits in reasoning about physical laws and equation derivations.                                                       | 🤗[dataset](https://huggingface.co/datasets/ZhiqiGao/TPBench) <br> 🌐[website](https://tpbench.org/)                                                       |


- **Medical Reasoning**

| Name | Number of Problems | Release Date | Authors | Description | Relevant Links |
|:-------:|:-----------:|:--------:|:------------------------:|:------------------------------------------------------------------------------------------:|:---------:|
|          MedQA          |       12,723       |     2020     |    Jin et al. (Tsinghua University)    | A medical exam question-answer dataset. Collected from the United States Medical Licensing Examination (USMLE) multiple-choice questions, covering subjects such as anatomy, physiology, pathology, etc. Includes English (12,723 questions) and simplified/traditional Chinese versions (approximately 50,000 questions in total). Used to evaluate models' ability to apply medical knowledge and diagnostic reasoning. |                    🌐[Google Drive](https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view) <br> 🌐[Repository](https://github.com/jind11/MedQA)                   |
| JAMA Clinical Challenge |        1,524       |     2024     | Chen et al. (Johns Hopkins University) |             The Clinical Challenge Case Set from the Journal of the American Medical Association (JAMA). Compiles 1,524 challenging clinical cases published by the journal, each with detailed case descriptions, questions, four options, and professional explanations. Focuses on assessing the model’s diagnostic decision-making ability and interpretability in real-world, complex clinical scenarios.            |                                                      🌐[Website](https://jamanetwork.com/collections/44038/clinical-challenge)                                                     |
|        Medbullets       |         308        |     2024     | Chen et al. (Johns Hopkins University) |    A simulated clinical Q\&A dataset. Composed of 308 multiple-choice questions in the USMLE Step 2/3 style, collected from the Twitter medical Q\&A account. Each question includes a case scenario, five options, and detailed explanations. While based on common clinical scenarios, the questions remain challenging and are used to evaluate model performance in clinical decision-making and interpretability.    |                                                           🌐[Website](https://github.com/HanjieChen/ChallengeClinicalQA)                                                           |
|        MedXpertQA       |        4,460       |     2024     |            Tsinghua C3I Team           |                                    A comprehensive benchmark for “expert-level” medical reasoning. Consists of 4,460 high-difficulty clinical knowledge Q\&A covering 17 specialties and 11 body systems. Available in both pure-text (case + Q\&A) and multimodal (including medical images) formats, used to evaluate models’ joint reasoning ability over medical texts and images.                                    | 🤗[Dataset](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) <br> 🌐[Repository](https://github.com/TsinghuaC3I/MedXpertQA) <br> 🌐[Website](https://medxpertqa.github.io/) |

#### 5.1.3 Capability Benchmarks
The focus is on the local perspective or the individual abilities of the model during the Long CoT reasoning process, examining finer granularity by investigating whether each step of the model's reasoning is correct and logical. For instance, whether it can correctly identify errors and correct them, or whether it can complete complex tasks step by step.

- **Deep Reasoning**

| Name | Number of Problems | Release Date | Authors | Description | Relevant Links |
|:-------:|:-----------:|:--------:|:------------------------:|:------------------------------------------------------------------------------------------:|:---------:|
| ZebraLogic |       \~1,000      |     2024     |     Bill Yuchen Lin et al.     |                               ZebraLogic is an AI benchmark focusing on logical reasoning, containing complex mathematical and linguistic reasoning problems used to assess advanced reasoning abilities of models. Its problem design is similar to the "Zebra Puzzle," challenging models to perform logical reasoning and problem-solving under constraints.                              | 🤗[dataset](https://huggingface.co/spaces/allenai/ZebraLogic) <br> 🌐[repository](https://github.com/WildEval/ZeroEval) <br> 🌐[website](https://huggingface.co/blog/yuchenlin/zebra-logic) |
|   BigGSM   |         610        |     2025     | Qiguang Chen et al. (HIT-SCIR) |                                  A mathematical reasoning benchmark designed to evaluate the performance of large language models on multi-step mathematical problems. It extends the classic GSM8K dataset and includes more challenging mathematical application problems that require models to perform more complex logical reasoning and computations.                                  |                          🤗[dataset](https://huggingface.co/datasets/LightChen2333/BigGSM) <br> 🌐[repository](https://github.com/LightChen233/reasoning-boundary)                          |
| GSM-Ranges |        30.1k       |     2025     |   Safal Shrestha et al. (NYU)  | GSM-Ranges is a dataset generator built upon the GSM8K benchmark. It systematically modifies numerical values in mathematical word problems to assess the robustness of large language models across a wide range of numerical scales. By introducing numerical perturbations, GSM-Ranges evaluates the ability of LLMs to reason mathematically with numbers beyond the distribution range. |                              🤗[dataset](https://huggingface.co/datasets/guactastesgood/GSM-Ranges) <br> 🌐[repository](https://github.com/minwukim/GSM-Ranges)                             |


- **Exploration Benchmarks**

| Name | Number of Problems | Release Date | Authors | Description | Relevant Links |
|:-------:|:-----------:|:--------:|:------------------------:|:------------------------------------------------------------------------------------------:|:---------:|
|  Sys2Bench  |          -         |     2025     |         Shubham Parashar et al.        | Sys2Bench is designed to systematically test large language models across various reasoning and planning tasks. The benchmark covers five major types of reasoning: algorithmic reasoning, planning, arithmetic reasoning, logical reasoning, and common-sense reasoning, consisting of 11 sub-tasks ranging from NP-hard problems (such as Rubik's Cube and Bin Packing) to multi-step math problems (such as GSM8K). Sys2Bench places special emphasis on intermediate steps in the reasoning process, highlighting the quality and efficiency of the reasoning path. Additionally, the project introduces AutoHD (Automated Heuristics Discovery) methods, allowing models to autonomously generate heuristic functions during the reasoning process to improve complex task planning capabilities. | 🤗[dataset](https://huggingface.co/datasets/dive-lab/Sys2Bench) <br> 🌐[repository](https://github.com/divelab/sys2bench) |
| BanditBench |          -         |     2025     | Allen Nie et al. (Stanford University) |            BanditBench is designed to evaluate the exploration and decision-making abilities of large language models in multi-armed bandit (MAB) and contextual bandit (CB) environments. The benchmark simulates LLMs as agents, relying solely on contextual information for multi-round interactions without updating parameters, to measure their performance in uncertain environments. BanditBench provides various task scenarios, including movie recommendation tasks based on the MovieLens dataset, covering different action numbers and reward distribution types (e.g., Gaussian and Bernoulli distributions). Additionally, researchers have introduced algorithm-guided reasoning support and algorithm distillation methods to enhance the exploration efficiency of LLMs.           |                           🌐[repository](https://github.com/allenanie/EVOLvE?tab=readme-ov-file)                          |

- **Reflection Benchmarks**

| Name | Number of Problems | Release Date | Authors | Description | Relevant Links |
|:-------:|:-----------:|:--------:|:------------------------:|:------------------------------------------------------------------------------------------:|:---------:|
|  RewardBench |          2,958         |       2024       |                Nathan Lambert et al. (AI2)               |                                                                                                                                RewardBench is the first systematic reward model evaluation benchmark, jointly released by AI2 and the University of Washington, designed to analyze and compare the performance of reward models under different training methods across alignment quality, reasoning ability, safety, and instruction following, providing a unified evaluation framework.                                                                                                                               |       🤗[dataset](https://huggingface.co/datasets/allenai/reward-bench) <br> 🌐[repository](https://github.com/allenai/reward-bench) <br> 🌐[website](https://huggingface.co/spaces/allenai/reward-bench)       |
| ProcessBench |          3,400         |       2024       |              Zheng Chujie et al. (Qwen Team)             | ProcessBench is a mathematical reasoning process evaluation benchmark proposed by Alibaba’s Qwen Team, consisting of 3,400 Olympiad-level problems with step-by-step solutions, where each step is manually annotated for errors. The benchmark requires models to identify the earliest error step in the reasoning process, focusing on process supervision rather than solely on the final answer. Evaluation results show that general language models (e.g., QwQ-32B-Preview) outperform specially trained process reward models (PRMs) in step-by-step critique tasks, approaching the performance level of GPT-4o. |                                            🤗[dataset](https://huggingface.co/datasets/Qwen/ProcessBench) <br> 🌐[repository](https://github.com/QwenLM/ProcessBench)                                           |
|   PRMBench   |          6,216         |       2025       | Mingyang Song et al. (Fudan University, Shanghai AI Lab) |                                                                PRMBench aims to fill the gap in existing benchmarks that primarily focus on step correctness and lack systematic evaluation of PRMs, offering a unified framework for evaluation across multiple dimensions including conciseness, robustness, and sensitivity. Each sample in the benchmark includes a question, a reasoning process with errors, annotations of erroneous steps, and the causes of the errors, aiming to evaluate the fine-grained error detection capabilities of PRMs.                                                                |                   🤗[dataset](https://huggingface.co/datasets/hitsmy/PRMBench_Preview) <br> 🌐[repository](https://github.com/ssmisya/PRMBench) <br> 🌐[website](https://prmbench.github.io/)                   |
|  CriticBench |         \~3,800        |       2024       |           Lan Tian et al. (Tsinghua University)          |                        CriticBench, proposed by Tsinghua University and other institutions, is a comprehensive benchmark for evaluating the critique and correction abilities of large language models. It covers five major reasoning areas: mathematics, commonsense, symbolism, programming, and algorithms, integrating 15 datasets to assess 17 LLMs in the stages of generation, critique, and correction. The study finds that models trained specifically for critique perform better in the Generate-Critique-Correct (GQC) task, and that larger models show higher critique consistency.                       |               🤗[dataset](https://huggingface.co/datasets/llm-agents/CriticBench) <br> 🌐[repository](https://github.com/CriticBench/CriticBench) <br> 🌐[website](https://criticbench.github.io/)              |
|  DeltaBench  |          1,236         |       2025       |                      OpenStellarTeam                     |                                                                              DeltaBench, released by the OpenStellar Team, is a benchmark designed to assess large language models' error detection capabilities in Long CoT (Chain of Thought) reasoning tasks. It includes 1,236 samples across areas such as mathematics, programming, physical-chemical-biological (PCB) reasoning, and general reasoning. Each sample is annotated with detailed manual labels identifying erroneous steps, strategy shifts, and reflection efficiency.                                                                              |    🤗[dataset](https://huggingface.co/datasets/OpenStellarTeam/DeltaBench) <br> 🌐[repository](https://github.com/OpenStellarTeam/DeltaBench) <br> 🌐[website](https://openstellarteam.github.io/DeltaBench/)   |
|  ErrorRadar  |          2,500         |       2024       |               Yan Yibo et al. (Squirrel AI)              |                                                                 ErrorRadar is a multimodal mathematical reasoning error detection benchmark designed to evaluate multimodal large language models' ability to identify and classify errors in student problem-solving processes. The benchmark contains 2,500 K-12 mathematics problems from real educational scenarios, incorporating both textual and image information, and annotating erroneous steps and error types. Evaluation tasks include error step localization and error type classification.                                                                | 🤗[dataset](https://huggingface.co/datasets/ErrorRadar/ErrorRadar) <br> 🌐[repository](https://anonymous.4open.science/r/Error-Radar/readme.md) <br> 🌐[website](https://anonymous.4open.science/r/Error-Radar) |
|     MEDEC    |          3,848         |       2024       |            Ben Abacha Asma et al. (Microsoft)            |                                                                                                                              MEDEC is the first public benchmark for medical error detection and correction, jointly released by Microsoft and the University of Washington. It contains 3,848 clinical texts, covering five types of errors, including diagnosis, treatment, and medication, providing a crucial tool for improving the accuracy and safety of medical document generation.                                                                                                                              |                                                                                 🌐[repository](https://github.com/abachaa/MEDEC)                                                                                |

#### 5.1.4 Advanced Benchmarks
Benchmarks designed specifically to evaluate large language models' capabilities in complex reasoning, cross-domain knowledge integration, and multimodal understanding. As basic evaluations are gradually saturated by top-tier models, researchers have started developing more challenging benchmarks to more accurately measure models' performance on real-world complex tasks.



- **Agentic & Embodied Reasoning**

| Name | Number of Problems | Release Date | Authors | Description | Relevant Links |
|:-------:|:-----------:|:--------:|:------------------------:|:------------------------------------------------------------------------------------------:|:---------:|
|  ToolComp  |                  485                 |       2025       |          Vaskar Nath et al. (Scale AI)          |                                               ToolComp is designed to assess large language models' reasoning and process supervision capabilities in complex multi-step tool usage tasks. The benchmark consists of 485 manually edited and verified prompts, involving the use of 11 different tools, and 1,731 step-by-step supervision labels, offering a comprehensive assessment of models' performance in multi-tool reasoning tasks.                                               |   🌐[website](https://scale.com/research/toolcomp-a-multi-tool-reasoning-and-process-supervision-benchmark)   |
|   OSWorld  |                  369                 |       2025       |   Xie Tianbao et al. (University of Hong Kong)  |                                        OSWorld is a multimodal agent evaluation benchmark jointly released by the University of Hong Kong, Salesforce Research, and other institutions, aiming to test AI's ability to complete open-ended tasks in real computer environments. The benchmark consists of 369 tasks across file operations, web browsing, office software usage, and other scenarios, supporting Ubuntu, Windows, and macOS systems.                                       |       🌐[repository](https://github.com/xlang-ai/OSWorld) <br> 🌐[website](https://os-world.github.io/)       |
|   WebShop  | 12,087 Instructions / 1.18M Products |       2022       |     Yao Shunyu et al. (Princeton University)    | WebShop simulates an e-commerce website environment and is designed to evaluate large language models' abilities in real web interactions. The benchmark includes 1.18 million real products and 12,087 user instructions, requiring agents to browse webpages, search, filter, and complete purchase tasks based on natural language instructions. WebShop focuses on evaluating models' performance in understanding complex instructions, handling web noise, and exploring strategies. |   🌐[repository](https://github.com/princeton-nlp/WebShop) <br> 🌐[website](https://webshop-pnlp.github.io/)  |
|  WebArena  |                  812                 |       2024       | Zhou Shuyan et al. (Carnegie Mellon University) |                                              WebArena is a high-fidelity web environment released by Carnegie Mellon University, designed to evaluate large language models' agent capabilities in real web tasks. The benchmark consists of 812 tasks covering e-commerce, social forums, content management, and collaborative development, requiring models to complete multi-step web interactions through natural language instructions.                                              |        🌐[repository](https://github.com/web-arena-x/webarena) <br> 🌐[website](https://webarena.dev/)        |
|  WebGames  |                  50+                 |       2025       |      Thomas George et al. (Convergence AI)      |                                                                                         WebGames is a web browsing agent benchmark, covering basic browsing operations, complex input handling, cognitive tasks, and workflow automation. WebGames provides a lightweight, verifiable test environment supporting rapid iteration and evaluation, suitable for developing more powerful web agents.                                                                                        | 🌐[repository](https://github.com/convergence-ai/webgames) <br> 🌐[website](https://webgames.convergence.ai/) |
| Text2World |                  103                 |       2025       |   Mengkang Hu et al. (University of Hong Kong)  |                                  Text2World is a benchmark proposed by the University of Hong Kong and other institutions, aiming to evaluate large language models' ability to generate symbolic world models from natural language. The benchmark is based on the Planning Domain Definition Language (PDDL) and covers hundreds of diverse domains, employing a multi-criteria, execution-based evaluation method to provide a more robust assessment.                                  |   🌐[repository](https://github.com/Aaron617/text2world) <br> 🌐[website](https://text-to-world.github.io/)   |

- **Multimodal Reasoning**
    - **Complex Mathematics:**

    | Name | Number of Problems | Release Date | Authors | Description | Relevant Links |
    |:-------:|:-----------:|:--------:|:------------------------:|:------------------------------------------------------------------------------------------:|:---------:|
    |  MathVista |        6,141       |     2023     |               Pan Lu et al. (UCLA)               |                                                                                                                                                                              MathVista is a multimodal mathematical reasoning evaluation benchmark jointly released by UCLA, the University of Washington, and Microsoft Research. It is designed to systematically assess the mathematical reasoning capabilities of large language models and multimodal models within a visual context.                                                                                                                                                                              |    🤗[dataset](https://huggingface.co/datasets/AI4Math/MathVista) <br> 🌐[repository](https://github.com/lupantech/MathVista) <br> 🌐[website](https://mathvista.github.io/)    |
    | MathVision |        3,040       |     2024     | Ke Wang et al. (Chinese University of Hong Kong) |                                                                                                         MathVision (MATH-V) is a multimodal mathematical reasoning evaluation benchmark released by the Chinese University of Hong Kong, among others. It aims to systematically evaluate the mathematical reasoning abilities of large vision-language models within visual contexts. The benchmark includes 3,040 problems across 16 mathematical disciplines, divided into five difficulty levels, with problems sourced from real mathematics competitions.                                                                                                         | 🤗[dataset](https://huggingface.co/datasets/MathLLMs/MathVision) <br> 🌐[repository](https://github.com/mathllm/MATH-V) <br> 🌐[website](https://mathllm.github.io/mathvision/) |
    |  MathVerse |      \~15,000      |     2024     | Zimu Lu et al. (Chinese University of Hong Kong) | MathVerse is a multimodal mathematical reasoning evaluation benchmark jointly released by MMLab at the Chinese University of Hong Kong and the Shanghai AI Lab. It is designed to comprehensively assess multimodal large language models' ability to understand mathematical diagrams. The benchmark includes 2,612 problems spanning areas such as plane geometry, solid geometry, and functions, annotated by experts. It generates six versions of multimodal information, totaling approximately 15,000 test samples. MathVerse introduces a Chain-of-Thought (CoT) evaluation strategy, leveraging GPT-4V for fine-grained analysis of model reasoning processes. |                          🤗[dataset](https://huggingface.co/datasets/luzimu/WebGen-Bench) <br> 🌐[repository](https://github.com/mnluzimu/WebGen-Bench)                         |

    - **Complex Code:**

    | Name | Number of Problems | Release Date | Authors | Description | Relevant Links |
    |:-------:|:-----------:|:--------:|:------------------------:|:------------------------------------------------------------------------------------------:|:---------:|
    | HumanEval-V |         253        |     2024     | Fengji Zhang et al. (City University of Hong Kong) |                                                                        HumanEval-V is a multimodal code generation evaluation benchmark proposed by the University of Hong Kong, aiming to test the capabilities of large multimodal models in complex diagram understanding and code generation tasks. This benchmark includes 253 Python programming tasks, each accompanied by key diagrams and function signatures, requiring the model to generate executable code based on visual information.                                                                        | 🤗[dataset](https://huggingface.co/datasets/HumanEval-V/HumanEval-V-Benchmark) <br> 🌐[repository](https://github.com/HumanEval-V/HumanEval-V-Benchmark) <br> 🌐[website](https://humaneval-v.github.io/) |
    | Code-Vision |       1,000+       |     2025     |       Hanbin Wang et al. (Peking University)       |                                  Code-Vision is a multimodal code generation evaluation benchmark jointly released by Peking University, Northeastern University, and the University of Hong Kong. It aims to test the ability of multimodal large language models to understand flowcharts and generate corresponding code. This benchmark fills the gap in existing benchmarks, which mainly focus on textual reasoning and lack a systematic evaluation of code generation in visual contexts, providing a unified evaluation framework.                                 |                                     🌐[repository](https://github.com/wanghanbinpanda/CodeVision) <br> 🌐[website](https://pingshengren0901.github.io/codevision.io/)                                     |
    |  ChartMimic |        4,800       |     2024     |       Cheng Yang et al. (Tsinghua University)      | ChartMimic is a multimodal code generation evaluation benchmark jointly released by Tsinghua University, Tencent AI Lab, and other institutions. It aims to evaluate the cross-modal reasoning abilities of large multimodal models in chart understanding and code generation, addressing the gap in existing benchmarks that focus mainly on textual reasoning and lack systematic evaluation of chart understanding and code generation. It includes two task types: Direct Mimic and Customized Mimic, with data sourced from scientific papers across multiple fields. |              🤗[dataset](https://huggingface.co/datasets/ChartMimic/ChartMimic) <br> 🌐[repository](https://github.com/ChartMimic/ChartMimic) <br> 🌐[website](https://chartmimic.github.io/)             |

    - **Complex Science:**

    | Name | Number of Problems | Release Date | Authors | Description | Relevant Links |
    |:-------:|:-----------:|:--------:|:------------------------:|:------------------------------------------------------------------------------------------:|:---------:|
    | ScienceQA |       21,208       |     2022     |        Pan Lu et al. (UCLA)        |                                                                        ScienceQA is a multimodal multiple-choice dataset consisting of 21,208 problems across natural sciences, language sciences, and social sciences, designed for K-12 grade levels. The dataset provides context with images and text, explanations, and detailed answers, supporting Chain-of-Thought (CoT) reasoning, aiming to assess and enhance the multi-step reasoning abilities and interpretability of AI models.                                                                       |         🤗[dataset](https://huggingface.co/datasets/TheMrguiller/ScienceQA) <br> 🌐[repository](https://github.com/lupantech/ScienceQA) <br> 🌐[website](https://scienceqa.github.io/)        |
    |   M3CoT   |       11,459       |     2024     | Qiguang Chen et al. (HIT-SCIR Lab) | M3CoT is a multimodal, multi-domain, multi-step reasoning dataset built upon ScienceQA, designed to assess the capabilities of AI models in complex reasoning tasks. Compared to ScienceQA, M3CoT-Science has an average reasoning step increase from 2.5 to 10.9, and the average text length grows from 48 to 294, significantly increasing task complexity. The dataset spans science, common sense, and mathematics, emphasizing cross-reasoning between image and text information, challenging the reasoning capabilities of existing multimodal large models. | 🤗[dataset](https://huggingface.co/datasets/LightChen2333/M3CoT) <br> 🌐[repository](https://github.com/LightChen233/M3CoT) <br> 🌐[website](https://lightchen233.github.io/m3cot.github.io/) |
    | MolPuzzle |         234        |     2024     |          Kehan Guo et al.          |                       MolPuzzle is a multimodal, multi-step reasoning dataset designed to evaluate large language models in molecular structure analysis tasks. The dataset involves various spectrometric data types, including infrared spectroscopy (IR), mass spectrometry (MS), and nuclear magnetic resonance (1H-NMR and 13C-NMR), as well as molecular formula information. Tasks are divided into three stages: molecular understanding, spectral analysis, and molecular construction, simulating real chemical reasoning processes.                       |   🤗[dataset](https://huggingface.co/datasets/kguo2/MolPuzzle_data) <br> 🌐[repository](https://github.com/KehanGuo2/MolPuzzle) <br> 🌐[website](https://kehanguo2.github.io/Molpuzzle.io/)   |

    - **Commonsense Puzzle:**

    | Name | Number of Problems | Release Date | Authors | Description | Relevant Links |
    |:-------:|:-----------:|:--------:|:------------------------:|:------------------------------------------------------------------------------------------:|:---------:|
    |   PuzzleVQA  |        2,000       |     2024     |          Yew Ken Chia et al.         | PuzzleVQA is a multimodal reasoning dataset consisting of 2,000 abstract graphic puzzles, designed to evaluate the visual perception, induction, and deduction abilities of large multimodal models in basic concepts such as color, numbers, shapes, and sizes. Experiments show that even advanced models like GPT-4V achieve an average accuracy of only 46.4% on single-concept puzzles, significantly lower than human performance, exposing limitations in abstract pattern recognition and multi-step reasoning. | 🤗[dataset](https://huggingface.co/datasets/declare-lab/PuzzleVQA) <br> 🌐[repository](https://github.com/declare-lab/LLM-PuzzleTest/tree/master/PuzzleVQA) <br> 🌐[website](https://puzzlevqa.github.io/) |
    | LEGO-Puzzles |        1,100       |     2025     | Kexian Tang et al. (Shanghai AI Lab) |                                                                                                  LEGO-Puzzles aims to evaluate the capability of large multimodal language models in multi-step spatial reasoning tasks. The dataset contains 1,100 visual question answering (VQA) tasks based on LEGO bricks, covering 11 task types, including spatial understanding, single-step and multi-step sequence reasoning.                                                                                                 |      🤗[dataset](https://huggingface.co/datasets/KexianTang/LEGO-Puzzles) <br> 🌐[repository](https://github.com/Tangkexian/LEGO-Puzzles) <br> 🌐[website](https://tangkexian.github.io/LEGO-Puzzles/)     |
    |     CVQA     |       10,374       |     2024     |     David Romero et al. (MBZUAI)     |                                                                                                         CVQA is a multimodal visual question answering dataset designed to assess models' abilities to integrate multiple visual cues for combined reasoning. The dataset includes three task types requiring models to extract and synthesize key information from multiple images to answer complex questions.                                                                                                        |                                                    🤗[dataset](https://huggingface.co/datasets/afaji/cvqa) <br> 🌐[website](https://cvqa-benchmark.org/)                                                   |

- **AI4Research:**

| Name | Number of Problems | Release Date | Authors | Description | Relevant Links |
|:-------:|:-----------:|:--------:|:------------------------:|:------------------------------------------------------------------------------------------:|:---------:|
|    SciWorld   | 30 tasks / 6,000+ instances |     2022     |                           Ruoyao Wang et al.                           |                                                                                                                                                                                                      SciWorld aims to evaluate the understanding and reasoning abilities of large multimodal models in complex scientific scenarios. The dataset integrates images, text, and structured data, covering multiple scientific domains and designed with multi-step reasoning tasks, challenging models' abilities to integrate multi-source information, perform causal reasoning, and provide interpretable answers. It consists of 30 tasks, each with multiple variants, totaling over 6,000 instances. The introduction of SciWorld has propelled the application of multimodal models in scientific education and research.                                                                                                                                                                                                     |           🌐[repository](https://github.com/allenai/ScienceWorld) <br> 🌐[website](https://sciworld.apps.allenai.org/)           |
|     HardML    |             100             |     2025     |                           Tidor-Vlad Pricope                           | HardML is a benchmark dataset designed specifically to evaluate AI's knowledge and reasoning abilities in the fields of data science and machine learning. Created by independent machine learning engineer Tidor-Vlad Pricope, it contains 100 carefully crafted multiple-choice questions covering topics such as natural language processing, computer vision, statistical modeling, and classical machine learning algorithms. These questions are so challenging that even seasoned machine learning engineers struggle to answer them all correctly. To avoid data contamination, most of the questions are original, reflecting recent advancements in machine learning over the past two years. Current state-of-the-art AI models have an error rate of about 30% on HardML, which is three times higher than on MMLU-ML, demonstrating HardML's effectiveness in distinguishing model capabilities. Additionally, the author has released the slightly easier EasyML dataset, designed for models with fewer parameters. |                                            📄[paper](https://arxiv.org/pdf/2501.15627)                                           |
|   MLE-BENCH   |              75             |     2024     |                                 OpenAI                                 |                                                                                                                                                            MLE-bench is a benchmark dataset released by OpenAI, designed to evaluate AI agents' practical capabilities in machine learning engineering (MLE) tasks. The benchmark selects 75 diverse competition tasks from Kaggle, covering fields such as natural language processing, computer vision, signal processing, and more, testing models' engineering skills in data preprocessing, model training, and experimental execution. In the evaluation, OpenAI's o1-preview model, combined with the AIDE framework, achieved Kaggle bronze-level performance on 16.9% of tasks. The research also explores the impact of resource scaling on performance and issues related to pre-training data contamination.                                                                                                                                                           |            🌐[repository](https://github.com/openai/mle-bench/) <br> 🌐[website](https://openai.com/index/mle-bench/)            |
| SolutionBench |            1,053            |     2025     | Zhuoqun Li et al. (Institute of Software, Chinese Academy of Sciences) |                                                                                                                                                                                                                                                           SolutionBench is a benchmark dataset designed to evaluate the capabilities of AI systems in complex engineering solution design. It aims to fill the gap in current retrieval-augmented generation (RAG) methods in handling multi-constraint engineering problems, characterized by real data sources and structured data. Additionally, the authors introduced a system named SolutionRAG, which, by combining tree search and dual-point thinking mechanisms, achieved leading performance on SolutionBench.                                                                                                                                                                                                                                                          | 🤗[dataset](https://huggingface.co/datasets/lzq2021/SolutionBench) <br> 🌐[repository](https://github.com/icip-cas/DeepSolution) |

### 5.2 Training Datasets

To build and enhance models with strong Long CoT capabilities, numerous open-source training datasets have emerged. These datasets provide foundational supervision signals for various domains such as mathematics, science, medicine, programming, and general reasoning. Based on their construction methods, we classify the datasets into four major categories: Manual Annotation, Direct Distillation, Search-based Distillation, and Validated Distillation.

In this section, we systematically list representative datasets under each category, covering key information such as their sources, modalities, applicable domains, and data scale, providing researchers and developers seeking suitable training resources with a comprehensive guide and convenient reference.

#### 5.2.1 Manual Annotation

These datasets are created through manual annotation or rule-based construction, typically offering high-quality samples with interpretable reasoning paths. While smaller in scale, they are critical for guiding the alignment and evaluation of initial models.

- [R1-OneVision](https://huggingface.co/datasets/Fancy-MLLM/R1-Onevision) combines high-quality data from LLaVA-OneVision with datasets from specific domains. It bridges the gap between visual and textual understanding, providing rich, context-aware reasoning tasks across natural scenes, science, mathematics, OCR-based content, and complex graphs.
- [M3CoT](https://github.com/LightChen233/M3CoT) lays the foundational work for multi-domain, multi-step, multi-modal chain-of-thought research.
- [Big-Math-RL-Verified](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified) is designed for RL training with large language models (LLMs) such as [PPO](https://arxiv.org/abs/1707.06347), [GRPO](https://arxiv.org/abs/2402.03300), etc.
- [GSM8K](https://github.com/openai/grade-school-math) is a high-quality, linguistically diverse dataset of elementary school math word problems.

| Name                 | Category             | Source | Modality      | Quantity |
| :------------------- | :------------------- | :----- | :------------ | :------- |
| R1-OneVision         | Mathematics, Science | Rule   | Vision + Lang | 119K     |
| M3CoT                | Mathematics, Science | Human  | Vision + Lang | 11K      |
| Big-Math-RL-Verified | Mathematics          | Human  | Lang          | 251K     |
| GSM8K                | Mathematics          | Human  | Lang          | 8K       |

#### 5.2.2 Direct Distillation

The method utilizes large language models to generate training data through prompt-based or chain-of-thought reasoning. These datasets can be scaled up to millions of examples, covering a wide range of domains.

- [NaturalReasoning](https://arxiv.org/abs/2502.13124) validates through knowledge distillation experiments that NaturalReasoning can effectively extract and transfer reasoning capabilities from powerful teacher models. It is equally effective for unsupervised self-training using external reward models or self-rewarding.
- [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) employs a chain-of-thought (CoT) format for solving each problem. The dataset covers Chinese high school math exercises, US and international Mathematical Olympiad problems, etc., collected from online exam PDFs and math forums. Processing steps include: (a) OCR recognition of the original PDFs; (b) segmentation into problem-solution pairs; (c) translation into English; (d) rearranging to generate chain-of-thought reasoning formats; and (e) final answer formatting.
- [NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) focuses on problems that produce numerical outputs selected from the NuminaMath-CoT dataset. A pipeline was constructed using GPT-4 to generate reasoning paths similar to TORA, execute code, and produce results until the final solution is completed, filtering out solutions where the final answer differs from the reference answer.
- [DART-Math-uniform](https://huggingface.co/datasets/hkust-nlp/dart-math-uniform) constructs datasets through the application of DARS-Uniform.
- [DART-Math-hard](https://huggingface.co/datasets/hkust-nlp/dart-math-hard) is a mathematical question-answer pair sample constructed using query sets from the DARS-Prop2DiffMATH and GSK8K training datasets. It achieves SOTA results on many challenging mathematical reasoning benchmarks, introducing a deliberate preference for hard queries, in contrast to traditional rejection sampling.
- [DART-Math-pool-math](https://huggingface.co/datasets/hkust-nlp/dart-math-pool-math) is a data pool synthesized from query sets of the MATH training dataset, including all samples with correct answers and additional metadata generated during the process. DART-Math-\- datasets are extracted from the DART-Math-pool-\- data pools.
- [DART-Math-pool-gsm8k](https://huggingface.co/datasets/hkust-nlp/dart-math-pool-gsm8k) is a data pool synthesized from query sets of the GSM8K training dataset, including all samples with correct answers and additional metadata. DART-Math-\- datasets are extracted from the DART-Math-pool-\- data pools.
- [OpenO1-SFT](https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT) is a dataset for fine-tuning language models on chain-of-thought activations using SFT.
- [OpenO1-SFT-Pro](https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT-Pro) is a dataset for fine-tuning language models on chain-of-thought activations using SFT.
- [OpenO1-SFT-Ultra](https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT-Ultra) is synthesized based on existing open-source datasets using the openo1-qwen-sft model.
- [Medical-o1](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) is an SFT medical reasoning dataset constructed based on medically verifiable questions and an LLM verifier.
- [AoPS-Instruc](https://huggingface.co/datasets/DeepStudentLlama/AoPS-Instruct) is a large-scale high-quality question-answer pair dataset for advanced mathematical reasoning created and maintained using a scalable approach.
- [Orca-Math](https://arxiv.org/abs/2402.14830) is a high-quality synthetic dataset of 200,000 math problems created in a multi-agent setup where agents collaborate to generate the data.
- [MATH-plus](https://huggingface.co/datasets/TIGER-Lab/WebInstructSub) collects 10 million naturally occurring instruction data from pre-trained network corpora.
- [UltraInteract-SFT](https://huggingface.co/collections/openbmb/eurus-660bc40bec5376b3adc9d1c5) is designed for complex reasoning tasks and helps explore preference learning in reasoning tasks. It is applicable for supervised fine-tuning and preference learning. Each instruction includes a preference tree consisting of: (1) reasoning chains with multiple planning strategies and consistent formats; (2) multi-round interaction trajectories with the environment and comments; and (3) paired data for facilitating preference learning.
- [MathCodeInstruct](https://huggingface.co/datasets/MathLLMs/MathCodeInstruct) is a novel high-quality dataset containing mathematical problems and their code-based solutions.
- MathCodeInstruct-Plus [$Paper¹](https://arxiv.org/abs/2310.03731), [²$](https://arxiv.org/abs/2308.07921) is a novel high-quality dataset containing mathematical problems and their code-based solutions.
- OpenMathInstruct-1[\[HuggingFace\]](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) is a math instruction adjustment dataset, generating 1.8 million problem-solution pairs using the Mixtral-8x7B model. The problems are sourced from the GSM8K and MATH training subsets; solutions are synthesized by the Mixtral model using text reasoning and Python interpreter-executed code blocks.
- [OpenMathInstruct-2](https://github.com/NVIDIA/NeMo-Skills) is a large-scale math reasoning dataset for training large language models (LLM).
- [AceMath-Instruct](https://huggingface.co/collections/nvidia/acemath-678917d12f09885479d549fe) is a training dataset for cutting-edge mathematical reasoning models in AceMath.
- [QwQ-LongCoT](https://huggingface.co/datasets/PowerInfer/QWQ-LONGCOT-500K) integrates prompts from multiple high-quality sources to create diverse and comprehensive training data.
- [SCP-116K](https://huggingface.co/datasets/EricLu/SCP-116K) is a high-quality set of scientific question-answer pairs automatically extracted from web-scraped documents. Each question is accompanied by a matching solution extracted from the source material, along with responses and reasoning processes generated by advanced language models.
- [R1-Distill-SFT](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT) is distilled using DeepSeek-R1-32b; generated using Numina-math and Tulu; each prompt samples a response.
- [Sky-T1-Data](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k) contains 5k encoded data from APPs and TACO, as well as 10k math data from the AIME, MATH, and Olympiads subsets of the NuminaMATH dataset. It also maintains 1k science and puzzle data from STILL-2.
- [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) is a reasoning dataset that includes questions, reasoning traces, and answers. It replicates and improves upon the Berkeley Sky-T1 data pipeline using SFT distilled data from DeepSeek-R1.
- [s1K](https://huggingface.co/datasets/simplescaling/s1K) contains 1,000 diverse, high-quality, and difficult problem examples (from Gemini Thinking), refining reasoning paths and solutions.
- MedThoughts-8K
- [SYNTHETIC-1](https://www.primeintellect.ai/blog/synthetic-1-release) is the largest open reasoning dataset generated by Deepseek-R1, covering reasoning trajectories for tasks in mathematics, programming, science, etc., with correctness verified by task-specific validators.
- [Medical-R1-Distill-Data](https://huggingface.co/datasets/FreedomIntelligence/Medical-R1-Distill-Data) is an SFT dataset distilled from Deepseek-R1 (Full Power Version), based on HuatuoGPT-o1’s medically verifiable questions.
- [Medical-R1-Distill-Data-Chinese](https://huggingface.co/datasets/FreedomIntelligence/Medical-R1-Distill-Data-Chinese) is an SFT Chinese version dataset distilled from Deepseek-R1 (Full Power Version), based on HuatuoGPT-o1’s medically verifiable questions.
- [RLVR-GSM-MATH](https://github.com/allenai/open-instruct) is used to train the Tulu3 model.
- [LIMO](https://huggingface.co/datasets/GAIR/LIMO) Less is More for Reasoning.
- [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) is an open synthetic reasoning dataset containing 114k high-quality examples covering math, science, code, and puzzles.
- [Magpie-Reasoning-V2](https://github.com/magpie-align/magpie) generates high-quality alignment data by aligning LLMs using pre-query template prompts.
- [Dolphin-R1](https://huggingface.co/datasets/cognitivecomputations/dolphin-r1) is an 800k sample dataset similar in composition to the datasets used for training the DeepSeek-R1 Distill model.


| Name                            | Category                           | Source                            | Modality | Quantity |
| :------------------------------ | :--------------------------------- | :-------------------------------- | :------- | :------- |
| NaturalReasoning                | Science, General                   | Llama3.3-70B                      | Lang     | 1M       |
| NuminaMath-CoT                  | Mathematics                        | GPT-4o                            | Lang     | 860K     |
| NuminaMath-TIR                  | Mathematics                        | GPT-4o                            | Lang     | 73K      |
| DART-Math-uniform               | Mathematics                        | DeepSeekMath-7B-RL                | Lang     | 591K     |
| DART-Math-hard                  | Mathematics                        | DeepSeekMath-7B-RL                | Lang     | 585K     |
| DART-Math-pool-math             | Mathematics                        | DeepSeekMath-7B-RL                | Lang     | 1.6M     |
| DART-Math-pool-gsm8k            | Mathematics                        | DeepSeekMath-7B-RL                | Lang     | 2.7M     |
| OpenO1-SFT                      | Mathematics, Science, General      | -                                 | Lang     | 78K      |
| OpenO1-SFT-Pro                  | Mathematics, Science, General      | -                                 | Lang     | 126K     |
| OpenO1-SFT-Ultra                | Mathematics, Science, General      | -                                 | Lang     | 28M      |
| Medical-o1                      | Medicine                           | DeepSeek R1                       | Lang     | 50K      |
| AoPS-Instruct                   | Mathematics                        | Qwen2.5-72B                       | Lang     | 647K     |
| Orca-Math                       | Mathematics                        | GPT-4                             | Lang     | 200K     |
| MATH-plus                       | Mathematics                        | GPT-4                             | Lang     | 894K     |
| UltraInteract-SFT               | Mathematics, Code, Logic           | GPT-4 CoT + PoT                   | Lang     | 289K     |
| MathCodeInstruct                | Mathematics                        | GPT-4 + Codellama PoT             | Lang     | 79K      |
| MathCodeInstruct-Plus           | Mathematics                        | -                                 | Lang     | 88K      |
| OpenMathInstruct-1              | Mathematics                        | Mixtral-8x7B PoT                  | Lang     | 5M       |
| OpenMathInstruct-2              | Mathematics                        | Llama3.1-405B                     | Lang     | 14M      |
| AceMath-Instruct                | Mathematics, General               | Qwen2.5-Math-72B + GPT-4o-mini    | Lang     | 5M       |
| QwQ-LongCoT                     | General                            | QwQ                               | Lang     | 286K     |
| SCP-116K                        | Science                            | QwQ + O1-mini                     | Lang     | 117K     |
| R1-Distill-SFT                  | Mathematics                        | DeepSeek-R1-32B                   | Lang     | 172K     |
| Sky-T1-Data                     | Mathematics, Code, Science, Puzzle | QwQ                               | Lang     | 17K      |
| Bespoke-Stratos-17k             | Mathematics, Code, Science, Puzzle | DeepSeek R1                       | Lang     | 17K      |
| s1K                             | Mathematics                        | DeepSeek R1                       | Lang     | 1K       |
| MedThoughts-8K                  | Medicine                           | DeepSeek R1                       | Lang     | 8K       |
| SYNTHETIC-1                     | Mathematics, Code, Science         | DeepSeek R1                       | Lang     | 894K     |
| Medical-R1-Distill-Data         | Medicine                           | DeepSeek R1                       | Lang     | 22K      |
| Medical-R1-Distill-Data-Chinese | -                                  | -                                 | Lang     | 17K      |
| RLVR-GSM-MATH                   | Mathematics                        | -                                 | Lang     | 30K      |
| LIMO                            | Mathematics                        | Human + DeepSeek R1 + Qwen2.5-32B | Lang     | 817      |
| OpenThoughts-114k               | Mathematics, Code, Science, Puzzle | -                                 | Lang     | 114K     |
| Magpie-Reasoning-V2             | Mathematics, Code                  | DeepSeek-R1 + Llama-70B           | Lang     | 250K     |
| Dolphin-R1                      | Mathematics, Science               | DeepSeek R1 + Gemini2 + Dolphin   | Lang     | 814K     |

#### 5.2.3 Search-based Distillation

The dataset based on search is constructed through an automated search algorithm, which explores the reasoning tree to generate the optimal reasoning trajectory. Although the scale is limited, these datasets typically generate high-quality and deep reasoning samples.

- [STILL-1](https://arxiv.org/abs/2411.11694) enhances the reasoning capabilities of large language models (LLMs) through a reward-guided tree search algorithm.

| Name    | Category                           | Source                       | Modality | Quantity |
| :------ | :--------------------------------- | :--------------------------- | :------- | :------- |
| STILL-1 | Mathematics, Code, Science, Puzzle | LLaMA-3.1-8B-Instruct + MCTS | Lang     | 5K       |

#### 5.2.4 Validated Distillation
The validated datasets contain rule-based filtering, test case verification, or LLM validation to ensure quality. These datasets strike a balance between scalability and reliability.

- [KodCode-V1](https://huggingface.co/datasets/KodCode/KodCode-V1) provides verifiable solutions and tests for coding tasks; specifically designed for supervised fine-tuning (SFT) and reinforcement learning (RL) optimization; covering various domains (from algorithms to domain-specific software knowledge) and difficulty levels (from basic coding exercises to interview and competitive programming challenges).
- [KodCode-V1-SFT-R1](https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-R1) provides verifiable solutions and tests for coding tasks; specifically designed for supervised fine-tuning (SFT) and reinforcement learning (RL) optimization; covering various domains (from algorithms to domain-specific software knowledge) and difficulty levels (from basic coding exercises to interview and competitive programming challenges).
- [OpenR1-Math](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) is a large-scale mathematics reasoning dataset, generated by DeepSeek R1 for NuminaMath version 1.5 problems, with two to four reasoning trajectories per question.
- [Chinese-DeepSeek-R1-Distill-Data](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k) is a Chinese open-source distilled dataset from DeepSeek-R1, containing not only math data but also a significant amount of general-type data.
- [AM-DeepSeek-R1-Distilled](https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M) includes problems from numerous open-source datasets, which have been semantically deduplicated and cleaned to eliminate test set contamination. The answers are extracted from reasoning models (primarily DeepSeek-R1) and undergo rigorous validation: mathematical problems are verified through answer checking, coding problems through test case validation, and other tasks through reward model evaluation.


| Name                             | Category                      | Source                               | Modality | Quantity |
| :------------------------------- | :---------------------------- | :----------------------------------- | :------- | :------- |
| KodCode-V1                       | -                             | GPT-4 + Test case validation         | Lang     | 447K     |
| KodCode-V1-SFT-R1                | Code                          | DeepSeek R1 + Test case validation   | Lang     | 443K     |
| OpenR1-Math                      | Mathematics                   | DeepSeek R1 + Rule & LLM Validation  | Lang     | 225K     |
| Chinese-DeepSeek-R1-Distill-Data | Mathematics, Science, General | DeepSeek R1 + Rule & LLM Validation  | Lang     | 110K     |
| AM-DeepSeek-R1-Distilled         | Mathematics, Code, General    | Reward Model + Rule & LLM Validation | Lang     | 1.4M     |


## 7. Paper Lists \& Awesome Resources
- [Awesome-Long-Chain-of-Thought-Reasoning](pages/paper.md) (Our Official Paper List, 1000+ papers)

- [Awesome-System2-Reasoning-LLM](https://github.com/zzli2022/Awesome-System2-Reasoning-LLM)

<img src="./assets/images/future.jpg" style="width: 580pt">


# 🎁 Citation
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


# Contribution
For any interesting news about Long CoT, you can also @[Qiguang_Chen](https://twitter.com/QiguangChen) on Twitter or email me at [charleschen2333@gmail.com](mailto:charleschen2333@gmail.com) to follow and update it at our GitHub repo.

Hope everyone enjoy the Long CoT era :)

<!-- omit in toc -->
# ⭐ Star History

<a href="https://star-history.com/#LightChen233/Awesome-Long-Chain-of-Thought-Reasoning&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=LightChen233/Awesome-Long-Chain-of-Thought-Reasoning&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=LightChen233/Awesome-Long-Chain-of-Thought-Reasoning&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=LightChen233/Awesome-Long-Chain-of-Thought-Reasoning&type=Date" />
 </picture>
</a>

