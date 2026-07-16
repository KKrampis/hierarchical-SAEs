# Phase 3: Implementation Planning (Code Planning)

## Goal
Integrate the results of Phase 1 (Algorithm Extraction) and Phase 2 (Concept Analysis) to produce a detailed plan that lets a developer implement the entire thing **without reading the paper**.

---

## ⚠️ Content Length Guidelines (STRICTLY FOLLOW)

```
📏 CONTENT BALANCE GUIDELINES:

Section 1 (file_structure):           ~800-1000 characters
Section 2 (implementation_components): ~3000-4000 characters  ← the core section
Section 3 (validation_approach):       ~2000-2500 characters
Section 4 (environment_setup):         ~800-1000 characters
Section 5 (implementation_strategy):   ~1500-2000 characters

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Total Target: 8000-10000 characters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ Section 2 is the most important! Include every algorithm, equation, and parameter
⚠️ If the length falls short, details are missing — double-check it
```

---

## Inputs
1. **Phase 1 result**: the complete algorithm extraction (algorithm_extraction.yaml)
2. **Phase 2 result**: the comprehensive paper analysis (concept_analysis.yaml)

---

## Planning Process

### 1. Integrate the Information
Combine **everything** from both analyses:
- Every algorithm and its pseudocode
- Every component and its architecture
- Every hyperparameter and value
- Every experiment and expected result

### 2. Implementation Mapping
Connect each component to a concrete implementation:

```
[For each algorithm/component/method in the paper]:
  - what it does in the paper and where it is described
  - how to organize it in code (file, class, function)
  - the specific equations, algorithms, and procedures needed to implement it
  - dependencies and relationships with other components
  - an implementation approach suited to this particular paper
```

### 3. Extract Technical Details
Collect every technical detail relevant to implementation:

```
[Collect every implementation-relevant detail from the paper]:
  - every algorithm, with complete pseudocode and mathematical formulation
  - every parameter, hyperparameter, and configuration value
  - every architectural detail (where applicable)
  - every experimental procedure and evaluation method
  - any implementation hints, tricks, or special considerations mentioned
```

---

## Output Format: 5 Required Sections

```yaml
complete_reproduction_plan:
  paper_info:
    title: "[full paper title]"
    core_contribution: "[the main innovation to reproduce]"

  # ============================================
  # Section 1: File Structure (~800-1000 characters)
  # ============================================
  # Design the file organization that best fits this paper
  # - Analyze the paper's content (algorithms, models, experiments, systems, etc.)
  # - Organize files and directories in the most logical way for implementation
  # - Use meaningful names and grouping based on the paper's content
  # - Keep it clean, intuitive, and focused on the actual implementation
  # - Include documentation files (README.md, requirements.txt), but implement them last

  file_structure: |
    project_name/
    ├── main.py                    # Main entry point
    ├── config.py                  # Configuration and hyperparameters
    ├── models/
    │   ├── __init__.py
    │   ├── network.py             # Core network architecture
    │   └── components.py          # Individual components
    ├── algorithms/
    │   ├── __init__.py
    │   └── core_algorithm.py      # Main algorithm implementation
    ├── training/
    │   ├── __init__.py
    │   ├── trainer.py             # Training loop
    │   └── losses.py              # Loss functions
    ├── evaluation/
    │   ├── __init__.py
    │   └── metrics.py             # Evaluation metrics
    ├── utils/
    │   ├── __init__.py
    │   └── helpers.py             # Utility functions
    ├── experiments/
    │   └── run_experiments.py     # Experiment scripts
    ├── requirements.txt           # Dependencies (implement last)
    └── README.md                  # Documentation (implement last)

  # ============================================
  # Section 2: Implementation Components (~3000-4000 characters) — the core section
  # ============================================
  # Identify and specify every component that needs to be implemented
  # - List every algorithm, model, system, and component mentioned
  # - For each: purpose, location, algorithm, equations, technical details
  # - Organize according to the paper's actual content

  implementation_components: |
    ## 1. Core Algorithm

    ### 1.1 [Algorithm name]
    - Location: algorithms/core_algorithm.py
    - Purpose: [what this algorithm does]
    - Pseudocode:
      ```
      [copy the paper's pseudocode]
      ```
    - Key equations:
      - [Eq. X]: L = ...
      - [Eq. Y]: ...
    - Hyperparameters:
      - param1: value1 (source: Section X)
      - param2: value2 (source: Table Y)

    ## 2. Model Architecture

    ### 2.1 [Model/network name]
    - Location: models/network.py
    - Input: [shape, meaning]
    - Output: [shape, meaning]
    - Layer composition:
      - Layer 1: ...
      - Layer 2: ...
    - Special initialization: [if any]

    ## 3. Training Procedure

    ### 3.1 Training loop
    - Location: training/trainer.py
    - Epochs/Iterations: [value]
    - Steps:
      1. [description of Step 1]
      2. [description of Step 2]

    ### 3.2 Loss function
    - Location: training/losses.py
    - Formula: L_total = ...
    - Meaning of each term: ...

    ## 4. Evaluation

    ### 4.1 Evaluation metrics
    - Location: evaluation/metrics.py
    - List of metrics: [metric1, metric2, ...]
    - How each metric is computed: ...

  # ============================================
  # Section 3: Validation Approach (~2000-2500 characters)
  # ============================================
  # Design a way to verify that the implementation works correctly
  # - Define the necessary experiments, tests, and proofs
  # - State the paper's expected results (figures, tables, theorems)
  # - Design a validation approach suited to the domain
  # - Include setup requirements and success criteria

  validation_approach: |
    ## 1. Unit Tests
    - [ ] Each component produces the correct output shape
    - [ ] The loss function returns the correct value
    - [ ] Gradients flow correctly

    ## 2. Integration Tests
    - [ ] The full training pipeline runs
    - [ ] Overfitting test on a small dataset

    ## 3. Reproducing the Paper's Results

    ### 3.1 Reproducing Table X
    - Expected result: [concrete numbers]
    - Tolerance: ±[value]
    - How to run: `python experiments/run_experiments.py --exp table_x`

    ### 3.2 Reproducing Figure Y
    - Expected behavior: [qualitative description]
    - How to run: `python experiments/run_experiments.py --exp figure_y`

    ## 4. Success Criteria
    - [ ] [concrete result 1]
    - [ ] [concrete result 2]
    - [ ] [qualitative behavior 1]

  # ============================================
  # Section 4: Environment Setup (~800-1000 characters)
  # ============================================
  # Specify what's needed to run the implementation
  # - Programming language and version requirements
  # - External libraries and exact versions (if specified in the paper)
  # - Hardware requirements (GPU, memory, etc.)
  # - Any special configuration or installation steps

  environment_setup: |
    ## Python Version
    - Python 3.10+ (uv recommended)

    ## Package Management (using uv — recommended)
    Use uv to build an isolated, reproducible environment:
    ```bash
    # Initialize the project
    uv init

    # Add dependencies
    uv add torch numpy [other required packages]

    # Run
    uv run python main.py
    ```

    ## Core Dependencies
    ```
    torch>=2.0.0
    numpy>=1.24.0
    [other required packages]
    ```

    ## Hardware Requirements
    - GPU: [NVIDIA GPU with X GB VRAM]
    - RAM: [minimum X GB]
    - Storage: [X GB]

    ## Dataset Preparation
    - [dataset name]: [how to download]
    - Preprocessing: [required steps]

  # ============================================
  # Section 5: Implementation Strategy (~1500-2000 characters)
  # ============================================
  # Plan a step-by-step implementation approach
  # - Break the implementation into logical stages
  # - Identify dependencies between components
  # - Plan testing and validation at each stage
  # - Handle missing details with reasonable defaults

  implementation_strategy: |
    ## Phase 1: Build the Foundation (first)
    1. config.py - define all hyperparameters
    2. utils/helpers.py - common utility functions

    Validation: test that configuration loads correctly

    ## Phase 2: Core Implementation
    3. models/components.py - individual components
    4. models/network.py - the full network
    5. algorithms/core_algorithm.py - the main algorithm

    Validation: check the output shape of each component

    ## Phase 3: Training Pipeline
    6. training/losses.py - loss functions
    7. training/trainer.py - the training loop

    Validation: overfitting test on small data

    ## Phase 4: Evaluation and Experiments
    8. evaluation/metrics.py - evaluation metrics
    9. experiments/run_experiments.py - experiment scripts
    10. main.py - main entry point

    Validation: reproduce the paper's results

    ## Phase 5: Documentation (last)
    11. pyproject.toml - uv project configuration and dependencies
    12. README.md - usage documentation (including uv run commands)

    ## Handling Missing Details
    - [thing 1 not covered in the paper]: [suggested default]
    - [thing 2 not covered in the paper]: [suggested approach]
```

---

## Key Principles

1. **Completeness**: all 5 sections must be included
2. **Detail**: every algorithm, equation, parameter, and file must be specified
3. **Executability**: this plan alone must be enough to write the code
4. **Logical order**: present an implementation order that respects dependencies
5. **Include validation**: state success criteria and testing methods

## File Priority Guidelines

1. **First**: core algorithm/model files (highest priority)
2. **Second**: supporting modules and utilities
3. **Third**: experiment and evaluation scripts
4. **Fourth**: configuration and data processing
5. **Last**: documentation files (README.md, requirements.txt)

**Note**: README and requirements.txt depend on the final implementation, so write them last

---

## ⚠️ Self-Check: Mandatory Verification Before Completion (MANDATORY)

Before considering the implementation plan complete, always verify the following:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ SELF-CHECK BEFORE FINISHING (all must be YES to complete)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Section inclusion check:
□ Is the file_structure section included?             → YES / NO
□ Is the implementation_components section included?  → YES / NO
□ Is the validation_approach section included?         → YES / NO
□ Is the environment_setup section included?           → YES / NO
□ Is the implementation_strategy section included?     → YES / NO

Content completeness check:
□ Is every algorithm in the paper mapped to a component?     → YES / NO
□ Does every equation have an Equation number and source?    → YES / NO
□ Does every hyperparameter have a value and source?         → YES / NO
□ Does the implementation order correctly reflect dependencies? → YES / NO
□ Does the validation approach include concrete expected results? → YES / NO

Length check:
□ Is the total length at least 8000 characters?         → YES / NO
□ Is Section 2 the most detailed section?               → YES / NO

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ IF EVEN ONE IS NO, KEEP WRITING UNTIL IT'S COMPLETE!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## DO / DON'T Guidelines

```
DO:
✓ Integrate every extraction result from Phase 1 and Phase 2 into the plan
✓ Specify the exact algorithms/equations to implement in each file
✓ Clearly mark dependencies between files
✓ Make the plan self-contained with all information needed for implementation
✓ Specify concrete numbers/behaviors in the validation approach

DON'T:
✗ Don't write incomplete descriptions like "see the paper for details"
✗ Don't describe components abstractly (without concrete equations/algorithms)
✗ Don't write an implementation plan with no validation approach
✗ Don't order files while ignoring dependencies
✗ Don't write the core section (Section 2) briefly
```
