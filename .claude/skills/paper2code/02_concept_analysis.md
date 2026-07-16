# Phase 2: Concept Analysis

## Goal
**Grasp the overall structure** of the research paper and identify **every element that must be implemented** for a successful reproduction.

---

## ⚠️ DO / DON'T Guidelines (CRITICAL)

```
DO:
✓ Systematically map every section of the paper
✓ Identify data flow and dependencies between all components
✓ Identify every environment/dataset/baseline used in the experiments
✓ Define success criteria as concrete numbers
✓ Assess implementation complexity and priority

DON'T:
✗ Don't confuse Related Work with implementation requirements
✗ Don't use abstract success criteria (e.g., "good performance")
✗ Don't omit relationships between components
✗ Don't leave out variants needed for ablation studies
```

---

## ⚠️ Output Format Restrictions (OUTPUT RESTRICTIONS)

```
⚠️ MANDATORY OUTPUT FORMAT:
- Output must be in YAML format only
- Pure YAML only, no markdown explanation or preamble
- Every required field must be filled in
- Include concrete numbers and sources

Output starts with: "```yaml"
Output ends with: "```"
```

---

## Analysis Protocol

### 1. Paper Structure Analysis
Produce a complete map of the paper:

```yaml
paper_structure_map:
  title: "[full paper title]"

  sections:
    1_introduction:
      main_claims: "[what the paper claims to achieve]"
      problem_definition: "[the exact problem being solved]"

    2_related_work:
      key_comparisons: "[methods this work builds on or competes with]"

    3_method:  # may have multiple subsections
      subsections:
        3.1: "[title and main content]"
        3.2: "[title and main content]"
      algorithms_presented: "[list of all algorithm names]"

    4_experiments:
      environments: "[all test environments/datasets]"
      baselines: "[all comparison methods]"
      metrics: "[all evaluation metrics used]"

    5_results:
      main_findings: "[key results that prove the method works]"
      tables_figures: "[important result tables/figures to reproduce]"
```

### 2. Method Decomposition
For the main method/approach:

```yaml
method_decomposition:
  method_name: "[full name and abbreviation]"

  core_components:  # broken down into implementable pieces
    component_1:
      name: "[e.g., State Importance Estimator]"
      purpose: "[why this component exists]"
      paper_section: "[where it is described]"

    component_2:
      name: "[e.g., Policy Refinement Module]"
      purpose: "[its role in the system]"
      paper_section: "[where it is described]"

  component_interactions:
    - "[how component 1 feeds into component 2]"
    - "[data flow between components]"

  theoretical_foundation:
    key_insight: "[the main theoretical insight]"
    why_it_works: "[intuitive explanation]"
```

### 3. Implementation Requirements Mapping
Map the paper's content onto code requirements:

```yaml
implementation_map:
  algorithms_to_implement:
    - algorithm: "[name as given in the paper]"
      section: "[where it is defined]"
      complexity: "[Simple/Medium/Complex]"
      dependencies: "[what it needs in order to run]"

  models_to_build:
    - model: "[neural network or other model]"
      architecture_location: "[section describing it]"
      purpose: "[what this model does]"

  data_processing:
    - pipeline: "[required data preprocessing]"
      requirements: "[what shape the data must be in]"

  evaluation_suite:
    - metric: "[metric name]"
      formula_location: "[where it is defined]"
      purpose: "[what it measures]"
```

### 4. Experiment Reproduction Plan
Identify **every** experiment that needs to be run:

```yaml
experiments_analysis:
  main_results:
    - experiment: "[name/description]"
      proves: "[the claim this validates]"
      requires: "[components needed to run it]"
      expected_outcome: "[concrete numbers/trends]"

  ablation_studies:
    - study: "[what is removed]"
      purpose: "[what this demonstrates]"

  baseline_comparisons:
    - baseline: "[method name]"
      implementation_required: "[Yes/No/Partial]"
      source: "[where an implementation can be found]"
```

### 5. Key Success Factors
Define what a successful reproduction looks like:

```yaml
success_criteria:
  must_achieve:
    - "[key result that must be reproduced]"
    - "[core behavior that must be demonstrated]"

  should_achieve:
    - "[supplementary result that validates the method]"

  validation_evidence:
    - "[specific figure/table to reproduce]"
    - "[qualitative behavior to demonstrate]"
```

---

## Output Format

```yaml
comprehensive_paper_analysis:
  executive_summary:
    paper_title: "[full title]"
    core_contribution: "[one-sentence summary]"
    implementation_complexity: "[Low/Medium/High]"
    estimated_components: "[number of major components to build]"

  complete_structure_map:
    # the full section breakdown above

  method_architecture:
    # the detailed component breakdown

  implementation_requirements:
    # all algorithms, models, data, and metrics

  reproduction_roadmap:
    phase_1: "[what to implement first]"
    phase_2: "[what to build next]"
    phase_3: "[final components and validation]"

  validation_checklist:
    - "[ ] [specific result to achieve]"
    - "[ ] [behavior to demonstrate]"
    - "[ ] [metric to match]"
```

---

## Key Principles

1. **Be thorough**: miss nothing. The output must be a complete blueprint for reproduction
2. **Structure it**: break every part of the paper into implementable pieces
3. **Capture relationships**: make dependencies and data flow between components explicit
4. **State validation criteria**: define exactly what counts as "a successful reproduction"
5. **Set priorities**: distinguish core contributions from supplementary elements

---

## ⚠️ Self-Check: Mandatory Verification Before Completion (MANDATORY)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ SELF-CHECK BEFORE FINISHING (all must be YES to complete)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Paper structure analysis check:
□ Have all Method sections been mapped?                    → YES / NO
□ Are all algorithm names listed?                           → YES / NO
□ Have all experiments in the Experiments section been identified? → YES / NO

Component analysis check:
□ Is the input/output of every component defined?           → YES / NO
□ Is the data flow between components clear?                → YES / NO
□ Has the dependency order been established?                → YES / NO

Experiment requirements check:
□ Have all environments/datasets been identified?           → YES / NO
□ Have all baseline methods been identified?                → YES / NO
□ Have all evaluation metrics been defined?                 → YES / NO
□ Have ablation study variants been identified?              → YES / NO

Success criteria check:
□ Do the must_achieve items include concrete numbers?        → YES / NO
□ Are specific tables/figures to reproduce specified?         → YES / NO

Output format check:
□ Was the output in pure YAML format?                        → YES / NO
□ Are all required fields filled in?                          → YES / NO

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ IF EVEN ONE IS NO, KEEP ANALYZING UNTIL IT'S COMPLETE!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
