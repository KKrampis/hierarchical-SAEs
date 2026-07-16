# Phase 1: Algorithm Extraction

## Goal
Extract **every technical detail** needed for implementation from the research paper.
A developer must be able to implement the entire paper using only this extraction result.

---

## ⚠️ DO / DON'T Guidelines (CRITICAL)

```
DO:
✓ Copy pseudocode from the paper exactly (do not change a single character)
✓ Record equations precisely, along with their equation numbers (Eq. X)
✓ Search for hyperparameters everywhere — text, tables, captions, appendices
✓ Identify and record anything missing but essential for implementation
✓ Cite a source (Section X, Table Y, Page Z) for every piece of information
✓ Keep variable names, symbols, and subscripts exactly as in the paper

DON'T:
✗ Don't "simplify" equations or pseudocode to make them easier to understand
✗ Don't guess at parameter values that aren't in the paper
✗ Don't record information without a source
✗ Don't substitute "commonly used" values
✗ Don't skip unclear parts (record them in missing_but_critical instead)
```

---

## ⚠️ Output Format Restrictions (OUTPUT RESTRICTIONS)

```
⚠️ MANDATORY OUTPUT FORMAT:
- Output must be in YAML format only
- Pure YAML only, no markdown explanation or preamble
- Every required field must be filled in
- For fields with no information, record "Not specified in paper"
- Tag any guessed value with "[INFERRED]"

Output starts with: "```yaml"
Output ends with: "```"
```

---

## Extraction Protocol

### 1. Algorithm Scan
Find and extract all of the following from the paper:
- Everything in the Method/Algorithm section(s)
- Algorithm boxes (Algorithm 1, 2, 3...)
- Equations and formulas (every Equation)
- Pseudocode
- Implementation Details

### 2. Deep Algorithm Extraction
For **every** algorithm/method/procedure found:

```yaml
algorithm_name: "[exact name as given in the paper]"
section: "[e.g., Section 3.2]"
algorithm_box: "[e.g., Algorithm 1 on page 4]"

pseudocode: |
  [copy the paper's pseudocode exactly]
  Input: ...
  Output: ...
  1. Initialize ...
  2. For each ...
     2.1 Calculate ...
  [preserve the exact formatting and numbering]

mathematical_formulation:
  - equation: "[copy the equation exactly, e.g., L = L_task + λ*L_explain]"
    equation_number: "[e.g., Eq. 3]"
    where:
      L_task: "task loss"
      L_explain: "explanation loss"
      λ: "weighting parameter (default: 0.5)"

step_by_step_breakdown:
  1. "[detailed explanation of what Step 1 does]"
  2. "[what Step 2 computes, and why]"

implementation_details:
  - "Uses softmax temperature τ = 0.1"
  - "Gradient clipping at norm 1.0"
  - "Initialize weights with Xavier uniform"
```

### 3. Component Extraction
For **every** component/module mentioned:

```yaml
component_name: "[e.g., Mask Network, Critic Network]"
purpose: "[this component's role in the system]"
architecture:
  input: "[shape and meaning]"
  layers:
    - "[Conv2d(3, 64, kernel=3, stride=1)]"
    - "[ReLU activation]"
    - "[BatchNorm2d(64)]"
  output: "[shape and meaning]"

special_features:
  - "[a distinctive characteristic]"
  - "[a special initialization method]"
```

### 4. Training Procedure Extraction
Extract the **complete** training process:

```yaml
training_loop:
  outer_iterations: "[count or condition]"
  inner_iterations: "[count or condition]"

  steps:
    1. "Sample batch of size B from buffer"
    2. "Compute importance weights using..."
    3. "Update policy with loss..."

  loss_functions:
    - name: "policy_loss"
      formula: "[the exact equation]"
      components: "[meaning of each term]"

  optimization:
    optimizer: "Adam"
    learning_rate: "3e-4"
    lr_schedule: "linear decay to 0"
    gradient_norm: "clip at 0.5"
```

### 5. Hyperparameter Collection
Search **everywhere** — text, tables, captions:

```yaml
hyperparameters:
  # Training
  batch_size: 64
  buffer_size: 1e6
  discount_gamma: 0.99

  # Architecture
  hidden_units: [256, 256]
  activation: "ReLU"

  # Algorithm-specific
  explanation_weight: 0.5
  exploration_bonus_scale: 0.1
  reset_probability: 0.3

  # Sources
  location_references:
    - "batch_size: Table 1"
    - "hidden_units: Section 4.1"
```

---

## Output Format

```yaml
complete_algorithm_extraction:
  paper_structure:
    method_sections: "[3, 3.1, 3.2, 3.3, 4]"
    algorithm_count: "[total number of algorithms found]"

  main_algorithm:
    # Fill in with full detail using the format above

  supporting_algorithms:
    - # Details for each supporting algorithm

  components:
    - # Every component and its architecture

  training_details:
    # The complete training procedure

  all_hyperparameters:
    # Every parameter, its value, and its source

  implementation_notes:
    - "[implementation hints mentioned in the paper]"
    - "[tricks mentioned in the text]"

  missing_but_critical:
    - "[something not stated but essential]"
    - "[along with a suggested default value]"
```

---

## Key Principles

1. **Be thorough**: a developer must be able to implement the entire paper using **only** this extraction result
2. **Be accurate**: copy equations, variable names, and values **exactly**
3. **Be exhaustive**: every algorithm, every equation, every parameter
4. **Cite sources**: record where in the paper each piece of information came from
5. **Identify gaps**: identify what's needed for implementation but not stated in the paper

---

## ⚠️ Self-Check: Mandatory Verification Before Completion (MANDATORY)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ SELF-CHECK BEFORE FINISHING (all must be YES to complete)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Algorithm extraction check:
□ Have all Algorithm boxes (Algorithm 1, 2, ...) been extracted?  → YES / NO
□ Is every procedure in the Method section included?              → YES / NO
□ Does every equation have an Equation number?                    → YES / NO

Hyperparameter check:
□ Have all parameters mentioned in the body text been collected?  → YES / NO
□ Have all parameters mentioned in tables been collected?         → YES / NO
□ Have parameters mentioned in captions/appendices been checked too? → YES / NO

Completeness check:
□ Is the training procedure fully described?                      → YES / NO
□ Are all terms of the loss function defined?                     → YES / NO
□ Is missing essential information recorded in missing_but_critical? → YES / NO

Output format check:
□ Was the output in pure YAML format?                              → YES / NO
□ Are all required fields filled in?                                → YES / NO

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ IF EVEN ONE IS NO, KEEP EXTRACTING UNTIL IT'S COMPLETE!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
