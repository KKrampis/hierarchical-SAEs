# Memory and Context Management Guide

## Purpose
When processing long papers, **prevent context overflow** and ensure **efficient progress**.

---

## Core Strategies

### 1. Save the Output of Each Phase

Save the result of each completed Phase to a file to reduce context load:

```
paper_workspace/
├── paper.txt                      # original paper text
├── 01_algorithm_extraction.yaml   # Phase 1 result
├── 02_concept_analysis.yaml       # Phase 2 result
├── 03_implementation_plan.yaml    # Phase 3 result
└── src/                           # code generated in Phase 4
    ├── config.py
    ├── models/
    ├── algorithms/
    └── ...
```

**Example save commands:**
```bash
# Save the Phase 1 result
cat > paper_workspace/01_algorithm_extraction.yaml << 'EOF'
[Phase 1 YAML output]
EOF

# Save the Phase 2 result
cat > paper_workspace/02_concept_analysis.yaml << 'EOF'
[Phase 2 YAML output]
EOF
```

### 2. Passing Context Between Phases

When moving to the next Phase, **pass only a core summary instead of the full output**:

```yaml
# Phase 1 → Phase 2 handoff summary
phase1_summary:
  algorithms_found: 3
  key_algorithms:
    - "Algorithm 1: [name] - [one-line summary]"
    - "Algorithm 2: [name] - [one-line summary]"
  hyperparameters_count: 15
  critical_equations: [3, 5, 7, 12]

# Phase 2 → Phase 3 handoff summary
phase2_summary:
  components_count: 5
  implementation_complexity: "Medium"
  key_dependencies:
    - "Component A → Component B"
    - "Component B → Component C"
  experiments_count: 4
```

### 3. Memory Optimization During Implementation

Manage context during the per-file implementation cycle:

```
File implementation cycle:
┌─────────────────────────────────────────────────────┐
│ 1. Load only the information needed for the current  │
│    file                                               │
│    - the relevant section from implementation_plan.yaml │
│    - the interfaces of dependent files (not their     │
│      full code)                                        │
├─────────────────────────────────────────────────────┤
│ 2. Implement the file                                 │
├─────────────────────────────────────────────────────┤
│ 3. After implementation, move to the next file        │
│    - reference the content of previous files only      │
│      when needed                                       │
│    - don't keep the full code resident in memory       │
└─────────────────────────────────────────────────────┘
```

---

## Tips for Handling Long Papers

### Reading the Paper in Sections

If the paper is very long, analyze it section by section:

```
Reading order (by priority):
1. Abstract + Introduction (grasp the core contribution)
2. The entire Method section (algorithm extraction)
3. Experiments section (environments, baselines, metrics)
4. Appendix (detailed hyperparameters)
5. Related Work (only if needed)

Can be skipped:
- Detailed content of Related Work (not needed for implementation)
- Long Discussion/Conclusion (summary only)
- Acknowledgments
```

### Splitting Up Large Algorithms

Break a complex algorithm down into sub-components:

```yaml
# Process in pieces rather than all at once
large_algorithm:
  component_1:
    extracted: true
    summary: "[summary]"
  component_2:
    extracted: true
    summary: "[summary]"
  component_3:
    extracted: false  # not yet processed
```

---

## Self-Monitoring Checkpoints

**Intermediate saves** are recommended in the following situations during implementation:

```
Intermediate save triggers:
□ Every time 5 files have been implemented
□ After implementing a complex algorithm (50+ lines)
□ Save the current state when an error occurs
□ Before starting a new Phase
□ Before starting a long task (expected to take 30+ minutes)
```

### Save Checklist

```yaml
checkpoint_save:
  current_phase: "[current Phase number]"
  completed_files:
    - "config.py"
    - "models/network.py"
  current_file: "algorithms/core.py"
  current_progress: "50%"  # progress on the current file
  next_steps:
    - "[next task 1]"
    - "[next task 2]"
  blockers:
    - "[anything currently blocking progress, if any]"
```

---

## Context Recovery Protocol

If the conversation is interrupted or context is lost:

```
Recovery steps:
1. Check the paper_workspace/ directory
2. Read the most recently completed Phase's result file
3. Check the list of generated code files
4. Determine the last point worked on
5. Resume from that point
```

**Example recovery commands:**
```bash
# Check the current state
ls -la paper_workspace/
ls -la paper_workspace/src/

# Check the last Phase's result
cat paper_workspace/03_implementation_plan.yaml

# Check the generated files
find paper_workspace/src -name "*.py" -type f
```

---

## Efficient Reference Patterns

### Reference Only the Interface

When referencing another file, you usually only need **its interface, not the full implementation**:

```python
# Reference only the signature instead of the full code
# Interface of models/network.py:
class NetworkModel:
    def __init__(self, config: Config): ...
    def forward(self, x: Tensor) -> Tensor: ...
    def get_features(self, x: Tensor) -> Tensor: ...
```

### Use a Dependency Graph

When deciding implementation order, refer to the dependency graph:

```
config.py (no dependencies)
    ↓
utils/helpers.py (depends only on config)
    ↓
models/components.py (depends on config, utils)
    ↓
models/network.py (depends on components)
    ↓
algorithms/core.py (depends on network)
    ↓
training/trainer.py (depends on everything)
```

---

## ⚠️ Notes

```
⚠️ MEMORY MANAGEMENT RULES:

1. Don't process the entire paper at once
   → process it section by section

2. Don't include the previous Phase's entire output in the next Phase
   → pass only a core summary

3. Don't keep all generated code resident in memory
   → save it to files and reference it only when needed

4. Periodically save progress during long tasks
   → so recovery is possible after an interruption

5. Avoid unnecessary repeated reading
   → keep a summary of information once it has been read
```

---

## Recommended Workflow

```
[Paper input]
    │
    ▼
[Phase 1: Algorithm extraction]
    │ → save 01_algorithm_extraction.yaml
    │ → produce core summary
    ▼
[Phase 2: Concept analysis]
    │ → save 02_concept_analysis.yaml
    │ → keep Phase 1 summary + Phase 2 summary
    ▼
[Phase 3: Implementation planning]
    │ → save 03_implementation_plan.yaml
    │ → keep only the core information needed for implementation
    ▼
[Phase 4: Code implementation]
    │ → implement and save file by file
    │ → checkpoint every 5 files
    ▼
[Complete]
```
