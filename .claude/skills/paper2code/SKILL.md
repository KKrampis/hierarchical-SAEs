---
name: paper2code
description: |
  Analyzes a research paper (PDF/arXiv URL) and converts it into executable code.
  Automatically activates on requests to replicate a paper, implement an algorithm,
  or reproduce research. Responds to requests like "implement this paper",
  "paper2code", or "convert this paper into code".
---

# Paper2Code: An AI Agent That Converts Research Papers into Code

## Overview

This Skill runs a **4+2 phase pipeline** that systematically analyzes a research paper and converts it into executable code.

**Core principle**: Rather than simply reading the paper and generating code, this skill first produces a **structured intermediate representation (YAML)** and then writes the code from that.

---

## ⚠️ Core Behavioral Control Rules (CRITICAL)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ MANDATORY BEHAVIORAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Implement only one file at a time
2. After implementing a file, move to the next file without asking for confirmation/permission
3. The paper's original specification always takes priority over reference code
4. The Self-Check for each Phase must be performed before completion
5. Save every intermediate result as a YAML file

DO:
✓ Implement exactly what is specified in the paper
✓ Write simple, direct code
✓ Prioritize working code over elegant code
✓ Test each component immediately
✓ Move on to the next file as soon as implementation is complete

DON'T:
✗ Ask "should I implement the next file?" between files
✗ Write extensive documentation not required for core functionality
✗ Add optimizations not required for reproduction
✗ Introduce excessive abstraction or design patterns
✗ Provide instructions only, without writing actual code
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Input Handling

### Supported Formats
1. **arXiv URL**: `https://arxiv.org/abs/xxxx.xxxxx` or `https://arxiv.org/pdf/xxxx.xxxxx.pdf`
2. **PDF file path**: `/path/to/paper.pdf`
3. **Already-converted text/markdown**: when the paper content is already supplied as text

### How to Handle Input

**If it's an arXiv URL:**
```bash
# Convert to the PDF URL and download it
curl -L "https://arxiv.org/pdf/xxxx.xxxxx.pdf" -o paper.pdf

# Convert the PDF to text (using pdftotext)
pdftotext -layout paper.pdf paper.txt
```

**If it's a PDF file:**
```bash
pdftotext -layout "/path/to/paper.pdf" paper.txt
```

---

## Pipeline Overview

```
[User input: paper URL/file]
        │
        ▼
┌─────────────────────────────────────────────┐
│ Step 0: Obtain the paper text                │
│ - arXiv URL → download PDF                   │
│ - PDF → convert to text                      │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ Phase 0: Reference code search (optional)    │
│ @[05_reference_search.md]                    │
│ Output: reference_search.yaml                │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ Phase 1: Algorithm extraction                │
│ @[01_algorithm_extraction.md]                │
│ Output: 01_algorithm_extraction.yaml         │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ Phase 2: Concept analysis                    │
│ @[02_concept_analysis.md]                    │
│ Output: 02_concept_analysis.yaml             │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ Phase 3: Implementation planning             │
│ @[03_code_planning.md]                       │
│ Output: 03_implementation_plan.yaml          │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ Phase 4: Code implementation                 │
│ @[04_implementation_guide.md]                │
│ Output: complete project directory           │
└─────────────────────────────────────────────┘
```

---

## Data Handoff Format Between Phases

### Phase 1 → Phase 2 handoff
```yaml
phase1_to_phase2:
  algorithms_found: "[number of algorithms found]"
  key_algorithms:
    - name: "[algorithm name]"
      section: "[paper section]"
      complexity: "[Simple/Medium/Complex]"
  hyperparameters_count: "[number of hyperparameters collected]"
  critical_equations: "[list of key equation numbers]"
  missing_info: "[list of missing information]"
```

### Phase 2 → Phase 3 handoff
```yaml
phase2_to_phase3:
  components_count: "[number of components identified]"
  implementation_complexity: "[Low/Medium/High]"
  key_dependencies:
    - "[component A] → [component B]"
  experiments_to_reproduce:
    - "[experiment name]: [expected result]"
  success_criteria:
    - "[concrete success criterion]"
```

### Phase 3 → Phase 4 handoff
```yaml
phase3_to_phase4:
  file_order: "[list of files in implementation order]"
  current_file: "[file currently being implemented]"
  completed_files: "[list of completed files]"
  blocking_dependencies: "[dependencies that must be resolved]"
```

---

## Phase Details

### Phase 0: Reference Code Search (Optional)
Using the @[05_reference_search.md](05_reference_search.md) prompt:
- Search for and evaluate 5 similar implementations
- Secure references to improve implementation quality
- **Output**: a YAML-formatted list of references

### Phase 1: Algorithm Extraction
Using the @[01_algorithm_extraction.md](01_algorithm_extraction.md) prompt:
- Extract every algorithm, equation, and pseudocode block
- Collect hyperparameters and configuration values
- Organize the training procedure and optimization method
- **Output**: a complete YAML-formatted algorithm specification

### Phase 2: Concept Analysis
Using the @[02_concept_analysis.md](02_concept_analysis.md) prompt:
- Map the paper's structure and sections
- Analyze the system architecture
- Identify relationships and data flow between components
- Organize experiment and validation requirements
- **Output**: a YAML-formatted implementation requirements specification

### Phase 3: Implementation Planning
Using the @[03_code_planning.md](03_code_planning.md) prompt:
- Integrate the results of Phase 1 and Phase 2
- Produce a detailed implementation plan with 5 required sections:
  1. `file_structure`: the project's file structure
  2. `implementation_components`: details of the components to implement
  3. `validation_approach`: validation and testing methodology
  4. `environment_setup`: environment and dependencies
  5. `implementation_strategy`: the step-by-step implementation strategy
- **Output**: a complete YAML implementation plan (8,000–10,000 characters)

### Phase 4: Code Implementation
Following the @[04_implementation_guide.md](04_implementation_guide.md) guide:
- Generate code file by file according to the plan
- Implement in dependency order
- Every file must be complete and executable
- **Output**: an executable codebase

---

## Memory Management
Refer to the @[06_memory_management.md](06_memory_management.md) guide for:
- Managing context when processing long papers
- Saving output at each phase
- The recovery protocol if interrupted

---

## Quality Standards

### Principles That Must Be Followed
- **Completeness**: a complete implementation, with no placeholders or TODOs
- **Accuracy**: equations and parameters must accurately reflect what the paper specifies
- **Executability**: code that can be run immediately
- **Reproducibility**: must be able to reproduce the paper's results

### File Implementation Order
1. Configuration and environment files (config, initial requirements.txt)
2. Core utilities and base classes
3. Main algorithm/model implementation
4. Training and evaluation scripts
5. Documentation (README.md, finalized requirements.txt)

---

## ✅ Final Completion Checklist (MANDATORY)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ BEFORE DECLARING COMPLETE - ALL MUST BE YES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

□ Are all algorithms from the paper implemented?              → YES / NO
□ Is every environment/dataset configured with the correct version? → YES / NO
□ Are all comparison methods referenced in the experiments implemented? → YES / NO
□ Is there a working integration that can run the paper's experiments? → YES / NO
□ Can all metrics, figures, and tables be reproduced?          → YES / NO
□ Is there basic documentation explaining how to reproduce the results? → YES / NO
□ Does the code run without errors?                             → YES / NO

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ IF EVEN ONE IS NO, IT IS NOT COMPLETE!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Usage Examples

### Example 1: arXiv paper
```
User: implement this paper https://arxiv.org/abs/2301.12345

Claude: I'll analyze the paper and convert it into code.

[Phase 0: Reference code search (optional)...]
[Phase 1: Algorithm extraction...]
[Phase 2: Concept analysis...]
[Phase 3: Implementation planning...]
[Phase 4: Code generation...]
```

### Example 2: PDF file
```
User: implement the algorithm in this paper /home/user/papers/attention.pdf
```

### Example 3: Requesting only a specific part
```
User: implement only the algorithm from Section 3 of this paper
```

---

## Related Files

- [01_algorithm_extraction.md](01_algorithm_extraction.md) - Phase 1: Algorithm extraction
- [02_concept_analysis.md](02_concept_analysis.md) - Phase 2: Concept analysis
- [03_code_planning.md](03_code_planning.md) - Phase 3: Implementation planning
- [04_implementation_guide.md](04_implementation_guide.md) - Phase 4: Implementation guide
- [05_reference_search.md](05_reference_search.md) - Phase 0: Reference search (optional)
- [06_memory_management.md](06_memory_management.md) - Memory management guide

---

## Notes

```
⚠️ REMEMBER:

1. Read the paper thoroughly: grasp the full content before starting implementation
2. Save intermediate results: save each Phase's YAML output to a file
3. Implement incrementally: proceed file by file rather than generating all the code at once
4. Include validation: include simple test code wherever possible
5. Treat references as inspiration: reference code is for understanding and adaptation, not copying
```
