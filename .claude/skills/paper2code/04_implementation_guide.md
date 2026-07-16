# Phase 4: Implementation Guide (Code Implementation)

## Goal
Produce a **complete, executable codebase** based on the implementation plan generated in Phase 3.

---

## Core Behavioral Control Rules

### ⚠️ CRITICAL BEHAVIORAL RULES

```
⚠️ SINGLE FILE PER RESPONSE:
- Implement exactly one file per response
- Do not ask for permission between files
- Keep implementing until it is complete

DO:
- Implement exactly what is specified in the paper
- Write simple, direct code
- Prioritize working code over elegant code
- Test each component immediately
- Move on to the next file as soon as implementation is complete

DON'T:
- Ask "should I implement the next file?" between files
- Waste time on fancy tooling instead of the paper's requirements
- Write extensive documentation not required for core functionality
- Add optimization utilities not required for reproduction
- Introduce excessive abstraction or design patterns
- Provide instructions only, without writing actual code
```

### Tool-Calling Strategy

```
TOOL CALLING STRATEGY:
1. ⚠️ Implement one file per message
2. Check the result, then plan the next step
3. File implementation cycle: analyze → implement → next file

EXECUTION PATTERN:
- Plan First: explain the reasoning before each task
- One Step at a Time: execute → check result → plan next → execute
- Iterative Progress: build the solution incrementally
- Strategic Sequencing: choose the next logical step based on prior results

⚠️ CRITICAL: use the bash and python tools to directly replicate the paper
            — don't just give instructions, actually implement it
```

---

## Top Priority

Implement **every** algorithm, experiment, and method mentioned in the paper.
Success is measured by **completeness and accuracy**, not by code elegance.

### Core Strategy
- Read the paper and the implementation plan thoroughly and identify every algorithm, method, and experiment
- Implement the core algorithm first, then the environment, then the integration
- Use the exact versions and specifications stated in the paper
- Test each component immediately after implementing it
- Focus on a working implementation rather than a perfect architecture

---

## Implementation Approach

### Building Incrementally, File by File
At each step:
1. **Identify**: check what to implement next from the implementation plan
2. **Implement**: implement one component at a time
3. **Test**: test immediately to catch problems early
4. **Integrate**: integrate with existing components
5. **Validate**: verify against the paper's specification

---

## Implementation Order

### Step 1: Configuration and Environment Files
```
pyproject.toml     # uv project configuration (created via uv init)
config.py          # all hyperparameters and settings
```

### Step 2: Core Utilities and Base Classes
```
utils/__init__.py
utils/helpers.py   # common utility functions
```

### Step 3: Main Implementation Modules
```
models/__init__.py
models/network.py      # core network architecture
models/components.py   # individual components

algorithms/__init__.py
algorithms/core.py     # main algorithm implementation
```

### Step 4: Training Pipeline
```
training/__init__.py
training/losses.py    # loss functions
training/trainer.py   # training loop
```

### Step 5: Evaluation and Experiments
```
evaluation/__init__.py
evaluation/metrics.py        # evaluation metrics
experiments/run_main.py      # main experiment script
```

### Step 6: Entry Point and Documentation
```
main.py            # main entry point
README.md          # usage documentation (including uv run commands)
```

### Environment Setup Commands (using uv)
```bash
# When starting the project
uv init
uv add torch numpy [required packages]

# Run
uv run python main.py
```

---

## Code Quality Standards

### Completeness
- **No** placeholders, TODOs, or incomplete functions
- Fully implemented functionality with proper error handling
- Complete APIs with correct signatures and documentation
- Every stated feature works immediately

### Quality
- Production-level code that follows language best practices
- Comprehensive type hints and docstrings
- Proper logging, validation, and resource management
- A clean architecture with separation of concerns

### Domain-Specific Adaptation

**Research/ML papers:**
- Mathematical accuracy
- Reproducibility (seeds, deterministic operations)
- Evaluation metrics
- Experiment logging

**Systems/tools:**
- CLI interface
- Configuration management
- Error handling
- Documentation

---

## ✅ Completion Checklist (MANDATORY)

Before considering the work complete, always verify:

```
✅ COMPLETENESS CHECKLIST:
- [ ] Every algorithm mentioned in the paper (including abbreviations or alternate names)
- [ ] Every environment/dataset at the exact version specified
- [ ] Every comparison method referenced in the experiments
- [ ] A working integration that can run the paper's experiments
- [ ] A complete codebase that reproduces all of the paper's metrics, figures, and tables
- [ ] Basic documentation explaining how to reproduce the results

⚠️ If not every item is checked, it is not complete!
```

---

## Key Success Factors

```
CRITICAL SUCCESS FACTORS:

1. Accuracy:
   - Exactly matches the paper's specification (versions, parameters, settings)
   - Correctly translates equations into code
   - Uses hyperparameter values exactly

2. Completeness:
   - Implements every method discussed, not just the main contribution
   - Also implements the variants needed for ablation studies
   - Also implements what's needed for baseline comparisons

3. Functionality:
   - The code actually works and successfully runs the experiments
   - Can train/evaluate without errors
   - Can actually reproduce the paper's results
```

---

## Execution Guidelines

### Before Implementing Each File
1. Check the requirements for that file in the implementation plan
2. Confirm that the files it depends on have already been implemented
3. Reference the relevant equations/algorithms in the paper

### While Implementing Each File
1. Write complete import statements
2. Define the class/function structure
3. Translate the paper's equations/algorithms into code
4. Add proper docstrings
5. Add error handling

### After Implementing Each File
1. Check for syntax errors
2. Check that all imports resolve
3. Run a simple test if possible
4. **Move on to the next file immediately** (don't ask for permission)

---

## File Template

### Basic Python File Structure
```python
"""
[file description]

Paper: [paper title]
Section: [relevant section number]
"""

import ...

# Hyperparameters from the paper
PARAM_NAME = value  # source: Section X / Table Y


class ComponentName:
    """
    [component description]

    Implements Equation X from the paper:
    [equation]
    """

    def __init__(self, ...):
        ...

    def forward(self, ...):
        # implements Eq. X
        ...


def main():
    """Main entry point"""
    ...


if __name__ == "__main__":
    main()
```

---

## Final Verification

After completing the implementation:

1. **Run test**: does `python main.py` run without errors?
2. **Training test**: does training proceed with a small amount of data?
3. **Result check**: can the paper's main results be reproduced?
4. **Documentation check**: is the README.md clear about how to run it?

If every item passes, the implementation is complete!

---

## ⚠️ REMEMBER

```
The goal is to replicate the entire paper —
not a single part or a minimal example.

The file-reading tool is PAGINATED, so reading every
relevant part of the paper may require multiple calls.

If you find patterns in reference code, use them only as
inspiration, and always implement according to the paper's
original specification.
```
