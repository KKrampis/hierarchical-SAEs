# Paper2Code Skill

A Claude Code Skill for converting research papers into executable code.

## Files

| File | Description |
|------|-------------|
| `SKILL.md` | Main entry point - orchestrates the entire pipeline |
| `01_algorithm_extraction.md` | Phase 1: Extract algorithms, equations, pseudocode |
| `02_concept_analysis.md` | Phase 2: Analyze paper structure and components |
| `03_code_planning.md` | Phase 3: Create detailed implementation plan |
| `04_implementation_guide.md` | Phase 4: File-by-file code generation |
| `05_reference_search.md` | Phase 0 (Optional): Find reference implementations |
| `06_memory_management.md` | Context management for long papers |

## Installation

Copy this folder to your Claude skills directory:

```bash
cp -r paper2code ~/.claude/skills/
```

## Usage

The skill activates automatically when you request paper implementation:

- `"implement this paper"`
- `"paper2code"`
- `"convert this paper into code"`

## Pipeline

```
Phase 0 (Optional) → Phase 1 → Phase 2 → Phase 3 → Phase 4
   Reference         Algorithm   Concept    Code      Code
   Search            Extraction  Analysis   Planning  Implementation
```

Source: translated and imported from [KKrampis/paper2code-skill-english](https://github.com/KKrampis/paper2code-skill-english).
