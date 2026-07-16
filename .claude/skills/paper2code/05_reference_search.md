# Phase 0: Reference Code Search - Optional Step

## Purpose
Before implementing the paper, **find similar implementations** to improve implementation quality.
Reference code is for **inspiration** — the paper's original specification **always takes priority**.

---

## ⚠️ Key Principles (CRITICAL)

```
⚠️ REFERENCE CODE USAGE PRINCIPLES:

1. Reference code is for inspiration
2. The paper's original specification always takes priority
3. Understand and adapt — don't copy
4. Even patterns found in references must be adjusted to fit the paper's requirements
5. Always check the license

DO:
✓ Reference structure and patterns
✓ Learn implementation tricks and optimization techniques
✓ Identify common pitfalls
✓ Reference testing methodology

DON'T:
✗ Copy code verbatim
✗ Copy bugs from the reference implementation along with the code
✗ Follow a reference's design where it differs from the paper
✗ Violate licenses
```

---

## Search Protocol

### Step 1: Analyze the Paper's References

From the paper's References section, identify papers that are likely to have a GitHub repository:

```
High-priority references:
1. Papers cited in the methodology/implementation section
2. Papers mentioned with phrases like "We build upon..." or "We extend..."
3. Methods used as baselines
4. Earlier papers by the same authors

Exclude:
- The target paper's own official implementation (if it exists, just use that)
- Purely theoretical papers
- Unrelated background citations
```

### Step 2: Find Repositories via Web Search

Search queries using Claude's web search capability:

```
Search query patterns:

1. Direct search:
   - "[paper title] GitHub"
   - "[paper title] code repository"
   - "[author name] [paper title] implementation"

2. Algorithm-based search:
   - "[algorithm name] PyTorch implementation"
   - "[algorithm name] TensorFlow GitHub"
   - "[core method] code example"

3. Keyword combinations:
   - "[key term 1] [key term 2] GitHub stars:>100"
   - "[method name] official implementation"
   - "[dataset name] [method name] benchmark"

Search tips:
- Search both the paper's acronym and its full name
- Check the authors' GitHub profiles
- Search Papers With Code (paperswithcode.com)
```

### Step 3: Evaluate and Rank Quality

Evaluate the repositories found against the following criteria:

```yaml
evaluation_criteria:
  repository_quality:  # 40% weight
    - stars: "[>100: Good, >500: Excellent]"
    - recent_activity: "[commits within 6 months: Active]"
    - documentation: "[quality of README, docstrings]"
    - issues_resolved: "[issue response rate]"
    - tests: "[whether test code exists]"

  implementation_relevance:  # 30% weight
    - algorithm_match: "[whether the implemented algorithm matches the paper]"
    - completeness: "[full pipeline vs. partial implementation]"
    - paper_citation: "[whether it cites the paper]"

  technical_depth:  # 20% weight
    - code_quality: "[readability, degree of structure]"
    - performance: "[whether benchmark results exist]"
    - flexibility: "[configurability, extensibility]"

  academic_credibility:  # 10% weight
    - author_affiliation: "[authors' affiliation]"
    - official: "[whether it is the official implementation]"
    - peer_reviewed: "[whether it was peer-reviewed alongside the paper]"
```

### Step 4: Select and Analyze the Top 5

For each repository, record the following:

```yaml
selected_references:
  - rank: 1
    title: "[paper/repository title]"
    repository_url: "[GitHub URL]"
    relevance_score: 0.95  # 0-1 scale

    key_contributions:
      - "[thing 1 that can be learned from this repository]"
      - "[thing 2 that can be learned from this repository]"

    implementation_value: |
      [detailed explanation of how this helps the implementation]

    usage_suggestion: |
      [which parts to reference, and how to apply them]

    caveats:
      - "[something to watch out for — where it differs from the paper]"
      - "[license restrictions]"
```

---

## Output Format

```yaml
reference_search_results:
  search_summary:
    total_found: "[number of related repositories found]"
    evaluated: "[number of repositories evaluated]"
    selected: 5

  official_implementation:
    exists: true/false
    url: "[URL, if it exists]"
    note: "[if an official implementation exists, prefer using that]"

  selected_references:
    - rank: 1
      title: "..."
      repository_url: "..."
      relevance_score: 0.95
      key_contributions: [...]
      implementation_value: "..."
      usage_suggestion: "..."
      caveats: [...]

    - rank: 2
      # ... same structure

    # ... rank 3, 4, 5

  search_queries_used:
    - "[search query 1 used]"
    - "[search query 2 used]"

  papers_with_code_link: "[URL of the paper's PWC page]"
```

---

## Usage Guide

### When to Perform This Step

```
Recommended when:
✓ Implementing a complex algorithm
✓ The paper lacks implementation detail
✓ A specific framework's (PyTorch, TensorFlow) implementation patterns are needed
✓ Performance optimization tips are needed

Can be skipped when:
- The algorithm is very simple
- The paper already has a detailed implementation description
- You already have experience with a similar implementation
- Time is limited
```

### How to Use References

```
1. Reference the structure:
   - file organization approach
   - class/function separation patterns
   - config management approach

2. Learn implementation tricks:
   - numerical stability handling
   - memory optimization
   - parallelization techniques

3. Testing methodology:
   - unit test structure
   - integration test scenarios
   - benchmark scripts

4. Identify caveats:
   - common bug patterns
   - performance bottlenecks
   - environment compatibility issues
```

---

## ⚠️ Self-Check: Confirming Reference Search Completion

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ REFERENCE SEARCH CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

□ Checked whether an official implementation exists?     → YES / NO
□ Tried at least 3 search queries?                        → YES / NO
□ Checked Papers With Code?                                → YES / NO
□ Evaluated the quality of the repositories found?         → YES / NO
□ Completed detailed analysis of the top 5?                → YES / NO
□ Checked the license of each reference?                   → YES / NO
□ Recorded differences (caveats) from the paper?           → YES / NO

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Notes

```
⚠️ REMEMBER:

Reference code is supplementary material only.

The final implementation must always follow the paper's specification.
If you find a place where reference code differs from the paper,
follow the paper's specification.

Bugs in reference code, or places where it disagrees with the paper,
must not be carried over into our implementation.
```
