# Submission Metadata
- team_id: GT-MSA-Spring-2026-Team-25
- repo_url: https://github.com/dlgoebel/bitcoin-analytics-capstone-template/
- eda_commit_sha: 4bdc20718e71
- context_classification: bitcoin

# Submission Synopsis
- synopsis: This is one of the strongest Bitcoin EDA submissions in the set. The executive notebook makes a clear, defensible argument that MVRV is the dominant accumulation-relevant on-chain signal while Polymarket has not yet shown incremental value beyond on-chain data, and it treats that negative result as analytically meaningful rather than as a failure. The technical notebook is substantial, well organized, and includes explicit data-integrity checks, stationarity testing, Granger causality, mutual information analysis, and a broad signal-comparison framework. The package is highly aligned with the assignment and already reads like a serious transition document into model-development work.
- primary_findings:
  - MVRV is presented as the most robust and regime-stable accumulation signal across multiple statistical methods.
  - Several other on-chain variables show weaker or more context-dependent value, and some apparent signals are better interpreted as non-linear or unstable effects.
  - Polymarket analysis does not yet justify inclusion as a primary signal, but the notebook frames that negative result rigorously and uses it to narrow the model-development search space.

# Rubric Scores
| criterion_id | level_0_to_4 | weight | weighted_points | evidence_note |
| --- | ---: | ---: | ---: | --- |
| executive_summary | 4 | 10 | 10.0 | `eda/EDA_Executive.ipynb` is extremely efficient and clearly states the strongest findings, methods, limitations, and modeling implications. |
| narrative_readability | 4 | 10 | 10.0 | Both notebooks are carefully structured and readable, with a strong separation between executive conclusions and technical validation. |
| data_provenance_reproducibility | 3 | 15 | 11.25 | The repository documents data sources and cleaning decisions well, though rerunning the full analysis still depends on external data retrieval beyond the repo snapshot. |
| data_quality_integrity | 4 | 10 | 10.0 | `eda/EDA.ipynb` includes explicit missingness review, dropped-column rationale, range-aware handling of early sparse Bitcoin data, and other integrity checks. |
| analytical_depth | 4 | 15 | 15.0 | The technical notebook is unusually deep, covering signal audits, stationarity, differencing, Granger tests, mutual information, and multiple comparison lenses. |
| statistical_reasoning | 4 | 10 | 10.0 | The submission uses several complementary statistical approaches and is careful about distinguishing robust signals from weaker or non-linear candidates. |
| visualization_quality | 3 | 10 | 7.5 | The notebook uses well-integrated tables and plots to support the argument, though the overall package emphasizes analytical rigor more than visual polish. |
| assignment_fit_and_next_step_motivation | 4 | 20 | 20.0 | The work is tightly aligned with Bitcoin accumulation strategy design and directly motivates a focused next phase around MVRV-centered model development. |

# Strengths
- The executive notebook is unusually strong and gives a fast, defensible read on the actual strategic conclusion.
- The technical notebook shows real statistical discipline rather than relying on a single correlation-based story.
- The submission handles a negative Polymarket result correctly by turning it into a narrower, better-motivated modeling direction.

# Highest-Priority Improvements
- Add a short rerun checklist at the top of the technical notebook to remove the remaining reproducibility ambiguity around external data access.
- Keep validating whether MVRV’s apparent dominance is partially circular because it includes price in the numerator, and document that caveat prominently in the next phase.
- If time permits, add one compact executive visual that summarizes the full signal-ranking hierarchy at a glance.

# Recommended Next Step
- Build the first production candidate around MVRV as the primary allocation driver, then test whether any secondary non-linear or regime-specific features add stable value without weakening the model’s interpretability.

# Overall Summary
- This is an excellent Bitcoin EDA submission with strong executive communication, rigorous technical validation, and clear assignment-specific direction. Total score: 93.75/100.
