# Exploratory Data Analysis Outline

## 📦 Deliverable: Exploratory Data Analysis Package

Each team will submit an EDA package inside this official capstone template repository. The goal is to communicate meaningful insights through a clear narrative that serves both technical and non-technical audiences.

All EDA work must live inside the `EDA` folder of your forked template repository.

---

## ✅ Submission Instructions

- All work must be completed inside your fork of the official template repo
  [https://github.com/TrilemmaFoundation/bitcoin-analytics-capstone-template](https://github.com/TrilemmaFoundation/bitcoin-analytics-capstone-template)

- Place all EDA related notebooks inside the `EDA` folder

- Your submission will consist of **two primary notebooks**

### 1. Required: `EDA_Executive.ipynb`

This is the notebook we will read first.

It should be a polished, narrative driven summary of your most engaging insights.

- Focus on storytelling and interpretation
- Highlight only your strongest findings
- Reference deeper work stored in other notebooks when appropriate

### 2. Required: `EDA.ipynb`

This notebook serves as your comprehensive EDA reference.

- Include all important exploratory work
- Show intermediate analysis, experiments, and supporting results
- Use this file as a technical appendix that supports `EDA_Executive.ipynb`

Push your work to your forked GitHub repository by the announced deadline. The latest commit before the deadline is what will be viewed for feedback.

---

## 📋 Notebook Structure

### 1. Executive Summary

Begin `EDA_Executive.ipynb` with an executive bullet point summary of your most novel and non-trivial findings.

A reader should understand your key conclusions in under one minute.

### 2. Narrative Driven Analysis

Organize your notebook as a clear story.

- Use descriptive section headers
- Explain your reasoning and decisions
- Connect observations to broader themes

Avoid presenting isolated plots without interpretation.

### 3. Data Retrieval

Load data using the official scripts and utilities provided in the template repository.

Clearly state:

- What data sources you used
- Any preprocessing steps
- Assumptions or limitations

### 4. General Dataset Overview

Provide a high level understanding of the dataset.

Include:

- Data integrity checks
- Data types and ranges
- Missingness and completeness
- Descriptive statistics
- Initial exploratory visualizations

### 5. Focused Exploration

Select specific features or themes for deeper investigation.

For each focus area:

- Justify why it is interesting
- Perform statistical and visual analysis
- Summarize what you learned

Your exploration should lead toward one of two conclusions:

- There appears to be signal worth modeling
- No meaningful signal was found but the investigation was informative

### 6. Insight Showcase

Demonstrate your strongest analytical thinking.

Examples include:

- Rigorous statistical tests
- Creative or informative visualizations
- Novel metrics or transformations
- Thoughtful comparisons across time or regimes

Reference supporting work in `EDA.ipynb` when needed.

---

## 📊 Evaluation Rubric

| Category              | Description                                          |
| --------------------- | ---------------------------------------------------- |
| Readability           | Can a layperson follow the high level narrative      |
| Technical Depth       | Can another data scientist build on your work        |
| Visualization Quality | Are plots clean, labeled, and informative            |
| Statistical Rigor     | Do you investigate deeper structure in the data      |
| Executive Summary     | Is your summary concise and insightful               |
| Engagement            | Is the notebook interesting to explore               |
| Reproducibility       | Can the notebook run top to bottom without confusion |

---

## 🚀 Success Tips

- Write with clarity and intention
- Use strong titles and captions
- Prioritize insight over code quantity
- Assume your reader has five minutes to review your work
- Link to additional notebooks in the `EDA` folder for deeper dives

Example reference:

See `EDA/volatility_deep_dive.ipynb` for extended analysis.

Your goal is to produce an EDA that is both technically rigorous and compelling to read.
