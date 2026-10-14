# The Unsampled Truth: Quantifying Prompt Artifacts in LM Psychometrics

Code for the paper [The Unsampled Truth: Quantifying Prompt Artifacts in LM Psychometrics](https://arxiv.org/abs/2606.03357) (Schwager, Hau, Münker & Rettinger, 2026).

We cross five baseline personas with five semantically equivalent variants of four prompt components (persona wording, task instruction, item wording, option symbol), measure response displacement with the 1-Wasserstein distance (APWD) and partition the variation among the components to quantify prompt artifacts.

## Setup

```bash
uv sync
```

The questionnaires are **not included** for licensing reasons. Place them in `data/questionnaires/` using this format (`<item>_<item wording>` IDs; `notebooks/00-transformations/` builds them from the original items and the Tosato et al. paraphrases):

```json
{"items": [{"item_id": "BFI0_V0", "text": "Is talkative.",
            "semantic_anchors": ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]}]}
```

## Usage

```bash
uv run python scripts/run_inference.py --config configs/antonia.yaml   # one baseline persona × one model
uv run python scripts/run_qwens.py                                     # all baseline personas × a model family
```

Results are written to `outputs/raw_results/<baseline persona>_<model>_raw.jsonl`. The analysis in `notebooks/01-analysis/` follows the paper:

| File | Paper |
|---|---|
| `4.1-total-displacement.ipynb` | §4.1, Tables 1 & 3 |
| `4.2-marginal-displacement.ipynb` | §4.2, Figures 2 & 4 |
| `4.3a-variation-partitioning.py` → `4.3b-variation-partitioning.ipynb` | §4.3, Figures 3 & 5 |

## Citation

```bibtex
@misc{schwager2026unsampled,
  title         = {The Unsampled Truth: Quantifying Prompt Artifacts in LM Psychometrics},
  author        = {Schwager, Nils and Hau, Christoph and M{\"u}nker, Simon and Rettinger, Achim},
  year          = {2026},
  eprint        = {2606.03357},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2606.03357}
}
```

## Acknowledgements

- **Data:** [Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) (Meyer & Corneil, 2025) for the baseline personas; the Big Five Inventory (John & Srivastava, 1999) and the Short Dark Triad (Jones & Paulhus, 2014); item paraphrases from Tosato et al. (2026); option symbol sets following Yang et al. (2025). Task instructions are adapted from Tosato et al. (2026), La Cava & Tagarelli (2025), Lee et al. (2025) and Zou et al. (2025).
- **Software:** [PyTorch](https://pytorch.org), [Transformers](https://github.com/huggingface/transformers), [NumPy](https://numpy.org), [SciPy](https://scipy.org), [pandas](https://pandas.pydata.org), [matplotlib](https://matplotlib.org) and [seaborn](https://seaborn.pydata.org).
- **Funding:** This work was conducted within the [EVAS Research Cooperative](https://evas-research.github.io/) and supported by EMOSCI (DFG, project 569156112) and KuKI (BMFTR, project 16KIS2515).
