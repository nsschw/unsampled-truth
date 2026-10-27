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
@inproceedings{schwager-etal-2026-unsampled,
    title = "The Unsampled Truth: Quantifying Prompt Artifacts in {LM} Psychometrics",
    author = {Schwager, Nils  and
      Hau, Christoph  and
      M{\"u}nker, Simon  and
      Rettinger, Achim},
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.1009/",
    pages = "21330--21341",
    ISBN = "979-8-89176-462-0",
}
```

## Acknowledgements

- **Data:** This repo contain data from [Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) (Meyer & Corneil, 2025) for the baseline personas. Due to copyright limitations we don't include the Big Five Inventory and the Short Dark Triad items, as well as their paraphrases from the [PERSIST](https://github.com/tosatot/PERSIST) paper.

- **Funding:** This work was supported by EMOSCI (DFG, project 569156112) and KuKI (BMFTR, project 16KIS2515).

- **EVAS:** This work was conducted within the [EVAS Research Cooperative](https://evas-research.github.io/)
