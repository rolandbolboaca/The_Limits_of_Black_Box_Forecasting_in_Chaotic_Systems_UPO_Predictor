# The Limits of Black Box Forecasting in Chaotic Systems with Changing Parameters and UPO-Predictor
Source code for the paper: An Empirical Study on The Limits of Black Box Forecasting in Chaotic Systems with Changing Parameters [SUBMITED AND UNDER REVIEW]

## ***Due to privacy concerns, source code files will be uploaded once the paper is accepted or provided upon request.***


## Description

This repository presents an experimental framework for studying the predictability of the Lorenz chaotic system using Standard and State-of-the-art approaches:
  - Focused on both Standard Trajectory Forecasting and Dynamical Systems Reconstruction.
  - Comprehensive evaluation of standard and state-of-the-art methods for black-box chaotic forecasting, focusing on the Lorenz system with multiple operating regimes under short- and long-term horizons, noisy conditions, together with trajectory and attractor reconstruction.
  - Proposal of a novel UPO-based forecasting framework with fast inference and competitive performance in trajectory prediction and attractor reconstruction.
  - Systematic analysis of unstable periodic orbits, including shadowing times, selection frequencies, and the impact of UPO availability on forecasting performance.
  - Identification of important limitations of existing machine learning approaches for chaotic systems prediction and reconstruction.
  - Models: LSTM-A (seq-to-seq), LSTM-B (generative long term), Standard Transformer, Time Fusion Transformer, zero-shot/pretrained (DynaMix, Chronos, Panda, TimesFM), proposed UPO Predictor.
  - Evaluation on the Lorenz System.

## Citation

If you use the code or dataset, please cite the paper:

```bibtex
@Article{rolandb_lorenzlstmeval,
  AUTHOR = {Bolboacă, Roland, Haller, Piroska},
  TITLE = {An Empirical Study on The Limits of Black Box Forecasting in Chaotic Systems with Changing Parameters},
  JOURNAL = {},
  VOLUME = {},
  YEAR = {2026},
  NUMBER = {},
  ARTICLE-NUMBER = {},
  URL = {},
  ISSN = {},
  DOI = {}
}
```

## Relevant Papers

### Foundation and Standard Models for Time Series
- **Chronos: Learning the Language of Time Series Forecasting** — Ansari et al.  
  https://arxiv.org/abs/2403.07815

- **DynaMix: True Zero-Shot Inference of Dynamical Systems Preserving Long-Term Statistics** - Christoph Jürgen Hemmer and Daniel Durstewitz
  https://arxiv.org/abs/2505.13192
  
- **TimesFM: A Foundation Model for Time Series Forecasting** — Das et al. (Google Research)  
  https://arxiv.org/abs/2310.10688

- **Temporal Fusion Transformers for Interpretable Multi-horizon Forecasting** — Bryan Lim, Sercan O. Arik, Nicolas Loeff, Tomas Pfister  
  https://arxiv.org/abs/1912.09363

- **Long Short-Term Memory (LSTM)** — Sepp Hochreiter and Jürgen Schmidhuber  
  https://arxiv.org/abs/1402.1128

### Unstable Periodic Orbits (UPOs) / Dynamical Systems
- **Unstable Periodic Orbits in the Lorenz System** — Chiara Cecilia Maiocchi and Valerio Lucarini and Andrey Gritsun
  https://arxiv.org/pdf/2108.04181

- **Unstable Periodic Orbits (PhD Thesis)** — Chiara Cecilia Maiocchi  
  https://centaur.reading.ac.uk/112633/1/Maiocchi_thesis.pdf
  
  
## Dataset link

The datasets used for the experimental assesment are available on [HARVARD Dataverse](https://doi.org/10.7910/DVN/LBCPMC).

## Source code for models and UPO extraction

Additional source code for the pretrained models and the utilized UPO extraction software is available at:
- [DynaMix](https://github.com/DurstewitzLab/DynaMix-python)
- [Chronos](https://github.com/amazon-science/chronos-forecasting)
- [Panda](https://github.com/abao1999/panda)
- [TimesFM](https://github.com/google-research/timesfm)
- [UPO Extraction C++ Software](https://websites.umich.edu/~divakar/lorenz/index.html)

## Setup and Running

```bash
pip install -r requirements.txt
```
## Project Structure

Most scripts are located in the **`modules/`** directory:

- `model.py` – LSTM implementations.
- `metrics.py` – Standard evaluation metrics, including MSE, SBD, etc.
- `metrics_dyna.py` – Metrics from the *DynaMix* paper (`Dstsp`, `Dh`) for attractor geometric reconstruction.
- `data.py` – Data loading and preprocessing utilities.

## Dataset

Datasets should be placed in **`data/`**.

Train/test filenames can be configured in `data.py`.

Example files:
- `Norm_IC_01_G_Rho_5_1_225_005_Sim_10_29_01_2026_11_32_41.csv`
- `Norm_IC_01_G_Rho_5_1_225_005_Sim_10_29_01_2026_11_32_44.csv`

## UPO Data

Extracted UPOs should be stored in **`data_upos_original/`**.

Structure:
- One directory per rho value: `DATA1_{RHO}`  
  Example: `DATA1_28`

The UPO notebook cells will:
1. Load UPO files from these directories.
2. Generate `.pkl` files for easier reuse.

Expected UPO file format:
- `orbitX.dat`, where `X` ranges from `1` to the maximum extracted UPO index.

These files should be stored inside their corresponding `DATA1_X` directories.

## Running

Run the Jupyter notebook to start experiments.
  
## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## Code Authors

**Roland Bolboaca**  
