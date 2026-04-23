# The Limits of Black Box Forecasting in Chaotic Systems with Changing Parameters and UPO-Predictor
**Source code for the paper: An Empirical Study on The Limits of Black Box Forecasting in Chaotic Systems with Changing Parameters**

**Due to privacy concers, the source code files will be uploaded once the paper is accepted or by request**


## Description

- This repository presents an experimental framework for studying the predictability of the Lorenz chaotic system using Standard and State-of-the-art approaches.
    - LSTMs (seq-to-seq and generative), Transformers, Time Fusion Transformers, UPO Predictor.
- The work explores how forecasting performance varies across different dynamic regimes caused by changes in system parameters and noise. 
- For each parameter configuration and forecast horizon, multiple metrics are computed to evaluate long-, mid-, and short-term predictability.
- An extensive sensitive analysis across multiple system regimes determined by RHO values.
- A novel UPO-based Predictor is proposed and evaluated.
  

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

## Dataset link

The datasets used for the experimental assesment are available on This site was built using [HARVARD Dataverse](https://doi.org/10.7910/DVN/LBCPMC).



## Setup and Running

```bash
pip install -r requirements.txt
```
- Run the jupyter notebook
- Most of the scripts are in the modules directory


## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## Code Authors

**Roland Bolboaca**  
