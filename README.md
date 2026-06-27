[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Visitor Badge](https://visitor-badge.laobi.icu/badge?page_id=Yingjie4Science.invest-mental-health)

# InVEST mental health model application to US cities 

## Directory structure
```
invest-mental-health/
  ├── code/
      ├── nature-exposure/ 
      ├── greening-targets/
      ├── 
  ├── data/ 
      ├── raw
      ├── intermediate
      ├── output
      ├── 
  ├── docs/
  ├── figures/

```

## Nature Exposure

## Urban Greening Targets (CLI Pipeline)

The full scraping + extraction pipeline is available at:

`pipelines/urban_greening_targets/`

This module contains all stages (A-G) and exports city targets to CSV/JSON/Markdown.

Quick start:

```bash
cd pipelines/urban_greening_targets
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
PYTHONPATH=src python -m urban_greening_targets.cli run --all
PYTHONPATH=src python -m urban_greening_targets.cli export
```

Outputs are written to:

`pipelines/urban_greening_targets/data/output/`


## Key References
- Li, Y., Mao, Y., Mandle, L., Rydström, A., Remme, R.P., Lan, X., Wu, T., Song, C., Lu, Y., Nadeau, K.C., Meyer-Lindenberg, A., Daily, G.C., Guerry, A.D., 2025. [Acute mental health benefits of urban nature](https://doi.org/10.1038/s44284-025-00286-y). ***Nature Cities*** 2, 720–731. https://doi.org/10.1038/s44284-025-00286-y
- Bratman, G.N., Anderson, C.B., Berman, M.G., Cochran, B., Vries, S. de, Flanders, J., Folke, C., Frumkin, H., Gross, J.J., Hartig, T., Kahn, P.H., Kuo, M., Lawler, J.J., Levin, P.S., Lindahl, T., Meyer-Lindenberg, A., Mitchell, R., Ouyang, Z., Roe, J., Scarlett, L., Smith, J.R., Bosch, M. van den, Wheeler, B.W., White, M.P., Zheng, H., Daily, G.C., 2019. [Nature and mental health: An ecosystem service perspective](https://advances.sciencemag.org/content/5/7/eaax0903). ***Science Advances*** 5, eaax0903. https://doi.org/10.1126/sciadv.aax0903
- Remme, R.P., Frumkin, H., Guerry, A.D., King, A.C., Mandle, L., Sarabu, C., Bratman, G.N., Giles-Corti, B., Hamel, P., Han, B., Hicks, J.L., James, P., Lawler, J.J., Lindahl, T., Liu, H., Lu, Y., Oosterbroek, B., Paudel, B., Sallis, J.F., Schipperijn, J., Sosič, R., Vries, S. de, Wheeler, B.W., Wood, S.A., Wu, T., Daily, G.C., 2021. [An ecosystem service perspective on urban nature, physical activity, and health](https://www.pnas.org/content/118/22/e2018472118). ***PNAS*** 118. https://doi.org/10.1073/pnas.2018472118


