# Preprocessing
Data source
- Census tract polygon: PyGris package
- Flow data: Cuebiq flow data of Feb 2020
- Protected attribute - PCI: [api.census.gov](https://api.census.gov)
- Other features: OpenStreetMap data

Processed data will be stored at [data/processed](data/processed) with suffix of `_feat.parquet`

# Modeling
Enviornment: `python3.8`

Run command: 
```[bash]
python main.py --model [model name] --coef [lagrangian coefficient] --device [cuda] --lr [0.0001] --place [place] --batch-size 256 --epochs [500] --op [binary or reg]
# default coef is 0 for other models except fairmb
# for fairmb, coef ranges from 0.1 to 1.0 with step 0.1
```
Check checkpoint at [checkpoints](/checkpoints)

Check test result at [test result](/outputs)

# Baseline models references
DG: https://github.com/scikit-mobility/DeepGravity

GMEL: https://github.com/jackmiemie/GMEL

RT: https://github.com/jadore801120/attention-is-all-you-need-pytorch

RM: https://github.com/scikit-mobility/scikit-mobility/blob/master/skmob/models/radiation.py
