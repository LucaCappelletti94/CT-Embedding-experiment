# CT-Embedding-experiment

Evaluating CT embedding models

## Executing the correlation tests

To re-compute the correlation tests for all the CSVs in the `scores` directory,
first install the required dependencies, which require `Python 3.10` or later:

```bash
pip install -r requirements.txt
```

next, run the following command:

```bash
python correlation_tests.py
```

Do note that the number of labelled images is quite minimal, so any results should be interpreted with a substantial grain of salt.
