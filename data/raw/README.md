# Raw Dataset

This directory is reserved for local copies of the NASA C-MAPSS dataset used by the benchmark.

Raw data are intentionally excluded from the repository.

For the current FD001 benchmark, the expected files are:

- `FD001/train_FD001.txt`
- `FD001/test_FD001.txt`
- `FD001/RUL_FD001.txt`

After obtaining the C-MAPSS archive or an extracted dataset directory, prepare the local files with:

```text
python scripts/prepare_fd001.py <path-to-cmapss-archive-or-directory>
```

The preparation script checks that the three required files are present and have the expected basic column structure before copying them into `data/raw/FD001/`.

Do not commit the raw C-MAPSS files to Git.
