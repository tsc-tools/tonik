# Tonik

Tonik provides you with a solution to store and retrieve scientific time-series data as well as serving it through an API.
For visualisations, the API can serve large requests very quickly by downsampling the data to the requested resolution on demand. The API was optimised to visualise time-series and data labels with [Grafana](https://grafana.com/oss/grafana/).

## Requirements
* h5py
* datashader
* xarray
* pandas
* netcdf4
* h5netcdf
* python-json-logger
* uvicorn
* fastapi
* matplotlib (only needed to reproduce the examples in the user guide)

## Installation
```
pip install -U tonik
```

## Documentation

Learn more about tonik in its official [documentation](https://tsc-tools.github.io/tonik)

## Get in touch

Report bugs, suggest features, view the source code, and ask questions [on GitHub](https://github.com/tsc-tools/tonik/issues).