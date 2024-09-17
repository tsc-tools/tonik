from argparse import ArgumentParser
from datetime import timedelta, datetime, timezone
import logging
import os
from urllib.parse import unquote

from cftime import num2date, date2num
import datashader as dsh
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Annotated

from .storage import StorageGroup
from . import get_data

logger = logging.getLogger(__name__)


class TonikAPI:

    def __init__(self, rootdir) -> None:
        self.rootdir = rootdir
        self.app = FastAPI()

        # -- allow any origin to query API
        self.app.add_middleware(CORSMiddleware,
                                allow_origins=["*"])

        self.app.get("/", response_class=HTMLResponse)(self.root)
        self.app.get("/feature")(self.feature)
        self.app.get("/inventory")(self.inventory)

    def root(self):
        with open(get_data("package_data/index.html"), "r", encoding="utf-8") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content, status_code=200)

    def preprocess_datetime(self, dt):
        """
        Convert datetime string to datetime object.
        """
        # remove timezone info
        dt = dt.split('+')[0]
        # remove 'Z' at the end
        dt = dt.replace('Z', '')
        # convert html encoded characters
        dt = unquote(dt)
        dt = datetime.fromisoformat(dt)
        dt = dt.replace(tzinfo=None)
        return dt

    def feature(self,
                group: str,
                name: str,
                starttime: str = None,
                endtime: str = None,
                resolution: str = 'full',
                verticalres: int = 10,
                log: bool = False,
                normalise: bool = False,
                subdir: Annotated[list[str] | None, Query()] = None):
        _st = self.preprocess_datetime(starttime)
        _et = self.preprocess_datetime(endtime)
        g = StorageGroup(group, rootdir=self.rootdir,
                         starttime=_st, endtime=_et)
        if subdir is None:
            c = g
        else:
            c = g.get_store(*subdir)
        try:
            feat = c(name)
        except ValueError as e:
            msg = f"Feature {name} not found in directory {c.path}:"
            msg += f"{e}"
            raise HTTPException(status_code=404, detail=msg)
        if len(feat.shape) > 1:
            # assume first dimension is frequency
            nfreqs = feat.shape[0]
            dates = feat.coords[feat.dims[1]].values
            if resolution != 'full':
                freq, dates, spec = self.aggregate_feature(
                    resolution, verticalres, feat, nfreqs, dates)
            else:
                spec = feat.values
                freq = feat.coords[feat.dims[0]].values
            vals = spec.ravel(order='C')
            if log and feat.name != 'sonogram':
                vals = 10*np.log10(vals)
            if normalise:
                vals = (vals - np.nanmin(vals)) / \
                    (np.nanmax(vals) - np.nanmin(vals))
            freqs = freq.repeat(dates.size)
            dates = np.tile(dates, freq.size)
            df = pd.DataFrame(
                {'dates': dates, 'freqs': freqs, 'feature': vals})
            output = df.to_csv(index=False,
                               columns=['dates', 'freqs', 'feature'])
        else:
            df = pd.DataFrame(data=feat.to_pandas(), columns=[feat.name])
            df['dates'] = df.index
            try:
                current_resolution = pd.Timedelta(df['dates'].diff().mean())
                if current_resolution < pd.Timedelta(resolution):
                    df = df.resample(pd.Timedelta(resolution)).mean()
            except ValueError as e:
                logger.warning(
                    f"Cannot resample {feat.name} to {resolution}: e")
            df.rename(columns={feat.name: 'feature'}, inplace=True)
            output = df.to_csv(index=False, columns=['dates', 'feature'])
        return StreamingResponse(iter([output]),
                                 media_type='text/csv',
                                 headers={"Content-Disposition":
                                          "attachment;filename=<tonik_feature>.csv",
                                          'Content-Length': str(len(output))})

    def aggregate_feature(self, resolution, verticalres, feat, nfreqs, dates):
        resolution = np.timedelta64(
            pd.Timedelta(resolution), 'ms').astype(float)
        ndays = np.timedelta64(dates[-1] - dates[0], 'ms').astype(float)
        canvas_x = int(ndays/resolution)
        canvas_y = min(nfreqs, verticalres)
        dates = date2num(dates.astype('datetime64[us]').astype(datetime),
                         units='hours since 1970-01-01 00:00:00.0',
                         calendar='gregorian')
        feat = feat.assign_coords({'datetime': dates})
        cvs = dsh.Canvas(plot_width=canvas_x,
                         plot_height=canvas_y)
        agg = cvs.raster(source=feat)
        freq_dim = feat.dims[0]
        freq, d, spec = agg.coords[freq_dim].values, agg.coords['datetime'].values, agg.data
        dates = num2date(
            d, units='hours since 1970-01-01 00:00:00.0', calendar='gregorian')
        return freq, dates, spec

    def inventory(self, group: str) -> dict:
        sg = StorageGroup(group, rootdir=self.rootdir)
        return sg.to_dict()

# ta = TonikAPI('/tmp').feature()


def main(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--rootdir", default='/tmp')
    args = parser.parse_args(argv)
    ta = TonikAPI(args.rootdir)
    uvicorn.run(ta.app, host="0.0.0.0", port=8003)


if __name__ == "__main__":
    main()
