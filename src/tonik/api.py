import logging
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from typing import Annotated, List, Union, Optional
from urllib.parse import unquote

import datashader as dsh
import numpy as np
import pandas as pd
import uvicorn
from cftime import date2num, num2pydate
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from . import get_data
from .storage import Storage

logger = logging.getLogger(__name__)

SubdirType = Annotated[Union[List[str], None], Query()]
InventoryReturnType = Union[list, dict]


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
        self.app.get("/labels")(self.labels)

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

    async def feature(self,
                      group: str,
                      name: str,
                      starttime: Union[str, None],
                      endtime: Union[str, None],
                      subdir: SubdirType = None,
                      resolution: str = 'full',
                      verticalres: int = 10,
                      log: bool = False,
                      normalise: bool = False):
        _st = self.preprocess_datetime(starttime)
        _et = self.preprocess_datetime(endtime)
        g = Storage(group, rootdir=self.rootdir,
                    starttime=_st, endtime=_et,
                    create=False)
        c = g
        if subdir:
            c = g.get_substore(*subdir)
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
            df['dates'] = pd.to_datetime(df.dates.values).strftime(
                '%Y-%m-%dT%H:%M:%SZ')
            output = df.to_csv(index=False,
                               columns=['dates', 'freqs', 'feature'])
        else:
            df = pd.DataFrame(data=feat.to_pandas(), columns=[feat.name])
            df['dates'] = df.index
            if resolution != 'full':
                try:
                    current_resolution = pd.Timedelta(
                        df['dates'].diff().mean())
                    if current_resolution < pd.Timedelta(resolution):
                        df = df.resample(pd.Timedelta(resolution)).median()
                except ValueError:
                    logger.warning(
                        f"Cannot resample {feat.name} to {resolution}: e")
            df.rename(columns={feat.name: 'feature'}, inplace=True)
            df['dates'] = df['dates'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
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
        dates = num2pydate(
            d, units='hours since 1970-01-01 00:00:00.0', calendar='gregorian')
        return freq, dates, spec

    async def inventory(self, group: str, subdir: SubdirType = None, tree: bool = True) -> InventoryReturnType:
        sg = Storage(group, rootdir=self.rootdir, create=False)
        try:
            c = sg.get_substore(*subdir)
        except TypeError:
            c = sg
        except FileNotFoundError:
            msg = "Directory {} not found.".format(
                '/'.join([sg.path] + subdir))
            raise HTTPException(status_code=404, detail=msg)
        if tree and not subdir:
            return sg.to_dict()
        else:
            dir_contents = os.listdir(c.path)
            if 'labels.json' in dir_contents:
                dir_contents.remove('labels.json')
            return [fn.replace('.nc', '').replace('.zarr', '') for fn in dir_contents]

    async def labels(self, group: str, subdir: SubdirType = None, starttime: Optional[str] = None, endtime: Optional[str] = None):
        _st = self.preprocess_datetime(starttime)
        _et = self.preprocess_datetime(endtime)
        sg = Storage(group, rootdir=self.rootdir,
                     starttime=_st, endtime=_et, create=False)
        try:
            c = sg.get_substore(*subdir)
        except TypeError:
            c = sg
        except FileNotFoundError:
            msg = "Directory {} not found.".format(
                '/'.join([sg.path] + subdir))
            raise HTTPException(status_code=404, detail=msg)
        return c.get_labels()


def main(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--rootdir", default='/tmp')
    parser.add_argument("-p", "--port", default=8003, type=int)
    parser.add_argument("--host", default='0.0.0.0')
    args = parser.parse_args(argv)
    ta = TonikAPI(args.rootdir)
    uvicorn.run(ta.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
