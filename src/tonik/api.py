from datetime import timedelta, datetime
import logging
import os

from cftime import num2date, date2num
import datashader as dsh
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List

from .storage import StorageGroup
from . import get_data

logger = logging.getLogger(__name__)

ROOTDIR = os.environ.get("ROOTDIR", default="/tmp")
                          
## -- API -- ##
app = FastAPI()

# -- allow any origin to query API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
)

# -- Home page gives instruction to build query
@app.get("/", response_class=HTMLResponse)
async def root():
    with open(get_data("package_data/index.html"), "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/feature")
def feature(name: str='rsam',
            group: str='Ruapehu',
            site: str='MAVZ',
            sensor: str='10',
            channel: str='HHZ',
            starttime: datetime=datetime.utcnow()-timedelta(days=30),
            endtime: datetime=datetime.utcnow(),
            resolution: str='full',
            verticalres: int=10,
            log: bool=True,
            normalise: bool=False):
    
    _st = datetime.fromisoformat(str(starttime))
    _st = _st.replace(tzinfo=None)
    _et = datetime.fromisoformat(str(endtime))
    _et = _et.replace(tzinfo=None)
    g = StorageGroup(group, rootdir=ROOTDIR,
                    starttime=_st, endtime=_et)
    c = g.channel(site=site, sensor=sensor, channel=channel)
    try:
        feat = c(name)
    except ValueError as e:
        msg = f"Feature {name} not found in directory {l.sitedir}:"
        msg += f"{e}"
        raise HTTPException(status_code=404, detail=msg)
    if len(feat.shape) > 1:
         # assume first dimension is frequency
        nfreqs = feat.shape[0]
        dates = feat.coords[feat.dims[1]].values
        if resolution != 'full':
            freq, dates, spec = aggregate_feature(resolution, verticalres, feat, nfreqs, dates)
        else:
            spec = feat.values
            freq = feat.coords[feat.dims[0]].values
        vals = spec.ravel(order='C')
        if log and feat.name != 'sonogram':
            vals = 10*np.log10(vals)
        if normalise:
            vals = (vals - np.nanmin(vals))/(np.nanmax(vals) - np.nanmin(vals))
        freqs = freq.repeat(dates.size)
        dates = np.tile(dates, freq.size)
        df = pd.DataFrame({'dates': dates, 'freqs': freqs, 'feature': vals})
        output = df.to_csv(index=False,
                           columns=['dates', 'freqs', 'feature'])
    else:
        df = pd.DataFrame(data=feat.to_pandas(), columns=[feat.name])
        df['dates'] = df.index
        try:
            df = df.resample(str(float(resolution)/60000.0)+'T').mean()
        except ValueError as e:
            logger.warning(f"Cannot resample {feat.name} to {resolution}: e")
        df.rename(columns={feat.name: 'feature'}, inplace=True)
        output = df.to_csv(index=False, columns=['dates', 'feature'])
    return StreamingResponse(iter([output]),
                             media_type='text/csv',
                             headers={"Content-Disposition":
                                      "attachment;filename=<VUMT_feature>.csv",
                                      'Content-Length': str(len(output))})


def aggregate_feature(resolution, verticalres, feat, nfreqs, dates):
    resolution = np.timedelta64(pd.Timedelta(resolution), 'ms').astype(float)
    ndays = np.timedelta64(dates[-1] - dates[0], 'ms').astype(float)
    canvas_x =  int(ndays/resolution)
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
    dates = num2date(d, units='hours since 1970-01-01 00:00:00.0', calendar='gregorian')
    return freq,dates,spec


#pydanticmodel output: Json file
class Feature(BaseModel):
    name: list


class Channel(BaseModel):
    name: str
    features: List[Feature] = []


class Location(BaseModel):
    name: str
    channels: List[Channel] = [] 


class Station(BaseModel):
    name: str
    lat: float
    lon: float
    locations: List[Location] = []


class Group(BaseModel):
    volcano: str
    stations: List[Station] = []


def get_pydanticModel(group, station, location, channel, feature_list):

    channels_data = {"name": channel, "features": feature_list}
    channel_models = []
    channel_model = Channel(**channels_data)
    channel_models.append(channel_model)

    location_data = {"name": location, "channels": channel_models}
    location_models = []
    location_model = Location(**location_data)
    location_models.append(location_model)

    stations_data = {"name": station, "lat": "42", "lon": "171",
                     "locations": location_models}
    station_models = []
    station_model = Station(**stations_data)
    station_models.append(station_model)

    group_model = Group(group=group, stations=station_models)

    # Exporting to JSON
    json_data = group_model.json()
    return json_data


# write a function that scans LOCKERROOMROOT for 
# available groups, stations, locations, channels, and features
# and returns a pydantic model
# def get_available_features():
#     groups = os.listdir(ROOT)
#     group_models = []
#     for group in groups:
#         stations = os.listdir(os.path.join(LOCKERROOMROOT, group))
#         station_models = []
#         for station in stations:
#             locations = os.listdir(os.path.join(LOCKERROOMROOT, group, station))
#             location_models = []
#             for location in locations:
#                 channels = os.listdir(os.path.join(LOCKERROOMROOT, group, station, location))
#                 channel_models = []
#                 for channel in channels:
#                     features = os.listdir(os.path.join(LOCKERROOMROOT, group, station, location, channel))
#                     feature_list = []
#                     for feature in features:
#                         feature_list.append(feature)
#                     channel_data = {"name": channel, "features": feature_list}
#                     channel_model = Channel(**channel_data)
#                     channel_models.append(channel_model)
#                 location_data = {"name": location, "channels": channel_models}
#                 location_model = Location(**location_data)
#                 location_models.append(location_model)
#             station_data = {"name": station, "lat": "42", "lon": "171", "locations": location_models}
#             station_model = Station(**station_data)
#             station_models.append(station_model)
#         group_data = {"volcano": group, "stations": station_models}
#         group_model = Group(**group_data)
#         group_models.append(group_model)
#     return group_models

# @app.get("/featureEndpoint")
# def featureEndpoint(group: str="all", station: str="all", channel: str="all",
#                     type: str="all"):
#     groups = vm.get_available_volcanoes()

#     station_model_list = []
#     channel_model_list = []
#     volcano_model_list = []
#     for _volcano in volcanoes:
#         streams = vm.get_available_streams(_volcano) 
#         for _stream in streams:
#             _, _station, _, _channel = _stream.split('.') 
#             stream_dir = os.path.join(FEATUREDIR, _volcano, _station, _channel)
#             try:
#                 feature_list = os.listdir(stream_dir)
#             except (NotADirectoryError, FileNotFoundError):
#                 continue
#             feature_list = sorted([str(os.path.basename(path)).split('.nc')[0] for path in feature_list])
#             channels_data = {"name": _channel, "features":feature_list}
#             channel_model = Channel(**channels_data)
#             channel_model_list.append(channel_model)
#             try:
#                 site_info = vm.get_site_information(_station)
#                 lat = site_info['latitude']
#                 lon = site_info['longitude'] 
#             except:
#                 lat, lon = -999.9, -999.9
#             stations_data = {"name": _station, "lat": lat, "lon": lon, "channels":channel_model_list}
#             station_model = Station(**stations_data)
#             station_model_list.append(station_model)

#         volcano_model = Volcano(volcano=_volcano, stations=station_model_list)
#         volcano_model_list.append(volcano_model)

#     if len(volcano_model_list) == 0:
#         return('no volcano')

#     scenario_model = Scenario(scenario='VUMT', volcanoes=volcano_model_list)
#     if volcano != "all":
#         # return all stations for a volcano
#         for _volcano in scenario_model.volcanoes:
#             if _volcano.volcano == volcano:
#                 if station == "all":
#                     return _volcano
#                 for _station in _volcano.stations:
#                     if _station.name == station:
#                         if channel == "all":
#                             return _station
#                         for _channel in _station.channels:
#                             if _channel.name == channel:
#                                 feature_list_filtered = []
#                                 for _f in _channel.features:
#                                     if _f in FeatureRequest.feat_dict[type]:
#                                         feature_list_filtered.append(_f)    
#                                 _channel.features = feature_list_filtered
#                                 return _channel

#     return scenario_model



def main(argv=None):
    uvicorn.run(app, host="0.0.0.0", port=8003)

if __name__ == "__main__":
    main()
