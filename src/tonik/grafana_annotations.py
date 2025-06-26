"""
Post labels as Grafana annotations and retrieve them via the Grafana API.
Labels are assumed to be in the following format:

[
    {"time": "2023-01-01T00:00:00Z",
     "title": "Label 1",
     "description": "Some description",
     "tags": ["tag1", "tag2"],
     "id": 12345
     },
    {"time": "2023-02-01T00:00:00Z",
     "timeEnd": "2023-02-01T01:00:00Z",
     "title": "Label 2", description:
     "description": "Another description",
     "tags": ["tag1", "tag2"],
     "id": 12346
     },
     ...
]

Note: The `timeEnd` field is optional. If it is provided, the grafana annotation will 
be a range annotation, otherwise it will be a point annotation. The label id is used to
check if an annotation was already added to Grafana. If the label id is not present in the
annotations, a new annotation will be created. If the label id is present, the annotation
will be skipped to avoid duplicates.
"""

import numpy as np
import requests
import uuid
from typing import List, Dict, Any, Optional


def get_annotations(baseurl: str, tags: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get existing annotations from the Grafana API.

    Parameters
    ----------
    baseurl : str
        The base URL of the Grafana API. This is typically in the format
        "http://<grafana-user>:<user-pwd>@<grafana-host>:<port>".
    tags : list of str, optional
        A list of tags to filter the annotations. If None, all annotations are retrieved.
    Returns
    -------
    annotations : dict
        A dictionary containing the annotations, where the keys are the label IDs
        and the values are dictionaries with the following keys:
            - 'text': The text of the annotation.
            - 'time': The start time of the annotation in milliseconds since epoch.
            - 'timeEnd': (optional) The end time of the annotation in milliseconds since epoch.
            - 'tags': A list of tags associated with the annotation.
            - 'id': The ID of the annotation.
    """
    url = baseurl + "/api/annotations"
    params = {}
    if tags is not None:
        params = {"tags": tags}
    rval = requests.get(url, params=params)
    if rval.status_code != 200:
        raise RuntimeError(
            f"Failed to retrieve annotations: {rval.status_code} {rval.text}")
    annotations = {}
    for atn in rval.json():
        # Extract label ID from text using regex
        try:
            label_id = atn["text"].split("Id: ")[-1].strip()
        except IndexError:
            label_id = str(uuid.uuid4())
        annotations[label_id] = {"text": atn["text"],
                                 "time": atn["time"],
                                 "timeEnd": atn.get("timeEnd", None),
                                 "tags": atn.get("tags", []),
                                 "id": atn["id"]}
    return annotations


def post_annotations(baseurl: str, labels: List[Dict[str, Any]]) -> None:
    """
    Post annotations from a list of labels if they do not already exist.

    Parameters
    ----------
    baseurl : str
        The base URL of the Grafana API. This is typically in the format
        "http://<grafana-user>:<user-pwd>@<grafana-host>:<port>".
    labels : list of dict
        A list of dictionaries containing label information. Each dictionary
        should have the following keys:
            - 'time': The start time of the annotation in ISO format.
            - 'timeEnd': (optional) The end time of the annotation in ISO format.
            - 'title': The title of the annotation.
            - 'description': The description of the annotation.
            - 'tags': A list of tags associated with the annotation.
            - 'id': A unique identifier for the label, used to check for duplicates.
    """
    url = baseurl + "/api/annotations"
    header = {"Content-type": "application/json"}
    for label in labels:
        existing_labels = get_annotations(baseurl, tags=label['tags'])
        if str(label['id']) in existing_labels:
            print(
                f"Label with ID {label['id']} already exists. Skipping post.")
            continue
        starttime = np.datetime64(label['time']).astype(
            'datetime64[ms]').astype(int)
        try:
            endtime = np.datetime64(label['timeEnd']).astype(
                'datetime64[ms]').astype(int)
            endtime = int(endtime)
        except KeyError:
            endtime = None
        text = f"{label['title']}\n{label['description']}\nId: {label['id']}"
        new_annotation = {
            "time": int(starttime),
            "timeEnd": endtime,
            "text": text,
            "tags": label['tags']
        }
        rval = requests.post(url, headers=header, json=new_annotation)
        if rval.status_code != 200:
            raise RuntimeError(
                f"Failed to post annotation: {rval.status_code} {rval.text}")
        else:
            print("Annotation posted successfully.")


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('url', type=str,
                        help='Grafana API URL for annotations.')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to JSON file containing labels to post.')
    parser.add_argument('--get-annotations', action='store_true',
                        help='Get existing annotations from Grafana API.')
    args = parser.parse_args(argv)
    if args.labels is not None:
        with open(args.labels, 'r') as f:
            labels = json.load(f)
        post_annotations(args.url, labels)
    elif args.get_annotations:
        annotations = get_annotations(args.url)
        print(json.dumps(annotations, indent=2))


if __name__ == "__main__":
    main()
