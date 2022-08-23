# /big_data/2021_code/
# 100.24.88.93
import json
import os
import sys
from typing import Dict, List, Optional

from doc_preprocessing.dataclasses import Span

import flask
from flask import request
from refined.processor import Refined

sys.path.append(".")

data_dir = "/big_data/2021_data"
# model_dir = '/big_data/2021_data/model_dir/wikipedia_model'
model_dir = "/big_data/2021_may_ft"
code_dir = "/big_data/2021_code"
datasets_dir = "/big_data/2021_data/datasets"


sys.path.append(code_dir)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = flask.Flask(__name__)
app.config["DEBUG"] = True

refined = None
is_ready = False


def init():
    global refined
    debug = False
    device = "cuda:0"
    refined = Refined(
        model_dir=model_dir,
        data_dir=data_dir,
        debug=debug,
        requires_redirects_and_disambig=True,
        backward_coref=True,
        device=device,
        use_cpu=False,
    )
    refined.preprocessor.max_candidates = 30
    refined.backward_coref = True
    refined.preprocessor.zero_features = False
    refined.preprocessor.zero_string_features = True
    refined.model.use_precomputed_descriptions = False
    refined.model.use_kg_layer = False


def convert_span_to_gerbil_format(span: Span, threshold=0.25) -> Optional[Dict]:
    if span.pred_entity_id is None or span.pred_entity_id[1] < threshold:
        return None
    # uri = f'http://www.wikidata.org/entity/{span.pred_entity_id[0]["wikidata_qcode"]}'
    # uri = f'http://en.wikipedia.org/wiki/{span.pred_entity_id[0]["wikipedia_title"]}'
    if "wikipedia_title" not in span.pred_entity_id[0]:
        # should print these cases to see why
        return None
    # uri = f'http://dbpedia.org/resource/{span.pred_entity_id[0]["wikipedia_title"]}'
    # consider replacing "%22" with " and replacing & with if using dbpedia resouce instead of Wikipedia
    # may need url encoding
    # Wikidata seems broken
    # 21/22 errors for N3-RSS-500
    uri = f'http://en.wikipedia.org/wiki/{span.pred_entity_id[0]["wikipedia_title"]}'

    return {"start": span.start, "length": span.ln, "uri": uri}


def process_request(request_dict: Dict) -> List[Dict]:
    print(request_dict)
    doc_text = request_dict["text"]
    spans: List[Span] = []
    request_dict["spans"].sort(key=lambda x: x["startPosition"])
    for gerbil_request_span in request_dict["spans"]:
        start = gerbil_request_span["startPosition"]
        ln = gerbil_request_span["length"]
        text = doc_text[start : start + ln]
        spans.append(Span(text=text, start=start, ln=ln))
    ents = refined.process_text(doc_text, spans=spans, apply_class_check=False)
    print(ents)
    return list(filter(lambda x: x is not None, map(convert_span_to_gerbil_format, ents)))


def process_request_el(request_dict: Dict) -> List[Dict]:
    print(request_dict)
    doc_text = request_dict["text"]
    ents = refined.process_text(doc_text, apply_class_check=False)
    ents = [ent for ent in ents if "wikidata_qcode" in ent.pred_entity_id[0]]
    print(ents)
    return list(filter(lambda x: x is not None, map(convert_span_to_gerbil_format, ents)))


@app.route("/ed", methods=["POST"])
def disamb() -> str:
    global refined, is_ready
    if not is_ready:
        print("init() called")
        init()
        is_ready = True
    print("request made")
    request_content = json.loads(request.data, strict=False)
    return json.dumps(process_request(request_content))


@app.route("/el", methods=["POST"])
def disamb_el() -> str:
    global refined, is_ready
    if not is_ready:
        print("init() called")
        init()
        is_ready = True
    print("request made")
    try:
        request_content = json.loads(request.data, strict=False)
    except Exception as err:
        print(err)
        print(f"JSONERROR data: {request.data}")
    return json.dumps(process_request_el(request_content))


app.run(port=80, host="0.0.0.0", threaded=False)
# app.run(port=80, host='127.0.0.1', threaded=False)
