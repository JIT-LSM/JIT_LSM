import os
from LargeModel.data.cut_code_to_snippet import cut as cut_code
from LargeModel.data.cut_json_to_commit import cut as cut_json

RQ1_JSON_FILE = "data/unbalanced_dataset_1730.json"
RQ2_JSON_FILE = "data/balanced_dataset_1000.json"
RQ1_JSON_DIR = "data/unbalanced_dataset_1730"
RQ2_JSON_DIR = "data/balanced_dataset_1000"

if os.path.isfile(RQ1_JSON_FILE):
    cut_code(RQ1_JSON_FILE, RQ1_JSON_DIR)
    cut_json(RQ1_JSON_FILE, RQ1_JSON_DIR)

if os.path.isfile(RQ2_JSON_FILE):
    cut_code(RQ2_JSON_FILE, RQ2_JSON_DIR)
    cut_json(RQ2_JSON_FILE, RQ2_JSON_DIR)