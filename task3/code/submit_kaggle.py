"""Submit a competition file to Kaggle, bypassing the www.googleapis.com upload host.

Kaggle's CLI uploads via www.googleapis.com which is unreachable here; the
same resumable-upload URL works on storage.googleapis.com.
Usage: python3 submit_kaggle.py submission.csv "description" [competition]
"""
import os, sys
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiStartSubmissionUploadRequest, ApiCreateSubmissionRequest

f = sys.argv[1]
desc = sys.argv[2] if len(sys.argv) > 2 else 'submission'
comp = sys.argv[3] if len(sys.argv) > 3 else '2026task3'

api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as kaggle:
    req = ApiStartSubmissionUploadRequest()
    req.competition_name = comp
    req.file_name = os.path.basename(f)
    req.content_length = os.path.getsize(f)
    req.last_modified_epoch_seconds = int(os.path.getmtime(f))
    r = kaggle.competitions.competition_api_client.start_submission_upload(req)
    url = r.create_url.replace('https://www.googleapis.com', 'https://storage.googleapis.com')
    token = r.token
    with open(f, 'rb') as fh:
        up = requests.put(url, data=fh, headers={'Content-Type': 'application/octet-stream'}, timeout=120)
    if up.status_code not in (200, 201):
        print('UPLOAD FAILED', up.status_code, up.text[:200]); sys.exit(1)
    creq = ApiCreateSubmissionRequest()
    creq.competition_name = comp
    creq.blob_file_tokens = token
    creq.submission_description = desc
    resp = kaggle.competitions.competition_api_client.create_submission(creq)
    print('SUBMITTED', resp)
