import os
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit, parse_qs
from kaggle.api.kaggle_api_extended import (KaggleApi, ApiStartSubmissionUploadRequest,
                                            ApiCreateSubmissionRequest)


def put(url, path, timeout=300):
    cmd = ['curl', '-sS', '--http1.1', '-X', 'PUT', '--upload-file', path,
           '-w', '\n%{http_code} %{time_total}', url]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    text = out.stdout.strip()
    code = text.split()[-2] if len(text.split()) >= 2 else ''
    return code, out.stderr[-300:], text[-80:]


def main():
    comp, path, msg = sys.argv[1], sys.argv[2], sys.argv[3]
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        req = ApiStartSubmissionUploadRequest()
        req.competition_name = comp
        req.file_name = os.path.basename(path)
        req.content_length = os.path.getsize(path)
        req.last_modified_epoch_seconds = int(os.path.getmtime(path))
        resp = client.competitions.competition_api_client.start_submission_upload(req)
        parts = urlsplit(resp.create_url)
        print('host', parts.netloc, 'path', parts.path[:90], 'qkeys', list(parse_qs(parts.query).keys()), flush=True)
        if parts.netloc == 'www.googleapis.com':
            urls = [urlunsplit((parts.scheme, 'storage.googleapis.com', parts.path, parts.query, ''))]
        else:
            urls = [resp.create_url]
        ok = False
        for u in urls:
            host = urlsplit(u).netloc
            try:
                code, err, tail = put(u, path)
            except subprocess.TimeoutExpired:
                print('timeout', host, flush=True)
                continue
            print('host', host, 'code', code, 'tail', tail, 'err', err, flush=True)
            if code in ('200', '201'):
                ok = True
                break
        if not ok:
            print('upload failed', flush=True)
            sys.exit(3)
        sreq = ApiCreateSubmissionRequest()
        sreq.competition_name = comp
        sreq.blob_file_tokens = resp.token
        sreq.submission_description = msg
        sresp = client.competitions.competition_api_client.create_submission(sreq)
        print('submitted:', sresp, flush=True)


if __name__ == '__main__':
    main()
