import os
import subprocess
import sys
from kaggle.api.kaggle_api_extended import (KaggleApi, ApiStartSubmissionUploadRequest,
                                            ApiCreateSubmissionRequest)


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
        print('got upload url, token', resp.token[:24], flush=True)
        cmd = ['curl', '-sS', '--http1.1', '-X', 'PUT', '--upload-file', path,
               '-w', '%{http_code} %{time_total}', resp.create_url]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=420)
        except subprocess.TimeoutExpired:
            print('curl timeout', flush=True)
            sys.exit(2)
        print('curl rc', out.returncode, 'out', out.stdout[-100:], 'err', out.stderr[-300:], flush=True)
        code = out.stdout.strip().split()[0] if out.stdout.strip() else ''
        if code not in ('200', '201'):
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
