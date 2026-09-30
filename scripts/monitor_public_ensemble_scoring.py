"""Poll exactly one accepted Kaggle submission; never launch or retry work."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def main() -> None:
    root = Path('/home/ch.jitheswar/RSNA/artifacts/private/public-notebook-adoption/scored-baseline-v1')
    status_path = root / 'scoring-status.json'
    state = json.loads(status_path.read_text())
    reference = state['submission_ref']
    api = KaggleApi()
    api.authenticate()
    (root / 'scoring-monitor.pid').write_text(str(os.getpid()) + '\n')
    while True:
        observed = {'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'submission_ref': reference}
        try:
            matches = [item for item in api.competition_submissions(state['competition'], page_size=100) if item.ref == reference]
            if len(matches) != 1:
                raise RuntimeError('exact submission reference missing from authoritative listing')
            item = matches[0]
            name = item.status.name
            observed.update(status=name, public_score=item.public_score, private_score=item.private_score, error_description=item.error_description, source_url=item.url)
            state.update(observed)
            temporary = status_path.with_suffix('.partial')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(status_path)
            print(json.dumps(observed), flush=True)
            if name in {'COMPLETE', 'ERROR', 'CANCELLED'}:
                with (root / 'scoring-observations.jsonl').open('a') as handle:
                    handle.write(json.dumps(observed) + '\n')
                return
        except Exception as exc:
            observed['observation_error'] = f'{type(exc).__name__}: {exc}'
            print(json.dumps(observed), flush=True)
        with (root / 'scoring-observations.jsonl').open('a') as handle:
            handle.write(json.dumps(observed) + '\n')
        time.sleep(60)


if __name__ == '__main__':
    main()
