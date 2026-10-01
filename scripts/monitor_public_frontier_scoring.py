"""Observe only Ticket 51's accepted reference; never submit, retry or select."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from score_public_frontier import ROOT, PROPOSAL_SHA, HANDLE, VERSION, ledger_action, local_evidence, read, save


def source_identity(item, reference: int, *, kernel_id=HANDLE, kernel_version=VERSION) -> dict:
    match = re.fullmatch(r"/code/" + re.escape(kernel_id) + r"\?scriptVersionId=(\d+)", item.url)
    if item.ref != reference or match is None or int(match.group(1)) <= 0:
        raise ValueError("accepted submission reference or source URL differs")
    return {"submission_ref": reference, "kernel_id": kernel_id, "kernel_version": kernel_version,
            "script_version_id": int(match.group(1)), "source_url": item.url,
            "binding_basis": "Accepted create-code-submission request explicitly sets the saved kernel version; exact request, response and preflight preserved."}


def resolve_observation(exact, listing, reference: int, *, kernel_id=HANDLE, kernel_version=VERSION):
    # GetSubmission.url is the raw CSV download, not the saved-code identity.
    # Keep its exact status/score record separate from ListSubmissions' source.
    if exact.ref != reference:
        raise ValueError("exact status response is for another submission")
    matches = [row for row in listing if row.ref == reference]
    if len(matches) != 1:
        raise ValueError("accepted reference missing or duplicated in source listing")
    listed = matches[0]
    source = source_identity(listed, reference, kernel_id=kernel_id, kernel_version=kernel_version)
    return source, listed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--watch", action="store_true", help="poll this accepted reference every 60 seconds")
    args = parser.parse_args()
    root = args.root.resolve()
    local = local_evidence(root)
    intent = read(root / "scoring-submit-intent.json")
    response = read(root / "scoring-submit-response.json")
    if intent["proposal_sha256"] != PROPOSAL_SHA or response["ref"] <= 0:
        raise ValueError("an accepted reference bound to the exact immutable request is required")
    reference = response["ref"]
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.competitions.types.competition_api_service import ApiGetSubmissionRequest
    api = KaggleApi()
    api.authenticate()
    while True:
        observed = {"observed_at_utc": datetime.now(timezone.utc).isoformat(), "submission_ref": reference}
        try:
            request = ApiGetSubmissionRequest()
            request.ref = reference
            with api.build_kaggle_client() as client:
                item = client.competitions.competition_api_client.get_submission(request)
            source_path = root / "accepted-scoring-source.json"
            listing = api.competition_submissions(local["proposal"]["competition"], page_size=100)
            source, listed = resolve_observation(item, listing, reference)
            if source_path.exists():
                if read(source_path) != source:
                    raise ValueError("accepted saved-source identity drifted")
            else:
                save(source_path, source)
            name = item.status.name
            observed.update(status=name, public_score=item.public_score, private_score=item.private_score,
                            error_description=item.error_description, source_url=source["source_url"],
                            output_download_url=item.url,
                            script_version_id=source["script_version_id"])
            if name in {"COMPLETE", "ERROR"}:
                evidence_path = root / "terminal-scoring-evidence.json"
                if not evidence_path.exists():
                    normalized = json.loads(item.to_json(ignore_defaults=False))
                    normalized["status"] = name
                    source_submission = json.loads(listed.to_json(ignore_defaults=False))
                    source_submission["status"] = listed.status.name
                    save(evidence_path, {"schema_version": "rsna-kaggle-scoring-terminal-evidence-v1",
                        "request_checksum": PROPOSAL_SHA, "job_id": str(reference),
                        "observed_at_utc": observed["observed_at_utc"],
                        "observer": "authenticated exact Kaggle GetSubmission status plus exact-reference ListSubmissions source",
                        "kernel_id": HANDLE, "kernel_version": VERSION,
                        "script_version_id": source["script_version_id"], "submission": normalized,
                        "source_submission": source_submission,
                        "field_provenance": {"status_scores_and_download_url": "GetSubmission",
                                             "saved_code_source_url": "ListSubmissions filtered to accepted ref"},
                        "allocation_timing_available": False,
                        "accounting_basis": "Charge full two-GPU reservation when hidden timing is unavailable."})
                observed["campaign_capacity"] = ledger_action("terminal", root)
                observed["conservative_hidden_gpu_seconds_charged"] = 57600
            temporary = root / "scoring-status.json.partial"
            temporary.write_text(json.dumps({**observed, "competition": local["proposal"]["competition"],
                "kernel_id": HANDLE, "kernel_version": VERSION, "request_checksum": PROPOSAL_SHA,
                "final_candidate_selection": False}, indent=2) + "\n")
            temporary.replace(root / "scoring-status.json")
            print(json.dumps(observed), flush=True)
            with (root / "scoring-observations.jsonl").open("a") as handle:
                handle.write(json.dumps(observed) + "\n")
            if name in {"COMPLETE", "ERROR", "CANCELLED"} or not args.watch:
                return
        except Exception as exc:
            # An observation failure does not prove termination or justify retry.
            observed["observation_error_type"] = type(exc).__name__
            print(json.dumps(observed), flush=True)
            with (root / "scoring-observations.jsonl").open("a") as handle:
                handle.write(json.dumps(observed) + "\n")
            if not args.watch:
                raise
        time.sleep(60)


if __name__ == "__main__":
    main()
