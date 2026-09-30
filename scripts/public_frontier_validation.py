"""Reject partial execution of the pinned public four-reader ensemble."""


COAT_MEMBERS = {"resgated_top3", "global96_top3", "d4_swa3", "repairv1_top3"}
COAT_EPOCHS = {
    "resgated_top3": [4, 6, 8],
    "global96_top3": [16, 23, 18],
    "repairv1_top3": [12, 7, 11],
}
COAT_STATUSES = {
    "resgated_top3": "VALID_COAT_RESGATED_EP10_TOP3_RANK_SUBMISSION",
    "global96_top3": "VALID_COATNET_GLOBAL96_TOP3_T4X2_SUBMISSION",
    "repairv1_top3": "VALID_COATNET_REPAIRV1_TOP3_PROBABILITY_RANK_SUBMISSION",
    "d4_swa3": "VALID_COATNET_D4_DEPTHZONE_SWA3_T4X2_SUBMISSION",
}
REQUIRED_PHASES = {
    "a5_rank_mean", "btkd_parent_after_rad", "coat_resgated_rank",
    "coat_global96_rank", "coat_d4_rank", "coat_repairv1_rank", "final",
}
FAILED_EVENTS = {
    "coverage_mismatch", "slot_no_files", "partial_slice_decode",
    "slot_all_decode_failed", "slot_shape_mismatch", "ordering_fallback",
    "deterministic_order_fallback", "slot_read_error", "slot_unfilled",
    "empty_study_rows", "dino_jitter_fp16_nonfinite_dropped",
    "dino_member_nonfinite_neutral", "dino_member_public_nonfinite_neutral",
    "submission_neutral_fill", "a5_order_unreadable_dropped", "a5_no_metadata",
    "a5_slot_no_files", "a5_slot_all_crops_failed", "a5_decode_failure",
    "a5_empty_study", "a5_nonfinite_neutral", "a5_study_failed",
    "rad_nonfinite_zeroed", "rad_head_nonfinite_neutral", "rad_invalid_values_repaired",
    "raptor_order_index_fallback", "raptor_order_nonfinite_fallback",
    "raptor_order_unreadable_skipped", "raptor_acquisition_empty",
    "raptor_study_failed", "raptor_neutral_fill", "raptor_preparation_identities_incomplete",
    "coat_family_unavailable", "coat_family_partial", "coat_member_probability_unavailable",
}


def validate_public_frontier(audit, blend, receipts, study_count):
    """Validate receipts; callers additionally verify prediction bytes and UID order."""
    def need(condition, message):
        if not condition:
            raise ValueError("Incomplete public frontier ensemble: " + message)

    need(study_count > 0, "empty study cohort")
    need(audit.get("status") == "COMPLETE", "pipeline status")
    need(audit.get("cohort") == study_count, "pipeline study count")
    need(audit.get("dino_members") == 20, "DINO member count")
    need(audit.get("a5_folds") == 5, "A5 fold count")
    need(audit.get("raptor_views") == 4, "Raptor view count")
    members = blend.get("members", [])
    need(len(members) == 4 and set(members) == COAT_MEMBERS, "all four CoAt readers required")
    need(blend.get("family_reduction") == "rank_of_member_probability_mean", "probability reduction")
    need(audit.get("family_reduction") == blend["family_reduction"], "audit reduction differs")
    need(audit.get("coat_family_members") == 4, "audit reader count")
    need(blend.get("within_coat") == {name: .25 for name in COAT_MEMBERS}, "equal reader weights")
    need(blend.get("private_alpha") == .4 and blend.get("public_raptor_alpha") == .6, "inner weights")
    need(blend.get("inner_rerank") is False and audit.get("inner_rerank") is False, "inner rerank")
    need(blend.get("study_count") == study_count, "blend study count")
    need(set(receipts) == COAT_MEMBERS, "reader receipts")
    for name, receipt in receipts.items():
        need(receipt.get("status") == COAT_STATUSES[name], name + " child status")
        need(receipt.get("fallback_studies") == 0, name + " fallback study count")
        need(not receipt.get("failures"), name + " failed studies")
        need(receipt.get("models") == (1 if name == "d4_swa3" else 3), name + " model count")
        if name in COAT_EPOCHS:
            need([item.get("epoch") for item in receipt.get("checkpoints", [])] == COAT_EPOCHS[name],
                 name + " checkpoint epochs")
    events = audit.get("events")
    need(isinstance(events, list), "missing event record")
    bad = []
    for event in events:
        kind = event.get("kind", "")
        if kind in FAILED_EVENTS or (kind.startswith("coat_") and
                kind.endswith(("_child_failed", "_output_rejected", "_fallback_studies"))):
            bad.append(kind)
    need(not bad, "failed/fallback events: " + ", ".join(bad))
    phases = audit.get("phases", {})
    need(REQUIRED_PHASES.issubset(phases), "missing component predictions")
    dino_phases = {name for name in phases if name.startswith("dino_")}
    need(len(dino_phases) == 20, "missing DINO prediction artifacts")
    for name in REQUIRED_PHASES | dino_phases:
        proof = phases[name]
        need(proof.get("shape") == [study_count, 12], name + " prediction shape")
        need(isinstance(proof.get("sha256"), str) and len(proof["sha256"]) == 64,
             name + " missing prediction digest")
    return {"complete_readers": sorted(COAT_MEMBERS), "prediction_artifacts": len(REQUIRED_PHASES | dino_phases),
            "failed_fallback_events": 0, "study_count": study_count}
