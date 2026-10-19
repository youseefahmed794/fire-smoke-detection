"""Presentation metadata; these assessments never establish fire authenticity."""

UNCERTAIN_LABELS = {
    "No Flicker": "Fire candidate - low temporal change",
    "Screen": "Fire candidate - screen overlap",
}
REASON_CODES = {"No Flicker": "low_temporal_change", "Screen": "screen_overlap"}


def assessment(model_class=None, *, eligible=False, rejection_reason=None, legacy_status=None):
    """Keep model class, heuristic rejection, and presentation separate.

    alert_eligible describes only candidate gates, not cooldown/dispatch policy.
    Neither passing nor failing a heuristic verifies real/spoof authenticity.
    """
    rejected = bool(rejection_reason)
    label = UNCERTAIN_LABELS.get(rejection_reason, model_class or "No Detection")
    return {
        "model_class": model_class, "confidence_eligible": bool(eligible),
        "rejected": rejected, "rejection_reason": rejection_reason,
        "reason_code": REASON_CODES.get(rejection_reason),
        "alert_eligible": bool(eligible and not rejected),
        "authenticity": "unverified", "display_label": label,
        "display_status": label if rejected else (legacy_status or label),
    }


def display_result(result, legacy_status=None):
    """Use structured metadata; adapt older injected detectors/reports if needed."""
    if isinstance(result, dict) and "display_label" in result:
        return result
    # Compatibility only: current Detector supplies structured fields directly.
    for reason in UNCERTAIN_LABELS:
        if legacy_status == f"Fake Fire ({reason})":
            return assessment("Fire", eligible=True, rejection_reason=reason)
    for label in ("Fire", "Smoke"):
        if legacy_status and (legacy_status == label or legacy_status.startswith(label + " ")):
            return assessment(label, eligible=True, legacy_status=legacy_status)
    return assessment()
