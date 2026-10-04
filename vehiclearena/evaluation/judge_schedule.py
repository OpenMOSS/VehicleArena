"""Pure configuration for bounded LLM acceptance checks (simulation seconds)."""
import math


def resolve_judge_schedule(*, window_s=None, max_checks=3, check_offsets_s=None,
                           acceptance_timeout_s=None, request_ttl_s=None):
    if (isinstance(max_checks, bool) or not isinstance(max_checks, int)
            or not 1 <= max_checks <= 100):
        raise ValueError("max_checks must be an integer in [1, 100]")
    if check_offsets_s is not None and window_s is not None:
        raise ValueError("Use check_offsets_s or legacy window_s, not both")
    if check_offsets_s is None:
        if window_s is None:
            if max_checks > 3:
                raise ValueError("More than 3 checks requires explicit offsets or window_s")
            offsets = [0.1, 1.0, 3.0][:max_checks]
        else:
            offsets = [float(window_s) * i for i in range(1, max_checks + 1)]
    else:
        offsets = [float(t) for t in check_offsets_s]
        if not offsets or len(offsets) > max_checks:
            raise ValueError("Check offsets must be nonempty and fit max_checks")
    if any(not math.isfinite(t) or t <= 0 for t in offsets) or any(
            a >= b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("Check offsets must be finite, positive and strictly increasing")
    if acceptance_timeout_s is not None and request_ttl_s is not None:
        raise ValueError("Use acceptance_timeout_s or legacy request_ttl_s, not both")
    timeout = (acceptance_timeout_s if acceptance_timeout_s is not None
               else request_ttl_s)
    timeout = offsets[-1] if timeout is None else float(timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Acceptance timeout must be finite and positive")
    # An earlier timeout clips the first out-of-budget check to the deadline.
    clipped = []
    for offset in offsets:
        clipped.append(min(offset, timeout))
        if offset >= timeout:
            break
    return clipped, min(timeout, clipped[-1])
