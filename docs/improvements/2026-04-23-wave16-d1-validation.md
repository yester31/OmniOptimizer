# Wave 16 D1 end-to-end validation — caught a merged-code bug

**Date**: 2026-04-23
**Related**: PR #11 (D1 shipped) — this doc records the post-merge validation that caught a None-handling crash.

## What was validated

`Result.build_ceiling_breached` round-trip through `scripts/run_trt.py::run()` on the real hardware (RTX 3060 Laptop + i7-11375H + TRT 10.16.0.72), across all three tracker states.

## Validation matrix

All three cases ran `yolo26n.pt` FP16 via `python scripts/run_trt.py --recipe <tmp> --out <tmp.json>`.

| Case | `build_ceiling_s` | Actual `build_time_s` | Expected | Observed | Pass? |
|------|-------------------|----------------------|----------|----------|:----:|
| Breach | `1` (unreachable) | 22.4s | `True` | `True` | ✓ |
| Clean | unset → 600 default | 332.5s | `False` | `False` | ✓ |
| Cache hit | `1` (unreachable) | `null` (engine cached from breach run) | `None` | `None` | ✓ |

## What the validation caught

**Bug in merged PR #11 code**: the tracker conditional at `scripts/run_trt.py:948` did:

```python
if build_time_s > _ceiling:
```

without guarding for `None`. The cached-engine path at `_build_engine`'s `return engine_path, None, None` (line 608) fed `build_time_s = None` straight into the comparison, raising `TypeError: '>' not supported between instances of 'NoneType' and 'int'`.

**Symptom**: running *any* recipe whose engine was already built would crash the runner at the tracker step, before the measurement even started. This was not caught by the D1 unit tests because those tested the schema only, not the runner's control flow against the cached-engine return shape.

## Fix

Extracted tracker logic into `_advance_ceiling_tracker(prev, build_time_s, ceiling)` — pure function with sticky-True semantics, explicitly handles `None`. Four unit tests pin the semantics table:

- `None` build time + any prev → prev unchanged
- `> ceiling` → `True` (sticky regardless of prev)
- `<= ceiling` and prev is `None` → promote to `False`
- `<= ceiling` and prev is `True` → stay `True` (sticky invariant)

Runner now calls the helper:

```python
build_ceiling_breached = _advance_ceiling_tracker(
    build_ceiling_breached, build_time_s, _ceiling
)
```

## Lessons

1. **Unit tests must exercise the return-shape matrix of dependencies, not just the feature itself.** D1's unit tests covered schema round-trip and recommend.py surfacing, but not the runner's loop conditional against `_build_engine`'s 5 return shapes (2 of which carry `build_time_s = None`). The bug was one `.` away from being caught.
2. **End-to-end validation is cheap and catches real bugs.** Three GPU runs (~12 min total) caught a runner-level crash that would have hit every user whose TRT engine was cached.
3. **"It merged, so it works" is a mistake.** Even with 179→186 test growth across 4 D1 commits, a latent crash shipped to main. Future PRs that mutate runner control flow should include at least one end-to-end run as evidence.

## Follow-up test coverage (this PR)

`tests/test_build_ceiling_breached.py` gains 4 tracker tests:

- `test_advance_ceiling_tracker_none_build_time_keeps_prev` — the exact regression
- `test_advance_ceiling_tracker_breach_sets_true` — sticky semantics from breach
- `test_advance_ceiling_tracker_under_ceiling_promotes_none_to_false` — initial promotion
- `test_advance_ceiling_tracker_under_ceiling_never_demotes_true` — sticky invariant

Full suite: **198 passed** (was 193 before D1 validation work).

## Not done

- **No CI-level integration test that runs a real GPU recipe.** The hardware dependency makes this hard to CI, but the local validation script described in this doc is reproducible on any TRT-enabled workstation.
- **No automatic ceiling-breach failure in CI.** The signal is still diagnostic-only — operators decide whether to act. This was Wave 16 D1's stated non-goal.

## Reproducing the validation

On a workstation with TRT + a GPU:

```bash
# Breach case
python scripts/run_trt.py --recipe <a-recipe-with-build_ceiling_s-1> --out /tmp/breach.json
python -c "import json; d=json.load(open('/tmp/breach.json')); print(d['build_ceiling_breached'])"
# → True

# Clean case
python scripts/run_trt.py --recipe <any-unmodified-recipe> --out /tmp/clean.json
python -c "import json; d=json.load(open('/tmp/clean.json')); print(d['build_ceiling_breached'])"
# → False (if build completes under the 600s default)

# Cache-hit case — re-run breach case, engine now cached
python scripts/run_trt.py --recipe <same-breach-recipe> --out /tmp/cache.json
python -c "import json; d=json.load(open('/tmp/cache.json')); print(d['build_ceiling_breached'])"
# → None
```

Temporary recipes with `build_ceiling_s: 1` used during this validation were deleted after the runs — not shipped.
