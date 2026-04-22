# Launch Operations

Operational checks for Day 1-7 community launch logs.

## URL Evidence Gate

Use this command to list all `PENDING` URL entries in launch runbooks:

```bash
./scripts/check_launch_url_evidence.sh
```

Behavior:
- exits `0` when all Day 1-7 `Published URL` fields are filled
- exits `1` when at least one `PENDING` URL remains

## Recommended Closing Flow

1. Fill all `Published URL` fields in `docs/day*_launch_*.md`.
2. Run `./scripts/check_launch_url_evidence.sh`.
3. Commit and push only launch URL evidence updates.
