# ResistanceMap v3 Release Artifacts

Upload all four files below to a GitHub release on `Abhignya-Jagathpally/temp` (tag e.g. `v3-artifacts`). Each file is under the 2 GB per-asset limit.

## Files

| File | Size | Contents |
|---|---|---|
| `checkpoints.tar.zst` | 883 MB | `ResistanceMap/checkpoints/` — all 8 trained model checkpoints (VAE pretrain/finetune, ESM2, trajectory, protein net, fusion, landscape, validation) |
| `data_raw.tar.zst.part-00` | 1.8 GB | First half of `ResistanceMap/data/raw/` (CCLE, GDSC, STRING, GEO h5ad, etc.) |
| `data_raw.tar.zst.part-01` | 245 MB | Second half — concatenate with part-00 before decompressing |
| `SHA256SUMS` | <1 KB | Integrity checksums |

## Download & restore

```bash
# Verify
sha256sum -c SHA256SUMS

# Checkpoints
tar --zstd -xf checkpoints.tar.zst -C ResistanceMap/

# Raw data (reassemble split, then extract)
cat data_raw.tar.zst.part-00 data_raw.tar.zst.part-01 | zstd -d | tar -xf - -C ResistanceMap/
```

After extraction you'll have `ResistanceMap/checkpoints/` and `ResistanceMap/data/raw/` populated. The pipeline can then resume from checkpoints via `python main.py --config configs/default.yaml`.

## Upload instructions (web UI)

1. Go to https://github.com/Abhignya-Jagathpally/temp/releases/new
2. Target branch: `v3`
3. Tag: `v3-artifacts` (create new)
4. Title: `ResistanceMap v3 — checkpoints + raw data`
5. Drag all 4 files (`checkpoints.tar.zst`, `data_raw.tar.zst.part-00`, `data_raw.tar.zst.part-01`, `SHA256SUMS`) into the attachments area.
6. Publish release.
