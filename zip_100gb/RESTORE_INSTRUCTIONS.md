# Boichitro-MoE full backup

This directory preserves both project folders before deletion:

- `Tamim_sir _work/`: the complete experiment workspace, including datasets,
  checkpoints, runs, caches, predictions, reports, notebooks, and original ZIP
  inputs
- `boichitro-moe-github/`: the compact GitHub-ready repository, including its
  `.git` history

The full archive is:

```text
boichitro-moe-complete-backup-2026-07-24.tar.zst
```

Two smaller recovery files are also included:

- `boichitro-moe-github-source-7cccbd8.zip`: the 475 committed project files,
  ready to extract or submit; it intentionally has no `.git` directory
- `boichitro-moe-github-history-7cccbd8.bundle`: the complete local Git
  repository and commit history

GNU tar with Zstandard compression is used instead of ordinary ZIP because the
experiment contains many hard-linked checkpoints. This format preserves those
links without duplicating tens of gigabytes.

## Verify before deleting anything

From this backup directory:

```bash
sha256sum --check SHA256SUMS.txt
zstd --test boichitro-moe-complete-backup-2026-07-24.tar.zst
tar --zstd --list --file boichitro-moe-complete-backup-2026-07-24.tar.zst > /dev/null
```

Do not delete source data unless the commands succeed and
`BACKUP_VERIFIED.txt` is present. Keep this entire backup directory.

## Restore the complete experiment

Choose a destination with at least 100 GB of free space:

```bash
mkdir -p /path/to/restore
tar --zstd --extract \
  --file boichitro-moe-complete-backup-2026-07-24.tar.zst \
  --directory /path/to/restore
```

After extraction, both original top-level folders will be present beneath the
chosen destination.

## Restore only the GitHub project

To get a normal source folder without Git history:

```bash
unzip boichitro-moe-github-source-7cccbd8.zip
```

To restore the complete committed Git repository:

```bash
git clone boichitro-moe-github-history-7cccbd8.bundle boichitro-moe-github
```

## GitHub note

Upload only `boichitro-moe-github/` or the source ZIP to GitHub. Do not upload
the full backup archive: it contains multi-gigabyte datasets and checkpoints
and must remain in local or external backup storage.
