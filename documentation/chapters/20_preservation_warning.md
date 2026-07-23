# Preservation, deletion consequences, and recovery limits

The documentation folder preserves descriptions and small evidence, not the executable state of the full experiment. Deleting the original workspace and the complete backup will permanently remove raw and processed Parquet data, packed pretraining blocks, `.pt` checkpoints, optimizer state, caches, full predictions, and original archives unless they exist elsewhere.

GitHub normally contains the compact source repository and frozen tokenizer, not the 91 GiB research state. A PDF, DOCX, or ZIP cannot reproduce training without the omitted artifacts.

Before deleting the original or backup, retain at least one independent copy of the complete archive on an external disk or trusted object store. If deletion proceeds anyway, this documentation will preserve scientific context, tables, figures, configurations, code descriptions, hashes recorded by the experiment, run metadata, and a complete former-file inventory—but not recoverable model weights or corpora.
