# SBFNav audit artifacts

These files accompany the independent, paper-based SBFNav reimplementation in
this repository. They are not artifacts from an official SBFNav release.

The repository tracks the reasonably sized, human-auditable outputs requested
for reproducing the current epoch-7 inspection:

- `formal_epochs_1_5/`: supervised training logs and provenance for epochs 1–5.
- `formal_epochs_6_7/`: supervised training logs and provenance for epochs 6–7.
- `formal_resume_e7_partial/`: a point-in-time snapshot of the subsequently
  resumed formal run. It is intentionally labelled partial and must not be
  interpreted as its final result.
- `epoch7_full_validation/`: complete Val-Seen and Val-Unseen metrics,
  one-record-per-episode prediction JSONL files, logs, and provenance.
- `epoch7_candidate_analysis/`: complete Val-Seen and Val-Unseen candidate
  prediction JSONL files, oracle-coverage metrics, logs, and provenance.
- `epoch7_case_visualizations/`: five representative successes and five
  failures from each validation split, plus selection manifests.
- `e7_valunseen_closed_loop_diagnostic_256/`: complete per-step predictions
  and the four-way SR/SPL failure audit for the first 256 Val-Unseen records.
- `SHA256SUMS`: content hashes for every artifact except this index and the
  checksum file itself.

Model checkpoints, learned weights, feature caches, source snapshots, datasets,
and other large/redundant files are deliberately excluded. Absolute data and
run paths found in provenance files identify the machine inputs used; those
inputs are not part of this repository. Test-Unseen predictions are absent and
the tracked effective configurations keep `allow_test_unseen: false`.

Validate the published files from the repository root with:

```bash
sha256sum --check artifacts/sbfnav/SHA256SUMS
```

The epoch-7 validation checkpoint itself is external. Its SHA-256 is
`ba8b48217ed948380ee8939df89b9eaeb8352e79f1bc5f9d0acc3a493e645564`;
the exact metrics and experiment interpretation are documented in
`docs/sbfnav_results.md`.
