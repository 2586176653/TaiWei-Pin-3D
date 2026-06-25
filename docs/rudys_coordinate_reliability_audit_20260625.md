# Rudys Coordinate Reliability Audit, 2026-06-25

This snapshot records the current Rudys-aware partitioning work for `nangate45_3D/swerv_wrapper`.

## Uploaded Snapshot

The full work snapshot is stored as a base64-encoded zip file:

```text
artifacts/rudys_coordinate_reliability_audit_75d428d.zip.b64
```

Decode it with:

```bash
base64 -d artifacts/rudys_coordinate_reliability_audit_75d428d.zip.b64 > rudys_coordinate_reliability_audit_75d428d.zip
unzip rudys_coordinate_reliability_audit_75d428d.zip
```

On PowerShell:

```powershell
$b64 = Get-Content artifacts/rudys_coordinate_reliability_audit_75d428d.zip.b64 -Raw
[IO.File]::WriteAllBytes("rudys_coordinate_reliability_audit_75d428d.zip", [Convert]::FromBase64String($b64))
Expand-Archive .\rudys_coordinate_reliability_audit_75d428d.zip
```

## Scope

The snapshot contains:

- `scripts_openroad/collect_rudys_bo_metrics.py`
- `scripts_openroad/select_partition_by_proxy.py`
- `run_rudys_bo_trial_2dpart.sh`
- `run_swerv_wrapper_best_macro_rudys.sh`
- `coord_rudys_reliability_audit.py`
- `coord_rudys_reliability_audit_swerv_fullppa.csv`
- `coord_rudys_reliability_audit_swerv_fullppa.json`
- this documentation file

## Main Finding

The coordinate audit compares:

- A: partition-stage coordinates from `2_2_floorplan_io.def`
- B: post macro-placement coordinates from `2_5_place_macro_upper.def` and `2_5_place_macro_bottom.def`
- C: OpenROAD final coordinates from `6_final.def`

For completed `swerv_wrapper` PPA runs, A-to-C macro displacement is large and A-to-C Rudys correlation is low. This means the partition-stage coordinate Rudys signal should be treated as a weak prior, not as a strong primary objective.

Observed macro-related Rudys Pearson correlation from A to C:

| Variant | Pearson | Top-k Hotspot Overlap |
|---|---:|---:|
| `openroad` | 0.072 | 22.7% |
| `rudys_bo_swerv_20trial_a_t003` | 0.038 | 28.0% |
| `rudys_bo_swerv_20trial_a_t015` | 0.041 | 26.1% |
| `rudys_bo_swerv_20trial_a_t016` | 0.043 | 20.2% |

## Hashes

Local source snapshot commit:

```text
75d428d7e83d330c1f494511dcebd69589173f77
```

Zip SHA256:

```text
ED01B9DBA4B113531A8B7D705070652FB8297BDAF12B995E1BBB117CDCD1A850
```

Base64 zip text SHA256:

```text
8A34B59537C554909AB91B25759AD7BEC568CF3FF852E8C59AEEC7AF0F907751
```

## Method Note

The C coordinate set is OpenROAD `6_final.def`. A Cadence final DEF was not found in the current project logs, so this audit does not claim Cadence-final coordinate correlation.
