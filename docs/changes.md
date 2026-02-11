# What's New

## 0.14.1 (February 2, 2026)

### 🐛 Bug Fixes

* Change smoothing surf from 'sphere' to 'midthickness' by @joey-scanga in https://github.com/PennLINC/xcp_d/pull/1565

## New Contributors

* @joey-scanga made their first contribution in https://github.com/PennLINC/xcp_d/pull/1565

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.14.0...0.14.1


## 0.14.0 (January 6, 2026)

### 🎉 Exciting New Features

* Generate pconn.niis from concatenated CIFTI time series by @tsalo in https://github.com/PennLINC/xcp_d/pull/1541
* Add parameter to control run-wise correlations by @tsalo in https://github.com/PennLINC/xcp_d/pull/1544
* Add --skip parameter to skip outputs during postprocessing by @psychelzh in https://github.com/PennLINC/xcp_d/pull/1535

### Other Changes

* Remove information on differences between modes by @tsalo in https://github.com/PennLINC/xcp_d/pull/1542
* Support BIDS proc-<label> entity in xcp_d path patterns by @magicbrainresearch in https://github.com/PennLINC/xcp_d/pull/1546

### New Contributors

* @magicbrainresearch made their first contribution in https://github.com/PennLINC/xcp_d/pull/1546

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.13.0...0.14.0


## 0.13.0 (November 11, 2025)

### 🎉 Exciting New Features

* Assume T1w and T2w are in same space if using fMRIPrep derivatives by @tsalo in https://github.com/PennLINC/xcp_d/pull/1514

### 🐛 Bug Fixes

* Estimate memory requirements more accurately by @tsalo in https://github.com/PennLINC/xcp_d/pull/1513
* Remove colorbar from brainsprite by @tsalo in https://github.com/PennLINC/xcp_d/pull/1529
* Account for longitudinal studies with a single anatomical session by @tsalo in https://github.com/PennLINC/xcp_d/pull/1534

### Other Changes

* Change metavar for `--mem-mb` parameter by @tsalo in https://github.com/PennLINC/xcp_d/pull/1510
* Update citation for MIDB atlas by @tsalo in https://github.com/PennLINC/xcp_d/pull/1523

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.12.0...0.13.0


## 0.12.0 (September 17, 2025)

### 🎉 Exciting New Features

* Use MNIInfant transforms from templateflow by @tsalo in https://github.com/PennLINC/xcp_d/pull/1487

### 🐛 Bug Fixes

* Change memory argument destination to `memory_gb` by @j1c in https://github.com/PennLINC/xcp_d/pull/1490
* Fix WM and CSF component selection in acompcor and acompcor_gsr by @roeysc in https://github.com/PennLINC/xcp_d/pull/1498
* Fix brainsprite when `--warp-surfaces-native2std` is disabled by @tsalo in https://github.com/PennLINC/xcp_d/pull/1501

### Other Changes

* Use fsLR-registered spheres when available by @tsalo in https://github.com/PennLINC/xcp_d/pull/1486
* Add hash entity to outputs by @tsalo in https://github.com/PennLINC/xcp_d/pull/1496

### New Contributors

* @j1c made their first contribution in https://github.com/PennLINC/xcp_d/pull/1490
* @roeysc made their first contribution in https://github.com/PennLINC/xcp_d/pull/1498

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.11.1...0.12.0


## 0.11.1 (July 28, 2025)

### 🐛 Bug Fixes

* Fix figure collection for longitudinal, but not session-wise, reports by @tsalo in https://github.com/PennLINC/xcp_d/pull/1482

### Other Changes

* Drop dir and run entities from single-run tasks by @tsalo in https://github.com/PennLINC/xcp_d/pull/1472
* Check DatasetType for "derivative" or "derivatives" by @tsalo in https://github.com/PennLINC/xcp_d/pull/1481

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.11.0...0.11.1


## 0.11.0 (July 15, 2025)

### 🛠 Breaking Changes

* Use `mem_mb` instead of `mem_gb` by @smeisler in https://github.com/PennLINC/xcp_d/pull/1417

### 🎉 Exciting New Features

* Add `--report-output-level` parameter by @tsalo in https://github.com/PennLINC/xcp_d/pull/1436
* Collect and apply transforms between native anatomical spaces by @tsalo in https://github.com/PennLINC/xcp_d/pull/1437

### 🐛 Bug Fixes

* Ensure T1w-only processing does not collect T2w by @tsalo in https://github.com/PennLINC/xcp_d/pull/1433
* Support session-wise anatomical processing by @tsalo in https://github.com/PennLINC/xcp_d/pull/1438
* Fix session-wise collection by @tsalo in https://github.com/PennLINC/xcp_d/pull/1439
* Fix session-wise collection of surface files by @tsalo in https://github.com/PennLINC/xcp_d/pull/1440
* Make workflow name unique for session-wise processing by @tsalo in https://github.com/PennLINC/xcp_d/pull/1441
* Add nifti brain mask to hcpya ingression by @smeisler in https://github.com/PennLINC/xcp_d/pull/1411
* Write out filtered motion parameters when dummy-scans is not zero by @tsalo in https://github.com/PennLINC/xcp_d/pull/1446
* Ingress BOLD brain mask in ABCD-BIDS and HCP processing by @tsalo in https://github.com/PennLINC/xcp_d/pull/1473

### Other Changes

* Update note about HCP compatibility by @tsalo in https://github.com/PennLINC/xcp_d/pull/1465

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.10.7...0.11.0


## 0.10.7 (April 8, 2025)

### 🎉 Exciting New Features

* Use masked functional and anatomical images for executive summary by @tsalo in https://github.com/PennLINC/xcp_d/pull/1425

### 🐛 Bug Fixes

* Warp fsLR surfaces to NLin6 for brainsprite by @tsalo in https://github.com/PennLINC/xcp_d/pull/1419
* Warp anat mask to anat space for coreg QC by @tsalo in https://github.com/PennLINC/xcp_d/pull/1423
* Remove background label in ConnectPlot by @tsalo in https://github.com/PennLINC/xcp_d/pull/1424

### Other Changes

* Run vertex normalization on output surfaces by @tsalo in https://github.com/PennLINC/xcp_d/pull/1420

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.10.6...0.10.7


## 0.10.6 (February 21, 2025)

### 🎉 Exciting New Features

* Add session-id parameter by @tsalo in https://github.com/PennLINC/xcp_d/pull/1379
* Add colorbar to correlation matrix plot by @tsalo in https://github.com/PennLINC/xcp_d/pull/1401

### 🐛 Bug Fixes

* Support `--abcc-qc` with `--file-format nifti` by @tsalo in https://github.com/PennLINC/xcp_d/pull/1388
* Do not attempt to concatenate runs when no runs in entity set are processed by @tsalo in https://github.com/PennLINC/xcp_d/pull/1398
* Make calculations robust to NaNs by @tsalo in https://github.com/PennLINC/xcp_d/pull/1399
* Fix atlas names in concatenated parcellated output files by @tsalo in https://github.com/PennLINC/xcp_d/pull/1407

### Other Changes

* Only estimate head radius if necessary by @tsalo in https://github.com/PennLINC/xcp_d/pull/1381
* Simplify the brainsprite workflow setup by @tsalo in https://github.com/PennLINC/xcp_d/pull/1384
* Parallelize image denoising by @mattcieslak in https://github.com/PennLINC/xcp_d/pull/1387
* Apply consistency across XCP-D quick-start instructions by @singlesp in https://github.com/PennLINC/xcp_d/pull/1382
* Test PlotSlicesForBrainSprite by @tsalo in https://github.com/PennLINC/xcp_d/pull/1393
* Update mode documentation by @B-Sevchik in https://github.com/PennLINC/xcp_d/pull/1392
* Add comments and debug logging to init_denoise_bold_wf by @tientong98 in https://github.com/PennLINC/xcp_d/pull/1391
* Add more OMP by @mattcieslak in https://github.com/PennLINC/xcp_d/pull/1386
* Note that automatic head radius estimation gives higher values than 50 by @tsalo in https://github.com/PennLINC/xcp_d/pull/1397
* Add file lock to avoid concurrent edits by @tientong98 in https://github.com/PennLINC/xcp_d/pull/1403
* Clarify mounting in apptainer instructions by @singlesp in https://github.com/PennLINC/xcp_d/pull/1394

### New Contributors

* @singlesp made their first contribution in https://github.com/PennLINC/xcp_d/pull/1382
* @B-Sevchik made their first contribution in https://github.com/PennLINC/xcp_d/pull/1392
* @tientong98 made their first contribution in https://github.com/PennLINC/xcp_d/pull/1391

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.10.5...0.10.6


## 0.10.5 (January 27, 2025)

### 🎉 Exciting New Features

* Use trimesh to generate brainsprite by @tsalo in https://github.com/PennLINC/xcp_d/pull/1368

### 🐛 Bug Fixes

* Stop autoescaping the boilerplate by @mattcieslak in https://github.com/PennLINC/xcp_d/pull/1369

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.10.4...0.10.5


## 0.10.4 (January 23, 2025)

This release makes the brainsprite work with composite transforms from Nibabies.

### 🐛 Bug Fixes

* Support multi-step composite transforms from Nibabies by @tsalo in https://github.com/PennLINC/xcp_d/pull/1365

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.10.3...0.10.4


## 0.10.3 (January 20, 2025)

This release fixes a testing bug that prevented 0.10.2 from deploying.

### 🐛 Bug Fixes

* Allow `sub-` prefix in participant labels by @tsalo in https://github.com/PennLINC/xcp_d/pull/1347
* Downgrade connectome workbench version by @tsalo in https://github.com/PennLINC/xcp_d/pull/1358
* Fix resource class of CIFTI tests by @tsalo in https://github.com/PennLINC/xcp_d/pull/1362

### Other Changes

* Clarify that warped surfaces can be overlaid on the MNI152NLin6Asym template by @tsalo in https://github.com/PennLINC/xcp_d/pull/1349
* Run ruff by @tsalo in https://github.com/PennLINC/xcp_d/pull/1357

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.10.1...0.10.3


## 0.10.2 (January 17, 2025)

A patch release.

**WARNING**: THIS RELEASE DID NOT DEPLOY. PLEASE USE 0.10.3.

### 🐛 Bug Fixes

* Downgrade connectome workbench version by @tsalo in https://github.com/PennLINC/xcp_d/pull/1358
* Allow `sub-` prefix in participant labels by @tsalo in https://github.com/PennLINC/xcp_d/pull/1347

### Other Changes

* Clarify that warped surfaces can be overlaid on the MNI152NLin6Asym template by @tsalo in https://github.com/PennLINC/xcp_d/pull/1349
* Run ruff by @tsalo in https://github.com/PennLINC/xcp_d/pull/1357

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.10.1...0.10.2


## 0.10.1 (November 21, 2024)

A patch release fixing bugs stemming from 0.10.0.

### 🐛 Bug Fixes

* Write out correlations when `--create-matrices` doesn't include "all" by @tsalo in https://github.com/PennLINC/xcp_d/pull/1326
* Use confounds file with non-steady-state columns to identify dummy volumes by @tsalo in https://github.com/PennLINC/xcp_d/pull/1330

### Other Changes

* Document the nichart mode by @tsalo in https://github.com/PennLINC/xcp_d/pull/1325
* Update indexed-gzip requirement from ~=1.8.7 to ~=1.9.1 by @dependabot in https://github.com/PennLINC/xcp_d/pull/1327

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.10.0...0.10.1


## 0.10.0 (November 17, 2024)

This backwards-incompatible release brings XCP-D further in line with the fMRIPost workflows in the NiPreps ecosystem.
We have added the ability to provide custom confound sets defined as YAML config files,
ingress derivatives from fMRIPost BIDS Apps (for example, fMRIPost-AROMA),
and provide external atlases organized as BIDS-Atlas datasets.

Please note that the BIDS-Atlas extension proposal has not been merged into BIDS yet,
so the expected organization of these external atlas datasets may change over time.

### 🛠 Breaking Changes

* Replace restricted confound sets with config files by @tsalo in https://github.com/PennLINC/xcp_d/pull/1255
* Rename `--derivatives` parameter to `--datasets` by @tsalo in https://github.com/PennLINC/xcp_d/pull/1276
* Accept external atlases by searching BIDS-Atlas datasets by @tsalo in https://github.com/PennLINC/xcp_d/pull/1265
* Move hemi entity before space by @tsalo in https://github.com/PennLINC/xcp_d/pull/1287

### 🎉 Exciting New Features

* Remove Freesurfer node from surface normalization workflow by @tsalo in https://github.com/PennLINC/xcp_d/pull/1273
* Track Sources with BIDSURI interface by @tsalo in https://github.com/PennLINC/xcp_d/pull/1283
* Add nichart mode by @tsalo in https://github.com/PennLINC/xcp_d/pull/1320

### 🐛 Bug Fixes

* Patch task_id into bids_filters by @tsalo in https://github.com/PennLINC/xcp_d/pull/1271
* Fix NIfTI-surface connection by @tsalo in https://github.com/PennLINC/xcp_d/pull/1284
* Update build image version by @tsalo in https://github.com/PennLINC/xcp_d/pull/1293
* Fix bug with surfaces and external atlases by @tsalo in https://github.com/PennLINC/xcp_d/pull/1304
* Apply notch filter appropriate number of times based on `--motion-filter-order` by @tsalo in https://github.com/PennLINC/xcp_d/pull/1300
* Fix ingression of standard-space mesh files by @tsalo in https://github.com/PennLINC/xcp_d/pull/1309
* Fix init_qc_report_wf connections when linc-qc and abcc-qc are disabled by @tsalo in https://github.com/PennLINC/xcp_d/pull/1314

### Other Changes

* Don't rerun alff if num threads changes by @mattcieslak in https://github.com/PennLINC/xcp_d/pull/1263
* Remove xcp_d_dir and clean datasinks by @tsalo in https://github.com/PennLINC/xcp_d/pull/1262
* Add dataset links to config object by @tsalo in https://github.com/PennLINC/xcp_d/pull/1266
* Document minimal required inputs by @tsalo in https://github.com/PennLINC/xcp_d/pull/1270
* Correct documentation regarding Freesurfer by @tsalo in https://github.com/PennLINC/xcp_d/pull/1277
* Improve custom confound documentation by @tsalo in https://github.com/PennLINC/xcp_d/pull/1280
* Improve documentation of new features by @tsalo in https://github.com/PennLINC/xcp_d/pull/1281
* Use ruff for linting by @tsalo in https://github.com/PennLINC/xcp_d/pull/1312
* Fix links to documentation by @tsalo in https://github.com/PennLINC/xcp_d/pull/1323

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.9.1...0.10.0


## 0.9.1 (September 13, 2024)

This release improves performance with multi-core processing.

### Other Changes

* Remove n_procs for nodes that do not use multiple cores by @tsalo in https://github.com/PennLINC/xcp_d/pull/1256
* Remove unused data files by @tsalo in https://github.com/PennLINC/xcp_d/pull/1259
* Parallelize ALFF by @tsalo in https://github.com/PennLINC/xcp_d/pull/1258
* Replace internal data module with acres by @tsalo in https://github.com/PennLINC/xcp_d/pull/1260

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.9.0...0.9.1


## 0.9.0 (September 6, 2024)

This backwards-incompatible release makes two important breaking changes.
First, we changed the default file format for 'linc' mode from NIfTI to CIFTI.
Second, XCP-D no longer ingresses and warps the anatomical tissue segmentation file (`*_dseg.nii.gz`),
since no one was using it and it's easy enough to collect separately.

Additionally, this release fixes a bug with the collection of sphere files needed to warp fsnative surfaces to fsLR space,
which should only affect derivatives from fMRIPrep and Nibabies versions >= 24.0.0
(i.e., versions that include `space-fsnative` in the spheres' filenames).

### 🛠 Breaking Changes

* Change linc mode default file format to CIFTI by @tsalo in https://github.com/PennLINC/xcp_d/pull/1248
* Stop ingressing anatomical-space tissue segmentation file by @tsalo in https://github.com/PennLINC/xcp_d/pull/1190

### 🎉 Exciting New Features

* Internally support voxel-wise confounds by @tsalo in https://github.com/PennLINC/xcp_d/pull/1244

### 🐛 Bug Fixes

* Fix subject sphere query by @tsalo in https://github.com/PennLINC/xcp_d/pull/1249

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.8.3...0.9.0


## 0.8.3 (August 23, 2024)

This patch release fixes a bug introduced in 0.8.1.

### 🐛 Bug Fixes

* Fix collection in case of derivatives with T1w and T2w but T1w is primary by @tsalo in https://github.com/PennLINC/xcp_d/pull/1240

### Other Changes

* Document how to request a new mode by @tsalo in https://github.com/PennLINC/xcp_d/pull/1238

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.8.2...0.8.3


## 0.8.2 (August 20, 2024)

This release adds the "none" option for the mode parameter.

### 🎉 Exciting New Features

* Implement "none" mode by @tsalo in https://github.com/PennLINC/xcp_d/pull/1232

### 🐛 Bug Fixes

* Account for floating point issues in ALFF time array creation by @tsalo in https://github.com/PennLINC/xcp_d/pull/1236

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.8.1...0.8.2


## 0.8.1 (August 16, 2024)

Patch release for HBCD to support Nibabies runs with both T1w and T2w data.

### 🐛 Bug Fixes

* Explicitly check for `-p none` and `--custom-confounds` by @tsalo in https://github.com/PennLINC/xcp_d/pull/1222

### Other Changes

* Restructure XCP-D to better match fMRIPrep by @tsalo in https://github.com/PennLINC/xcp_d/pull/1225
* Document the apptainer version requirements by @tsalo in https://github.com/PennLINC/xcp_d/pull/1233
* Make collect_data work for T1w+T2w nibabies derivatives by @tsalo in https://github.com/PennLINC/xcp_d/pull/1234

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.8.0...0.8.1


## 0.8.0 (July 31, 2024)

This is a backwards-incompatible release that will be used to postprocess the beta version of the first release of the HBCD dataset.
There are two major breaking changes.
First, there is a new required `--mode` parameter that automatically defines sets of parameters to match recommendations from different studies/labs.
Each parameter that the `--mode` parameter controls can be overridden, so users should be able to reproduce their original settings from previous versions of XCP-D.
Second, XCP-D now loads certain sphere files from the BIDS derivatives, rather than non-BIDS Freesurfer or MCRIBS folders.
This means that the anatomical workflow (`--warp-surfaces-native2std`) will only work for fMRIPrep versions >= 23.1.0 and Nibabies versions >= 23.1.0.
The rest of XCP-D should still work for versions of fMRIPrep and Nibabies that were supported previously, but we cannot guarantee this.

### 🛠 Breaking Changes

* Add required `--mode` parameter to define sets of parameters by @tsalo in https://github.com/PennLINC/xcp_d/pull/1109
* Load spheres for anatomical workflow from TemplateFlow and BIDS derivatives by @tsalo in https://github.com/PennLINC/xcp_d/pull/1207

### 🎉 Exciting New Features

* Add discretized MID-B and Myers-Labonte atlases by @tsalo in https://github.com/PennLINC/xcp_d/pull/1192
* Plot CIFTI maps on subject-specific surfaces when available by @tsalo in https://github.com/PennLINC/xcp_d/pull/1208
* Allow bids-filter-file terms for mesh and morphometry files by @tsalo in https://github.com/PennLINC/xcp_d/pull/1210

### 🐛 Bug Fixes

* Generate session-wise executive summaries by @tsalo in https://github.com/PennLINC/xcp_d/pull/1202
* Look for space entity in T2w-only anatomical inputs by @tsalo in https://github.com/PennLINC/xcp_d/pull/1203
* Use standard-space anatomical brain mask by @tsalo in https://github.com/PennLINC/xcp_d/pull/1204
* Drop volumetric Myers-Labonte atlas by @tsalo in https://github.com/PennLINC/xcp_d/pull/1212
* Enable linc-qc for abcd and hbcd modes by default by @tsalo in https://github.com/PennLINC/xcp_d/pull/1220

### Other Changes

* Document anatomical workflow in more detail by @tsalo in https://github.com/PennLINC/xcp_d/pull/1191
* Fix up documentation by @tsalo in https://github.com/PennLINC/xcp_d/pull/1209
* Reorganize QC interfaces by @tsalo in https://github.com/PennLINC/xcp_d/pull/1215
* Allow `space-fsaverage` for sphere files by @tsalo in https://github.com/PennLINC/xcp_d/pull/1217
* Continue to improve documentation of anatomical workflow by @tsalo in https://github.com/PennLINC/xcp_d/pull/1216

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.7.5...0.8.0


## 0.7.5 (June 28, 2024)

This is a patch release that fixes how the correlation matrix plot is generated when users select only one atlas for parcellation.
The bug was likely introduced in 0.7.4 and only affects runs with a single atlas.

### 🐛 Bug Fixes

* Fix ConnectPlot by @tsalo in https://github.com/PennLINC/xcp_d/pull/1188

### Other Changes

* Refactor CIFTI parcellation and connectivity workflows to use wb_command by @tsalo in https://github.com/PennLINC/xcp_d/pull/1174
* Replace internal smoothing spheres with ones from TemplateFlow by @tsalo in https://github.com/PennLINC/xcp_d/pull/1160

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.7.4...0.7.5


## 0.7.4 (May 29, 2024)

This is a patch release that fixes two important bugs.
The two bugs are:
(1) band-pass filter values were not respected in versions 0.7.1 - 0.7.3 (they were hardcoded to 0.01 - 0.1) and
(2) when processing CIFTI files, parcellated ReHo values in TSVs were not correct, due to a problem with how we were reconstructing CIFTI ReHo files.
The dense CIFTI files should still be usable though.

### 🎉 Exciting New Features

* Make ConnectPlot robust to chosen atlases by @tsalo in https://github.com/PennLINC/xcp_d/pull/1161

### 🐛 Bug Fixes

* Fix band-pass filter settings in Config by @tsalo in https://github.com/PennLINC/xcp_d/pull/1172
* Allow ALFF to work with low-pass or high-pass filters by @tsalo in https://github.com/PennLINC/xcp_d/pull/1176
* Use CiftiCreateDenseFromTemplate for CIFTI ReHo by @tsalo in https://github.com/PennLINC/xcp_d/pull/1175

### Other Changes

* Use nireports for HTML report generation by @tsalo in https://github.com/PennLINC/xcp_d/pull/1169


**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.7.3...0.7.4

## 0.7.3 (April 29, 2024)

Small patch release for manuscript's executive summary.

**There is a known bug with the band-pass filter settings in this release. Upper and lower band-pass values were hardcoded to 0.01 - 0.1.**

### 🐛 Bug Fixes

* Drop concatenated rest-as-run section from executive summary by @tsalo in https://github.com/PennLINC/xcp_d/pull/1156

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.7.2...0.7.3


## 0.7.2 (April 24, 2024)

This is a patch release fixing small bugs in 0.7.1.

**There is a known bug with the band-pass filter settings in this release. Upper and lower band-pass values were hardcoded to 0.01 - 0.1.**

### 🎉 Exciting New Features

* Make `GeneratedBy` in preprocessing derivatives' `dataset_description.json` optional by @tsalo in https://github.com/PennLINC/xcp_d/pull/1151

### 🐛 Bug Fixes

* Fix exit code bug in `cli.run.main()` by @tsalo in https://github.com/PennLINC/xcp_d/pull/1152

### Other Changes

* Replace pkgrf with load_data by @tsalo in https://github.com/PennLINC/xcp_d/pull/1147
* Update to new build image (v0.0.12) by @tsalo in https://github.com/PennLINC/xcp_d/pull/1153

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.7.1...0.7.2


## 0.7.1 (April 19, 2024)

This release prepares for the XCP-D manuscript.

**There is a known bug with the band-pass filter settings in this release. Upper and lower band-pass values were hardcoded to 0.01 - 0.1.**

### 🛠 Breaking Changes

* Change default `--min-time` from 100 to 240 by @tsalo in https://github.com/PennLINC/xcp_d/pull/1115

### 🎉 Exciting New Features

* Add `--bids-database-dir` parameter by @tsalo in https://github.com/PennLINC/xcp_d/pull/1116
* Ignore subcortical figures and plot FOV center of reference brains in executive summary by @tsalo in https://github.com/PennLINC/xcp_d/pull/1145

### 🐛 Bug Fixes

* Modify metadata and boilerplate to reflect updated motion filter parameters by @tsalo in https://github.com/PennLINC/xcp_d/pull/1114
* Add "subject" to native-space surface query by @tsalo in https://github.com/PennLINC/xcp_d/pull/1118
* Stop using `config.nipype.memory_gb` as memory limit in workflow nodes by @tsalo in https://github.com/PennLINC/xcp_d/pull/1122
* Pin dependencies to fix RTD build by @tsalo in https://github.com/PennLINC/xcp_d/pull/1136
* Fix adjacency matrix for CIFTI ReHo by @tsalo in https://github.com/PennLINC/xcp_d/pull/1120
* Use run-specific cwd for UK Biobank ingression interface by @tsalo in https://github.com/PennLINC/xcp_d/pull/1137
* Add third config to BIDSLayout that defines cohort entity by @tsalo in https://github.com/PennLINC/xcp_d/pull/1143
* Use standardized DVARS from Nipype by @tsalo in https://github.com/PennLINC/xcp_d/pull/1135

### Other Changes

* Adopt Nipreps-style Config object by @tsalo in https://github.com/PennLINC/xcp_d/pull/1040
* Use pytest-env to capture warnings on CI runs by @tsalo in https://github.com/PennLINC/xcp_d/pull/1107
* Convert coverage values to float32 to address pandas warning by @tsalo in https://github.com/PennLINC/xcp_d/pull/1112
* Only describe the atlases selected by the user by @tsalo in https://github.com/PennLINC/xcp_d/pull/1126
* Fix API documentation by @tsalo in https://github.com/PennLINC/xcp_d/pull/1144
* Improve denoising tests by @tsalo in https://github.com/PennLINC/xcp_d/pull/1146

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.7.0...0.7.1


## 0.7.0 (March 22, 2024)

This is a large, backwards-incompatible release.
The changes in this release mostly stem from reviewer comments on the XCP-D manuscript.

I expect to release 0.7.1, which will add the NiPreps Config object, very soon.

### 🛠 Breaking Changes

* [DCAN] Remove `--dcan-qc` parameter by @tsalo in https://github.com/PennLINC/xcp_d/pull/1096
* Bring outputs up to date with BEPs 11, 12, 17, and 38 by @tsalo in https://github.com/PennLINC/xcp_d/pull/1065
* Stop appending "xcp_d" to output directory by @tsalo in https://github.com/PennLINC/xcp_d/pull/1061

### 🎉 Exciting New Features

* Create reformatted BrainSwipes figures for HBCD QC by @tsalo in https://github.com/PennLINC/xcp_d/pull/1091

### Other Changes

* Add information about ReHo to read-the-docs by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/1097
* Update landing page figure by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/1099
* Add band over censored volumes in executive summary carpet plots by @tsalo in https://github.com/PennLINC/xcp_d/pull/1077
* Fix image link by @psychelzh in https://github.com/PennLINC/xcp_d/pull/1100

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.6.4...0.7.0


## 0.6.4 (March 19, 2024)

### 🎉 Exciting New Features

* Replace `bids:xcp_d:` BIDS URI with `bids::` by @tsalo in https://github.com/PennLINC/xcp_d/pull/1088

### 👎 Deprecations

* [DCAN] Enable DCAN QC by default by @tsalo in https://github.com/PennLINC/xcp_d/pull/1086

### 🐛 Bug Fixes

* Raise error if atlas affines don't match across participants by @tsalo in https://github.com/PennLINC/xcp_d/pull/1075
* Fix query error report in `collect_mesh_data` by @tsalo in https://github.com/PennLINC/xcp_d/pull/1092
* Adopt Lindquist-compliant denoising method by @tsalo in https://github.com/PennLINC/xcp_d/pull/1087

### Other Changes

* Update changelog for 0.6.3 by @tsalo in https://github.com/PennLINC/xcp_d/pull/1079

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.6.3...0.6.4


## 0.6.3 (March 8, 2024)

### 🎉 Exciting New Features

* Make "DatasetType" key recommended, not required by @tsalo in https://github.com/PennLINC/xcp_d/pull/1068

### 🐛 Bug Fixes

* Do not warn if `fd-thresh` and `min-time` are both set to 0 by @tsalo in https://github.com/PennLINC/xcp_d/pull/1071
* Use `copy_atlas` to write out atlas tsv and json files by @tsalo in https://github.com/PennLINC/xcp_d/pull/1073
* Raise error if FS_LICENSE environment variable is not an existing file by @tsalo in https://github.com/PennLINC/xcp_d/pull/1072
* Generate T2w executive summary plots when T1w is also available by @tsalo in https://github.com/PennLINC/xcp_d/pull/1078

### Other Changes

* Add release notes for releases 0.6.1 and 0.6.2 by @tsalo in https://github.com/PennLINC/xcp_d/pull/1070

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.6.2...0.6.3


## 0.6.2 (February 27, 2024)

### 🐛 Bug Fixes

* Raise interpretable error when no BOLD runs survive minimum-time threshold by @tsalo in https://github.com/PennLINC/xcp_d/pull/1048
* Address bug with T2w-only processing by @tsalo in https://github.com/PennLINC/xcp_d/pull/1060
* Use strict file-matching in collect_run_data by @tsalo in https://github.com/PennLINC/xcp_d/pull/1063
* Fix potential atlas file race condition by @tsalo in https://github.com/PennLINC/xcp_d/pull/1066

### Other Changes

* Make description of `smoothing` parameter clearer when set to `0` by @psychelzh in https://github.com/PennLINC/xcp_d/pull/1056

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.6.1...0.6.2


## 0.6.1 (February 12, 2024)

### 🛠 Breaking Changes

* Support ALFF for censored data by @tsalo in https://github.com/PennLINC/xcp_d/pull/1020
* Move atlases into a subfolder by @tsalo in https://github.com/PennLINC/xcp_d/pull/1024

### 🎉 Exciting New Features

* Add colorbar to ALFF and ReHo plots by @tsalo in https://github.com/PennLINC/xcp_d/pull/1023
* Add options for UK Biobank data by @tsalo in https://github.com/PennLINC/xcp_d/pull/1022
* Add options to subset atlases by @tsalo in https://github.com/PennLINC/xcp_d/pull/1034
* Support MCRIBS derivatives by @tsalo in https://github.com/PennLINC/xcp_d/pull/1029
* Add `--stop-on-first-crash` option to command-line interface by @tsalo in https://github.com/PennLINC/xcp_d/pull/1044

### 🐛 Bug Fixes

* Rescale ALFF based on original BOLD standard deviation by @tsalo in https://github.com/PennLINC/xcp_d/pull/1033
* Drop `dir` entity from concatenated filenames by @tsalo in https://github.com/PennLINC/xcp_d/pull/1037
* Fix dscalar CIFTI generation by @tsalo in https://github.com/PennLINC/xcp_d/pull/1036
* Fix TR in denoised NIfTI headers by @tsalo in https://github.com/PennLINC/xcp_d/pull/1038

### Other Changes

* Add dependabot Action by @tsalo in https://github.com/PennLINC/xcp_d/pull/1006
* Update to Python 3.10 by @tsalo in https://github.com/PennLINC/xcp_d/pull/1016
* Update docker image link by @psychelzh in https://github.com/PennLINC/xcp_d/pull/1045

### New Contributors

* @dependabot made their first contribution in https://github.com/PennLINC/xcp_d/pull/1007
* @psychelzh made their first contribution in https://github.com/PennLINC/xcp_d/pull/1045

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.6.0...0.6.1


## 0.6.0 (December 8, 2023)

### 🛠 Breaking Changes
* Replace Schaefer atlases with AtlasPack by @tsalo in https://github.com/PennLINC/xcp_d/pull/928
* Update the QC metrics by @tsalo in https://github.com/PennLINC/xcp_d/pull/958
* Track `Sources` for outputs with BIDS URIs by @tsalo in https://github.com/PennLINC/xcp_d/pull/966
* Replace AtlasPack 0.0.5 atlases with AtlasPack 0.1.0 atlases by @tsalo in https://github.com/PennLINC/xcp_d/pull/991

### 🎉 Exciting New Features
* Split HCP tasks into task and run entities by @tsalo in https://github.com/PennLINC/xcp_d/pull/952
* Concatenate across directions as well as runs by @tsalo in https://github.com/PennLINC/xcp_d/pull/965
* Retain orthogonalized confounds by @tsalo in https://github.com/PennLINC/xcp_d/pull/975
* Improve connectivity plots by @tsalo in https://github.com/PennLINC/xcp_d/pull/988
* Produce correlation matrices for concatenated time series by @tsalo in https://github.com/PennLINC/xcp_d/pull/990

### 🐛 Bug Fixes
* Fix resting-state plots in executive summary by @tsalo in https://github.com/PennLINC/xcp_d/pull/941
* Load T1w-to-standard transform to same space as volumetric BOLD scan by @tsalo in https://github.com/PennLINC/xcp_d/pull/926
* Pin Nilearn version by @tsalo in https://github.com/PennLINC/xcp_d/pull/955
* Don't interpolate volumes at beginning/end of run by @tsalo in https://github.com/PennLINC/xcp_d/pull/950
* Look for flirtbbr figure by @tsalo in https://github.com/PennLINC/xcp_d/pull/980
* Make registration figure in executive summary optional by @tsalo in https://github.com/PennLINC/xcp_d/pull/981
* Don't convert CIFTIs to int16 by @tsalo in https://github.com/PennLINC/xcp_d/pull/979
* Store temporary files in working directory by @tsalo in https://github.com/PennLINC/xcp_d/pull/985
* Fix HCP ingression for non-rest tasks by @tsalo in https://github.com/PennLINC/xcp_d/pull/986
* Allow executive summary registration figure to really be empty by @tsalo in https://github.com/PennLINC/xcp_d/pull/987
* Allow motion filtering when motion-based censoring is disabled by @tsalo in https://github.com/PennLINC/xcp_d/pull/994
* Fix node labels and communities in Gordon atlas by @tsalo in https://github.com/PennLINC/xcp_d/pull/996
* Fix coverage CIFTIs by @tsalo in https://github.com/PennLINC/xcp_d/pull/998
* Remove unused native-space support by @tsalo in https://github.com/PennLINC/xcp_d/pull/1003

### Other Changes
* Update documentation for 0.5.0 release by @tsalo in https://github.com/PennLINC/xcp_d/pull/937
* Try to reduce memory requirements of functional connectivity nodes by @tsalo in https://github.com/PennLINC/xcp_d/pull/982
* Try to reduce memory requirements of denoising node by @tsalo in https://github.com/PennLINC/xcp_d/pull/989

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.5.0...0.6.0


## 0.5.1

The 0.5.1 fixes some bugs for the XCP-D manuscript.

### 🛠 Breaking Changes
* Update the QC metrics by @tsalo in https://github.com/PennLINC/xcp_d/pull/958

### 🎉 Exciting New Features
* Split HCP tasks into task and run entities by @tsalo in https://github.com/PennLINC/xcp_d/pull/952
* Concatenate across directions as well as runs by @tsalo in https://github.com/PennLINC/xcp_d/pull/965

### 🐛 Bug Fixes
* Fix resting-state plots in executive summary by @tsalo in https://github.com/PennLINC/xcp_d/pull/941
* Load T1w-to-standard transform to same space as volumetric BOLD scan by @tsalo in https://github.com/PennLINC/xcp_d/pull/926
* Pin Nilearn version by @tsalo in https://github.com/PennLINC/xcp_d/pull/955
* Don't interpolate volumes at beginning/end of run by @tsalo in https://github.com/PennLINC/xcp_d/pull/950

### Other Changes
* Update documentation for 0.5.0 release by @tsalo in https://github.com/PennLINC/xcp_d/pull/937

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.5.0...0.5.1


## 0.5.0

The 0.5.0 release prepares for the XCP-D manuscript, so I plan to not introduce any backwards-incompatible changes between this release and 1.0.0 (the official paper release).

### 🛠 Breaking Changes

* Add CIFTI subcortical atlas to XCP-D by @tsalo in https://github.com/PennLINC/xcp_d/pull/864
* Change "ciftiSubcortical" atlas name to "HCP" by @tsalo in https://github.com/PennLINC/xcp_d/pull/932

### 🎉 Exciting New Features

* Write out top-level sidecar for PennLINC QC file by @tsalo in https://github.com/PennLINC/xcp_d/pull/894
* Parcellate ReHo, ALFF, and surface morphometric maps by @tsalo in https://github.com/PennLINC/xcp_d/pull/839
* Refactor dcan/hcp ingestion and add more surface files by @tsalo in https://github.com/PennLINC/xcp_d/pull/887
* Add "none" option for denoising by @tsalo in https://github.com/PennLINC/xcp_d/pull/879
* Add `--exact-time` parameter by @tsalo in https://github.com/PennLINC/xcp_d/pull/885
* Allow white matter surface suffix to be either "_smoothwm" or "_white" by @tsalo in https://github.com/PennLINC/xcp_d/pull/899
* Add `--fs-license-file` parameter to command line interface by @tsalo in https://github.com/PennLINC/xcp_d/pull/930
* Generate executive summary figures without `--dcan-qc` by @tsalo in https://github.com/PennLINC/xcp_d/pull/936

### 🐛 Bug Fixes

* Standardize executive summary carpet if params is "none" by @tsalo in https://github.com/PennLINC/xcp_d/pull/916
* Support CIFTI morphometry files and add PNC test data by @tsalo in https://github.com/PennLINC/xcp_d/pull/922
* Correct ABCD/HCP surface ingression by @tsalo in https://github.com/PennLINC/xcp_d/pull/927

### Other Changes

* Add contributing documentation by @tsalo in https://github.com/PennLINC/xcp_d/pull/893
* Update changelog and CFF file for 0.4.0 by @tsalo in https://github.com/PennLINC/xcp_d/pull/896
* Change packaging to use hatch-vcs and pyproject.toml by @tsalo in https://github.com/PennLINC/xcp_d/pull/897
* Fix description of signal denoising method in docs by @tsalo in https://github.com/PennLINC/xcp_d/pull/898
* Improve QC sidecar contents by @tsalo in https://github.com/PennLINC/xcp_d/pull/900
* Rename the parcellated ALFF/ReHo outputs by @tsalo in https://github.com/PennLINC/xcp_d/pull/902
* Remove workaround for nonbinary Nibabies brain masks by @tsalo in https://github.com/PennLINC/xcp_d/pull/905
* Update Landing Page Figure by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/908
* Remove unused functions in filemanip module by @tsalo in https://github.com/PennLINC/xcp_d/pull/911
* Add tests for xcp_d.utils.utils module by @tsalo in https://github.com/PennLINC/xcp_d/pull/910
* Add tests for xcp_d.utils.execsummary module by @tsalo in https://github.com/PennLINC/xcp_d/pull/912
* Expand tests by @tsalo in https://github.com/PennLINC/xcp_d/pull/913
* Test CLI parameter validation by @tsalo in https://github.com/PennLINC/xcp_d/pull/918
* Refactor collection functions by @tsalo in https://github.com/PennLINC/xcp_d/pull/917

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/04.0...0.5.0


## 0.4.0

### 🛠 Breaking Changes

* Change default highpass filter cutoff from 0.009 to 0.01 by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/755
* Refactor anatomical workflow by @tsalo in https://github.com/PennLINC/xcp_d/pull/706
* Do not require `--combineruns` to generate DCAN QC files by @tsalo in https://github.com/PennLINC/xcp_d/pull/753
* Change QC filenames and fix `xcp_d-combineqc` command-line interface by @tsalo in https://github.com/PennLINC/xcp_d/pull/762
* Retain cohort entity in derivative filenames by @tsalo in https://github.com/PennLINC/xcp_d/pull/769
* Require the `--cifti` flag in order to use `--warp-surfaces-native2std` by @tsalo in https://github.com/PennLINC/xcp_d/pull/770
* Zero out parcels with <50% coverage by @tsalo in https://github.com/PennLINC/xcp_d/pull/757
* Use constant padding and maximum padlen for temporal filtering by @tsalo in https://github.com/PennLINC/xcp_d/pull/779
* Replace non-aggressive denoising with orthogonalization and streamline denoising by @tsalo in https://github.com/PennLINC/xcp_d/pull/808
* Add design matrix to report by @tsalo in https://github.com/PennLINC/xcp_d/pull/824
* Write out censored results by @tsalo in https://github.com/PennLINC/xcp_d/pull/820
* Allow users to disable censoring, and only generate ALFF if censoring is disabled by @tsalo in https://github.com/PennLINC/xcp_d/pull/828
* Implement `--min-time` parameter by @tsalo in https://github.com/PennLINC/xcp_d/pull/836
* Remove deprecated `--dummytime` parameter by @tsalo in https://github.com/PennLINC/xcp_d/pull/837
* Change FD Threshold from 0.2 to 0.3 by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/854
* Use filtered motion in nuisance regression by @tsalo in https://github.com/PennLINC/xcp_d/pull/871

### 🎉 Exciting New Features

* Add option to estimate brain radius from brain mask by @tsalo in https://github.com/PennLINC/xcp_d/pull/754
* Output warped atlases to derivatives by @tsalo in https://github.com/PennLINC/xcp_d/pull/647
* Add `min_coverage` parameter to threshold parcel coverage by @tsalo in https://github.com/PennLINC/xcp_d/pull/782
* Improve executive summary carpet plots by @tsalo in https://github.com/PennLINC/xcp_d/pull/747
* Output T2w images if available by @tsalo in https://github.com/PennLINC/xcp_d/pull/648
* Generate CIFTI and TSV versions of coverage, timeseries, and correlation files by @tsalo in https://github.com/PennLINC/xcp_d/pull/785
* Add colorbar to executive summary carpet plots by @tsalo in https://github.com/PennLINC/xcp_d/pull/829
* Support fsLR-space shape files generated by preprocessing pipelines by @tsalo in https://github.com/PennLINC/xcp_d/pull/773
* Support preprocessing derivatives with T2w, but no T1w by @tsalo in https://github.com/PennLINC/xcp_d/pull/838
* Support high-pass or low-pass only filtering by @smeisler in https://github.com/PennLINC/xcp_d/pull/862

### 🐛 Bug Fixes

* Replace missing vertices' values with NaNs by @tsalo in https://github.com/PennLINC/xcp_d/pull/743
* Select MNI152NLin6Asym target space for T1w from CIFTI derivatives by @tsalo in https://github.com/PennLINC/xcp_d/pull/759
* Only generate brainsprite figures when the `--dcan_qc` flag is used by @tsalo in https://github.com/PennLINC/xcp_d/pull/766
* Check for existence of dataset_description.json in fmri_dir by @tsalo in https://github.com/PennLINC/xcp_d/pull/806
* Move dataset desc check to after conversion by @tsalo in https://github.com/PennLINC/xcp_d/pull/809
* Refactor HCP/DCAN ingression and fix converted filenames by @tsalo in https://github.com/PennLINC/xcp_d/pull/714
* Connect custom confounds to confound consolidation node by @tsalo in https://github.com/PennLINC/xcp_d/pull/835
* Work around load_confounds aCompCor bug by @tsalo in https://github.com/PennLINC/xcp_d/pull/851
* Allow for smoothing to be zero by @tsalo in https://github.com/PennLINC/xcp_d/pull/861
* Use appropriate T1w/T2w in brainsprite workflow by @tsalo in https://github.com/PennLINC/xcp_d/pull/860
* Test DCAN and HCP ingestion and fix related bugs by @tsalo in https://github.com/PennLINC/xcp_d/pull/848

### Other Changes

* Add tests for cifti smoothness by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/659
* Fix CIFTI downcasting test by @tsalo in https://github.com/PennLINC/xcp_d/pull/752
* Use pytest to test command-line interface by @tsalo in https://github.com/PennLINC/xcp_d/pull/740
* Upload coverage reports to CodeCov by @tsalo in https://github.com/PennLINC/xcp_d/pull/758
* Link to `xcp_d-examples` in documentation by @tsalo in https://github.com/PennLINC/xcp_d/pull/761
* Add information about preprocessing requirements by @tsalo in https://github.com/PennLINC/xcp_d/pull/772
* Replace MultiLabel interpolation with GenericLabel by @tsalo in https://github.com/PennLINC/xcp_d/pull/786
* Mention cosine regressors in aCompCor boilerplate by @tsalo in https://github.com/PennLINC/xcp_d/pull/788
* Update base Docker image by @tsalo in https://github.com/PennLINC/xcp_d/pull/799
* Document Singularity requirements by @tsalo in https://github.com/PennLINC/xcp_d/pull/805
* Use workflow nodes for workflow tests by @tsalo in https://github.com/PennLINC/xcp_d/pull/807
* Adjust code for niworkflows 1.7.3 by @tsalo in https://github.com/PennLINC/xcp_d/pull/810
* Use a workflow for concatenation by @tsalo in https://github.com/PennLINC/xcp_d/pull/821
* Create prepare_confounds and denoise_bold workflows by @tsalo in https://github.com/PennLINC/xcp_d/pull/827
* Simplify the anatomical workflow some more by @tsalo in https://github.com/PennLINC/xcp_d/pull/841
* Remove codecov dependency by @tsalo in https://github.com/PennLINC/xcp_d/pull/850
* Add boilerplate to the executive summary by @tsalo in https://github.com/PennLINC/xcp_d/pull/857
* Add additional argument aliases by @smeisler in https://github.com/PennLINC/xcp_d/pull/870
* Update docs to mention HCPYA version by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/869
* Improve boilerplate by @tsalo in https://github.com/PennLINC/xcp_d/pull/866
* Update gitignore by @tsalo in https://github.com/PennLINC/xcp_d/pull/874
* Fix issues with HCP Ingression by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/872
* Document how to process multiple, but not all, tasks by @tsalo in https://github.com/PennLINC/xcp_d/pull/876
* Describe scope of XCP-D by @tsalo in https://github.com/PennLINC/xcp_d/pull/878
* Update docs by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/882
* Always generate the DCAN executive summary by @tsalo in https://github.com/PennLINC/xcp_d/pull/888
* Clean up testing framework and documentation by @tsalo in https://github.com/PennLINC/xcp_d/pull/889

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.3.2...04.0


## 0.3.2

This release adopts a workbench show-scene-based brainsprite for the executive summary. It also removes the brainsprite figure from the nipreps report.

### 🎉 Exciting New Features

* Adopt executive summary's brainsprite using jinja templates by @tsalo in https://github.com/PennLINC/xcp_d/pull/702

### Other Changes

* Remove example data and fix workflow graphs by @tsalo in https://github.com/PennLINC/xcp_d/pull/738
* Replace CiftiDespike with connected nodes by @tsalo in https://github.com/PennLINC/xcp_d/pull/737

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.3.1...0.3.2


## 0.3.1

### 🛠 Breaking Changes

* Remove `--nuissance-regressors` and `--bandpass_filter` by @tsalo in https://github.com/PennLINC/xcp_d/pull/658

### 🎉 Exciting New Features

* Add `--dummy-scans` parameter and deprecate `--dummytime` by @tsalo in https://github.com/PennLINC/xcp_d/pull/616
* Add `--bids-filter-file` parameter by @tsalo in https://github.com/PennLINC/xcp_d/pull/686
* Enable non-aggressive denoising with signal regressors by @tsalo in https://github.com/PennLINC/xcp_d/pull/697
* Improve identification of FreeSurfer derivatives by @tsalo in https://github.com/PennLINC/xcp_d/pull/719
* Collect preprocessed surfaces in new function by @tsalo in https://github.com/PennLINC/xcp_d/pull/731

### 🐛 Bug Fixes

* Remove dummy volumes from custom confounds files by @tsalo in https://github.com/PennLINC/xcp_d/pull/660
* Remove dummy volumes from beginning of each run in concatenation workflow by @tsalo in https://github.com/PennLINC/xcp_d/pull/661
* Retain noise components instead of dropping them in load_aroma by @tsalo in https://github.com/PennLINC/xcp_d/pull/670
* Use input_type to determine order of preferred spaces by @tsalo in https://github.com/PennLINC/xcp_d/pull/688
* Infer volumetric space from transform in executive summary with cifti data by @tsalo in https://github.com/PennLINC/xcp_d/pull/689
* Change input type from HPC to HCP by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/691
* Replace failing non-aggressive AROMA denoising with working aggressive denoising by @tsalo in https://github.com/PennLINC/xcp_d/pull/693
* Select best volumetric space for transforms even when using `--cifti` flag by @tsalo in https://github.com/PennLINC/xcp_d/pull/695
* Downcast >32-bit files to 32-bit by @tsalo in https://github.com/PennLINC/xcp_d/pull/666
* Use appropriate intent codes for cifti outputs by @tsalo in https://github.com/PennLINC/xcp_d/pull/690
* Only concatenate processed runs by @tsalo in https://github.com/PennLINC/xcp_d/pull/713
* Do not use downcasted files as name sources by @tsalo in https://github.com/PennLINC/xcp_d/pull/712
* Warp segmentation file to appropriate space for carpet plots by @tsalo in https://github.com/PennLINC/xcp_d/pull/727
* Use brain mask in NIFTI connectivity workflow by @tsalo in https://github.com/PennLINC/xcp_d/pull/733

### Other Changes

* Consolidate confounds at beginning of denoising workflows by @tsalo in https://github.com/PennLINC/xcp_d/pull/664
* Remove unused outputnodes from nifti and cifti workflows by @tsalo in https://github.com/PennLINC/xcp_d/pull/667
* Move QC/censoring plots into new workflow by @tsalo in https://github.com/PennLINC/xcp_d/pull/668
* Testing affines don't change across XCP runs by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/649
* Load confounds via Nilearn functionality by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/675
* Update docs by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/682
* Add masks to package data by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/715
* Use ds001491 fMRIPrep derivatives for tests by @tsalo in https://github.com/PennLINC/xcp_d/pull/698
* Add tests for confound loading function by @tsalo in https://github.com/PennLINC/xcp_d/pull/730
* Refactor executive summary workflow by @tsalo in https://github.com/PennLINC/xcp_d/pull/721
* Track start of workflow with sentry by @tsalo in https://github.com/PennLINC/xcp_d/pull/732

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.3.0...0.3.1


## 0.3.0

We are happy to announce a new minor release, with several backwards-incompatible changes.

Two big breaking changes are (1) there is a new `--dcan-qc` flag that determines if the executive summary and DCAN-format QC files will be generated, and (2) custom confounds should now have headers, should be tab-delimited, and should have the same names as the fMRIPrep confounds, for easier indexing.

### 🛠 Breaking Changes

* Output ReHo as a CIFTI by @tsalo in https://github.com/PennLINC/xcp_d/pull/601
* Add `--dcan-qc` flag by @tsalo in https://github.com/PennLINC/xcp_d/pull/650
* Support custom confounds with headers by @tsalo in https://github.com/PennLINC/xcp_d/pull/642

### 🎉 Exciting New Features

* Use BIDSLayout in concatenation code by @tsalo in https://github.com/PennLINC/xcp_d/pull/600
* [ENH] Support Nibabies ingression into XCP by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/602
* Add static ALFF and ReHo plots to report by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/604
* Write confounds out to derivatives by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/634
* Add column names to confounds df by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/641

### 🐛 Bug Fixes

* Select BOLD files in a single space by @tsalo in https://github.com/PennLINC/xcp_d/pull/603
* Censor data in executive summary plots correctly by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/614
* Only generate ALFF derivatives if bandpass filtering is enabled by @tsalo in https://github.com/PennLINC/xcp_d/pull/628
* Do not merge in concatenated files when re-running concatenation workflow by @tsalo in https://github.com/PennLINC/xcp_d/pull/633
* Map abbreviated entities to full names for templateflow get call by @tsalo in https://github.com/PennLINC/xcp_d/pull/654
* Ensure `dummyscans` is an integer in PlotSVGData by @tsalo in https://github.com/PennLINC/xcp_d/pull/655

### Other Changes

* Use Nilearn for brainsprite generation by @tsalo in https://github.com/PennLINC/xcp_d/pull/607
* Cache the downloaded test data by @tsalo in https://github.com/PennLINC/xcp_d/pull/629
* Use BIDSLayout instead of globbing functions to collect necessary files by @tsalo in https://github.com/PennLINC/xcp_d/pull/621
* Start to standardize interface parameter calls by @tsalo in https://github.com/PennLINC/xcp_d/pull/638
* Simplify transform-getting functions by @tsalo in https://github.com/PennLINC/xcp_d/pull/623
* Lint with black without linting workflow connections by @tsalo in https://github.com/PennLINC/xcp_d/pull/640

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.2.0...0.3.0


## 0.2.2

This is a patch release for the 0.2 series. The main bug being fixed is that using `--dummytime` was causing crashes in the executive summary workflow.

### 🎉 Exciting New Features

* Write confounds out to derivatives by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/634
* Add column names to confounds df by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/641

### 🐛 Bug Fixes

* Do not merge in concatenated files when re-running concatenation workflow by @tsalo in https://github.com/PennLINC/xcp_d/pull/633
* Map abbreviated entities to full names for templateflow get call by @tsalo in https://github.com/PennLINC/xcp_d/pull/654
* Ensure `dummyscans` is an integer in PlotSVGData by @tsalo in https://github.com/PennLINC/xcp_d/pull/655

### Other Changes

* Cache the downloaded test data by @tsalo in https://github.com/PennLINC/xcp_d/pull/629
* Use BIDSLayout instead of globbing functions to collect necessary files by @tsalo in https://github.com/PennLINC/xcp_d/pull/621
* Start to standardize interface parameter calls by @tsalo in https://github.com/PennLINC/xcp_d/pull/638
* Simplify transform-getting functions by @tsalo in https://github.com/PennLINC/xcp_d/pull/623
* Lint with black without linting workflow connections by @tsalo in https://github.com/PennLINC/xcp_d/pull/640

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.2.1...0.2.2


## 0.2.1

This is a patch release for 0.2.0.

There is a known bug with the concatenation workflow, so we advise users not to use the `-m`/`--combineruns` option with this release.

### 🛠 Breaking Changes

* Output ReHo as a CIFTI by @tsalo in https://github.com/PennLINC/xcp_d/pull/601

### 🎉 Exciting New Features

* Use BIDSLayout in concatenation code by @tsalo in https://github.com/PennLINC/xcp_d/pull/600
* [ENH] Support Nibabies ingression into XCP by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/602
* Add static ALFF and ReHo plots to report by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/604

### 🐛 Bug Fixes

* Select BOLD files in a single space by @tsalo in https://github.com/PennLINC/xcp_d/pull/603
* Censor data in executive summary plots correctly by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/614
* Only generate ALFF derivatives if bandpass filtering is enabled by @tsalo in https://github.com/PennLINC/xcp_d/pull/628

### Other Changes

* Use Nilearn for brainsprite generation by @tsalo in https://github.com/PennLINC/xcp_d/pull/607

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.2.0...0.2.1


## 0.2.0

This is a big release. There are a lot of backwards-incompatible changes, as well as a number of bug-fixes and enhancements.

There is a full list of the changes made between 0.1.3 and 0.2.0 below. However, here are some highlights:

1. We have renamed and reorganized a number of the outputs created by `xcp_d` to better follow BIDS convention. There is a lot in `xcp_d` that falls outside the current BIDS specification, so we took inspiration from a number of BIDS Extension Proposals (BEPs) that folks have written over the years.
2. There is a new `--warp-surfaces-native2std` flag, which produces a number of subject-specific surfaces in fsLR space. This was previously run by default in version 0.1.3. The workflow that this flag triggers is also much improved, thanks to @madisoth.
3. We have fixed a major bug, in which the parameter set users selected was ignored, and "36P" was used no matter what.
4. The HTML reports have been improved. The interactive segmentation image from the executive summary has been added to the main report, along with the BOLD-T1w coregistration figure. We have also added a new plot to show the impact of filtering motion parameters on censoring.

### 🛠 Breaking Changes

* Ensure TSV files are tab-delimited by @tsalo in https://github.com/PennLINC/xcp_d/pull/541
* Rename derivatives to be more BIDS-compliant by @tsalo in https://github.com/PennLINC/xcp_d/pull/553
* Replace `--func-only` with `--warp-surfaces-native2std` by @tsalo in https://github.com/PennLINC/xcp_d/pull/562
* Add headers to motion.tsv and outliers.tsv files by @tsalo in https://github.com/PennLINC/xcp_d/pull/587

### 🎉 Exciting New Features

* Create dataset description file by @tsalo in https://github.com/PennLINC/xcp_d/pull/561
* Distinguish preprocessed dataset formats by @tsalo in https://github.com/PennLINC/xcp_d/pull/567
* Output temporal mask by @tsalo in https://github.com/PennLINC/xcp_d/pull/586
* Add censoring plot to summary reports by @tsalo in https://github.com/PennLINC/xcp_d/pull/579
* Add existing SVG figures and brainplot HTML figure to HTML report by @tsalo in https://github.com/PennLINC/xcp_d/pull/590
* Output all filtered motion parameters by @tsalo in https://github.com/PennLINC/xcp_d/pull/592
* Deprecate `--bandpass_filter` in favor of `--disable-bandpass-filter` by @tsalo in https://github.com/PennLINC/xcp_d/pull/588

### 👎 Deprecations

* Deprecate nuissance-regressors in favor of nuisance-regressors by @tsalo in https://github.com/PennLINC/xcp_d/pull/513

### 🐛 Bug Fixes

* Fix fsLR32k reg and ApplyWarpfield issues by @madisoth in https://github.com/PennLINC/xcp_d/pull/442
* Replace MIT license with BSD-3 by @tsalo in https://github.com/PennLINC/xcp_d/pull/457
* Update utils.py to fix MNILin6 imports by @smeisler in https://github.com/PennLINC/xcp_d/pull/458
* Potential fix for "no tasks found" problem by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/464
* Fix "manadatory" typo in interfaces by @tsalo in https://github.com/PennLINC/xcp_d/pull/540
* Change "hpc" in CLI to "hcp" by @tsalo in https://github.com/PennLINC/xcp_d/pull/560
* Pin pybids to version 0.15.1 by @madisoth in https://github.com/PennLINC/xcp_d/pull/537
* Fix resampling of fsnative structural surfaces to fsLR by @madisoth in https://github.com/PennLINC/xcp_d/pull/496
* Fix documentation and validation of motion filtering parameters by @tsalo in https://github.com/PennLINC/xcp_d/pull/575
* Show framewise displacement before filtering in QCPlot by @tsalo in https://github.com/PennLINC/xcp_d/pull/581
* Fix concatenation code w.r.t. recent output changes by @tsalo in https://github.com/PennLINC/xcp_d/pull/593
* Use user-requested parameters in regression by @tsalo in https://github.com/PennLINC/xcp_d/pull/596
* Update load_aroma so the correct file is being read in by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/599

### Other Changes

* Rf/smoothing by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/415
* Rf/misc. by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/421
* Rf/fconts by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/424
* [RF] Miscellaneous by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/439
* [RF] Reho Computation by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/437
* [DOC] Update docs by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/433
* [RF] ALFF Computation by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/435
* [RF] fixes by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/443
* [RF] Executive summary by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/445
* [RF] Compute qcplot.py by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/447
* [RF] Make TR tests shorter by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/462
* [DOC] Add badges to README by @tsalo in https://github.com/PennLINC/xcp_d/pull/466
* [DOC] Add issue templates and route to NeuroStars by @tsalo in https://github.com/PennLINC/xcp_d/pull/465
* [REF] Use nilearn for loading and masking nifti data by @tsalo in https://github.com/PennLINC/xcp_d/pull/459
* [TST] Add linting GH workflow by @tsalo in https://github.com/PennLINC/xcp_d/pull/467
* [DOC] Address formatting issues in documentation by @tsalo in https://github.com/PennLINC/xcp_d/pull/469
* [REF] Replace all relative imports with absolute ones by @tsalo in https://github.com/PennLINC/xcp_d/pull/472
* [FIX] Address failing RTD build by @tsalo in https://github.com/PennLINC/xcp_d/pull/477
* [REF] Use f-strings consistently throughout codebase by @tsalo in https://github.com/PennLINC/xcp_d/pull/473
* [DOC] Use welcome bot to comment on new contributors' PRs by @tsalo in https://github.com/PennLINC/xcp_d/pull/475
* [RF] Linting by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/479
* [DOC] Add or format docstrings throughout package by @tsalo in https://github.com/PennLINC/xcp_d/pull/474
* [REF] Lint interfaces by @tsalo in https://github.com/PennLINC/xcp_d/pull/480
* [REF] Lint workflows by @tsalo in https://github.com/PennLINC/xcp_d/pull/481
* [REF] Lint utils by @tsalo in https://github.com/PennLINC/xcp_d/pull/485
* [REF] Rename functions to snake_case by @tsalo in https://github.com/PennLINC/xcp_d/pull/487
* [REF] Rename classes to CamelCase by @tsalo in https://github.com/PennLINC/xcp_d/pull/486
* [RF] Get rid of unused variables by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/493
* [DOC] Add docstrings to CLI module by @tsalo in https://github.com/PennLINC/xcp_d/pull/490
* [REF] Address remaining linter warnings by @tsalo in https://github.com/PennLINC/xcp_d/pull/494
* [REF] Move functions and classes into appropriate modules by @tsalo in https://github.com/PennLINC/xcp_d/pull/489
* [DOC] Add API to documentation by @tsalo in https://github.com/PennLINC/xcp_d/pull/498
* [DOC] Clean up workflow docstrings by @tsalo in https://github.com/PennLINC/xcp_d/pull/503
* [REF] Remove duplicate functions by @tsalo in https://github.com/PennLINC/xcp_d/pull/502
* [REF] Empty __init__.py files by @tsalo in https://github.com/PennLINC/xcp_d/pull/499
* [REF] Centralize DerivativesDataSink definition by @tsalo in https://github.com/PennLINC/xcp_d/pull/510
* [MAINT] Add CITATION.cff file by @tsalo in https://github.com/PennLINC/xcp_d/pull/482
* [TST] Store CircleCI artifacts by @tsalo in https://github.com/PennLINC/xcp_d/pull/521
* [REF] Start replacing common parameters with fill_doc by @tsalo in https://github.com/PennLINC/xcp_d/pull/509
* [REF] Remove the notebooks module by @tsalo in https://github.com/PennLINC/xcp_d/pull/531
* [DOC] Add TemplateFlow citation to boilerplate by @tsalo in https://github.com/PennLINC/xcp_d/pull/520
* [DOC] Use sphinxcontrib-bibtex to embed BibTeX references in documentation by @tsalo in https://github.com/PennLINC/xcp_d/pull/516
* [TST] Make local tests stricter by @tsalo in https://github.com/PennLINC/xcp_d/pull/545
* [TST] Check for existence of outputs in tests by @tsalo in https://github.com/PennLINC/xcp_d/pull/542
* [REF] Remove FSL from dependencies by @tsalo in https://github.com/PennLINC/xcp_d/pull/528
* [TST] Restructure local patch handling in testing setup by @tsalo in https://github.com/PennLINC/xcp_d/pull/550
* [REF] Simplify BOLD/CIFTI post-processing workflow call by @tsalo in https://github.com/PennLINC/xcp_d/pull/534
* [REF] Remove unused init_post_process_wf by @tsalo in https://github.com/PennLINC/xcp_d/pull/535
* [REF] Use MapNodes to simplify functional connectivity workflows by @tsalo in https://github.com/PennLINC/xcp_d/pull/546
* Add release notes template by @tsalo in https://github.com/PennLINC/xcp_d/pull/552
* Add warning to documentation about M1 chip by @tsalo in https://github.com/PennLINC/xcp_d/pull/572
* Improve confound plot readability and fix moving registration plot by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/530
* Fix the local pytest script and documentation by @tsalo in https://github.com/PennLINC/xcp_d/pull/564
* [TEST] Add tests for outstanding modules by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/529
* [FIX] Fcon workflow tests are incompatible with changes from main by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/584

### New Contributors

* @tsalo made their first contribution in https://github.com/PennLINC/xcp_d/pull/457

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.1.3...0.2.0


## 0.1.3

* Add analysis-level option
*  Quick fix for bugs in bold and cifti workflow

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.1.2...0.1.3


## 0.1.2

* Fix malloc failure by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/395
* [RF] Filtering by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/390
* Rf/rename by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/398
* Rf/rename by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/400
* Rf/rename by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/401
* [RF] Smoothing by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/404

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.1.1...0.1.2


## 0.1.1

* Replace NiPype's filemanip.py by @madisoth in https://github.com/PennLINC/xcp_d/pull/383
* [RF] Fix typos by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/386
* ENH: Update filemanip.py "special_extensions" with the rest of the CIFTI-2 multi-part extensions by @madisoth in https://github.com/PennLINC/xcp_d/pull/388
* [RF] Interpolation by @kahinimehta in https://github.com/PennLINC/xcp_d/pull/387
* FIX: reho / write_gii by @madisoth in https://github.com/PennLINC/xcp_d/pull/385

**Full Changelog**: https://github.com/PennLINC/xcp_d/compare/0.1.0...0.1.1


## 0.1.0

Additional features for surface processing and expanded CI testing


## 0.0.9

1. Now supports Schaefer 100,200,300,400,500,600,700,800,900,1000
2. CompCor fix, where previously we were not including cosine
3. Removed Temporal CompCor; no one should use this anyways
4. Added aroma_gsr and compcor_gsr, which include global signal regression
5. Huge update of the documentation
6. This is the handoff update from @a3sha2 to @mb3152


## 0.0.8

afni-reho-despike correction

* Hcpdcan by @a3sha2 in https://github.com/PennLINC/xcp_abcd/pull/211
* Hcpdcan by @a3sha2 in https://github.com/PennLINC/xcp_abcd/pull/212

**Full Changelog**: https://github.com/PennLINC/xcp_abcd/compare/0.0.7...0.0.8


## 0.0.7

* Hcpdcan by @a3sha2 in https://github.com/PennLINC/xcp_abcd/pull/181
* Hcpdcan by @a3sha2 in https://github.com/PennLINC/xcp_abcd/pull/182
* Hcpdcan by @a3sha2 in https://github.com/PennLINC/xcp_abcd/pull/183
* thread/meg_gb included in plugin settings by @a3sha2 in https://github.com/PennLINC/xcp_abcd/pull/184
* nthread/ompthread by @a3sha2 in https://github.com/PennLINC/xcp_abcd/pull/185
* remove labelled by @a3sha2 in https://github.com/PennLINC/xcp_abcd/pull/186
* increased bold memgb size by @a3sha2 in https://github.com/PennLINC/xcp_abcd/pull/187
* Hcpdcan by @a3sha2 in https://github.com/PennLINC/xcp_abcd/pull/194

**Full Changelog**: https://github.com/PennLINC/xcp_abcd/compare/0.0.6...0.0.7


## 0.0.6

anatomical update
executive summary
dcan ingression


## 0.0.5

NIBABIES TEST
- [x] MNI152NLin6Asym template for nibabies #156
- [x] update executive summary and report #69 #115 #157  #151
- [x] anatomical workflow added including freesurfer/freesurfer*
- [x] outputs  for cifti and nifti


## 0.0.4

final release for hcp


## 0.0.3

hcp
Update cifti.py


## 0.0.2

Merge pull request #113 from PennLINC/template

Template


## 0.0.1

Merge pull request #64 from PennLINC/test_xcp

add version automatically
