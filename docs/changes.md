# What's New

## 0.6.0

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
<!-- Release notes generated using configuration in .github/release.yml at main -->

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
