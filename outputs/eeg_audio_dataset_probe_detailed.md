# EEG-audio dataset probe detailed report

This report is generated from real remote metadata probes. Full EEG archives are not downloaded by default; partial byte-range artifacts are explicitly marked.

## Dataset Summary

| Dataset | Source | Priority | Fit | URL |
| --- | --- | --- | --- | --- |
| ds004408 | OpenNeuro | core | Natural speech pretraining; word/phoneme TextGrid alignment | https://openneuro.org/datasets/ds004408 |
| ds005345 | OpenNeuro | core | Natural Mandarin speech; single/multi-talker semantic and acoustic alignment | https://openneuro.org/datasets/ds005345 |
| ds004718 | OpenNeuro | core | Best word/prosody alignment; older Cantonese natural speech | https://openneuro.org/datasets/ds004718 |
| ds006104 | OpenNeuro | controlled | Phoneme/articulation probe; not continuous natural speech | https://openneuro.org/datasets/ds006104 |
| ds006434 | OpenNeuro | alignment | High-rate timing/stimulus-alignment stress test | https://openneuro.org/datasets/ds006434 |
| ds003774 | OpenNeuro | music | Natural music self-supervised EEG tokenization; weak affect labels | https://openneuro.org/datasets/ds003774 |
| ds007591 | OpenNeuro | secondary | Speech production/covert speech sanity check; small subject count | https://openneuro.org/datasets/ds007591 |
| zenodo-4004271 | Zenodo | baseline | Classic envelope/AAD benchmark; use strict trial/story split | https://zenodo.org/records/4004271 |
| zenodo-1199011 | Zenodo | baseline | Reverberant competing speech; good robustness test | https://zenodo.org/records/1199011 |
| zenodo-4518754 | Zenodo | spatial | High-density spatial tokenizer and sensor-ablation benchmark | https://zenodo.org/records/4518754 |
| zenodo-7078451 | Zenodo | secondary | Mandarin tonal-language AAD; useful after core OpenNeuro sets | https://zenodo.org/records/7078451 |
| openmiir | GitHub/Figshare | music | Beat/tempo/meter alignment probe before larger natural music sets | https://github.com/sstober/openmiir |
| zenodo-4537751 | Zenodo | music-secondary | Small music AAD set for target-instrument attention | https://zenodo.org/records/4537751 |

## Target-Level Evidence

### ds004408 - EEG responses to continuous naturalistic speech

OpenNeuro S3 first page: `1000` keys; truncated=`True`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| dataset_description | json | 338 | False | outputs/probe_artifacts/ds004408/01_dataset_description.json | Name=EEG responses to continuous naturalistic speech; DatasetDOI=doi:10.18112/openneuro.ds004408.v1.0.8; License=CC0 |
| participants | tsv | 577 | False | outputs/probe_artifacts/ds004408/02_participants.tsv | columns=['\ufeffparticipant_id', 'age', 'sex', 'hand', 'weight', 'height']; preview_rows=[['sub-001', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a'], ['sub-002', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a']] |
| audio01 TextGrid | textgrid | 180689 | False | outputs/probe_artifacts/ds004408/03_audio01_TextGrid.TextGrid | nonempty_labels=2465; duration_sec_hint=177.54; first_nonempty=['sil', 'HH', 'IY1', 'W', 'AH0', 'Z', 'AH0', 'N', 'OW1', 'L'] |
| audio01 wav header | wav | 4096 | True | outputs/probe_artifacts/ds004408/04_audio01_wav_header.wav.header.bin | sample_rate=44100; channels=2; bits=16; duration_sec_est=177.563 |
| run01 EEG sidecar | json | 487 | False | outputs/probe_artifacts/ds004408/05_run01_EEG_sidecar.json | SamplingFrequency=512.0; EEGReference=n/a; Manufacturer=Brain Products |
| run01 channels | tsv | 5920 | False | outputs/probe_artifacts/ds004408/06_run01_channels.tsv | columns=['name', 'type', 'units', 'description', 'sampling_frequency', 'status', 'status_description']; preview_rows=[['A1', 'EEG', 'V', 'ElectroEncephaloGram', '512.0', 'good', 'n/a'], ['A2', 'EEG', 'V', 'ElectroEncephaloGram', '512.0', 'good', 'n/a']] |
| run01 BrainVision header | text | 2984 | False | outputs/probe_artifacts/ds004408/07_run01_BrainVision_header.txt | bytes=2984; magic=None; range=None |
| run01 raw EEG bytes | binary | 512 | True | outputs/probe_artifacts/ds004408/08_run01_raw_EEG_bytes.bin | bytes=512; magic=78870f5132ad1c51e66b6dd05bead0cf; range=bytes 0-511/51380224 |

### ds005345 - Le Petit Prince Multi-talker

OpenNeuro S3 first page: `704` keys; truncated=`False`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| dataset_description | json | 578 | False | outputs/probe_artifacts/ds005345/01_dataset_description.json | Name=Le Petit Prince (LPP) Multi-talker: Naturalistic 7T fMRI and EEG Dataset; DatasetDOI=doi:10.18112/openneuro.ds005345.v1.0.1; License=CC0 |
| female word info | csv | 64469 | False | outputs/probe_artifacts/ds005345/02_female_word_info.csv | columns=['word', 'onset', 'offset', 'duration', 'logfreq', 'pos', 'td', 'bu', 'lc']; preview_rows=[['当', '0', '19', '19', '8.529', 'ADP', '3', '1', '2'], ['我', '19', '34', '15', '10.534', 'PRON', '5', '2', '3']] |
| female acoustic sample | csv | 262144 | True | outputs/probe_artifacts/ds005345/03_female_acoustic_sample.csv | columns=['time', 'f0', 'intensity']; preview_rows=[['0.000', '0', '35.04929769'], ['0.001', '0', '37.00205663']] |
| single female wav header | wav | 4096 | True | outputs/probe_artifacts/ds005345/04_single_female_wav_header.wav.header.bin | sample_rate=44100; channels=1; bits=16; duration_sec_est=603.0 |
| raw EEG sidecar | json | 1009 | False | outputs/probe_artifacts/ds005345/05_raw_EEG_sidecar.json | SamplingFrequency=500.0; EEGReference=Cz; Manufacturer=Brain Products |
| raw EEG BrainVision header | text | 11426 | False | outputs/probe_artifacts/ds005345/06_raw_EEG_BrainVision_header.txt | bytes=11426; magic=None; range=None |
| raw EEG bytes | binary | 512 | True | outputs/probe_artifacts/ds005345/07_raw_EEG_bytes.bin | bytes=512; magic=33f930010ffb9b0434f47a028c029afe; range=bytes 0-511/182722560 |
| preprocessed fif bytes | binary | 512 | True | outputs/probe_artifacts/ds005345/08_preprocessed_fif_bytes.bin | bytes=512; magic=000000640000001f0000001400000000; range=bytes 0-511/76938881 |

### ds004718 - LPPHK older Cantonese speakers

OpenNeuro S3 first page: `1000` keys; truncated=`True`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| dataset_description | json | 361 | False | outputs/probe_artifacts/ds004718/01_dataset_description.json | Name=Le Petit Prince Hong Kong: Naturalistic fMRI and EEG dataset from older Cantonese speakers; DatasetDOI=doi:10.18112/openneuro.ds004718.v1.1.2; License=CC0 |
| sentences | text | 21228 | False | outputs/probe_artifacts/ds004718/02_sentences.txt | bytes=21228; magic=None; range=None |
| word timing xlsx | xlsx | 346830 | False | outputs/probe_artifacts/ds004718/03_word_timing_xlsx.xlsx | sheets=['lppHK_word_information']; shape=(4474, 12); preview_rows=[['word', 'word_id', 'sentence_id', 'onset_in_sentence', 'offset_in_sentence', 'onset_all', 'offset_all', 'POS'], ['六歲', '1', '1.003', '0.049196', '0.39119600000000004', '0.049', '0.391', 'ADV']] |
| trigger sentence xlsx | xlsx | 17004 | False | outputs/probe_artifacts/ds004718/04_trigger_sentence_xlsx.xlsx | sheets=['Sheet1']; shape=(556, 2); preview_rows=[['ID of the sentence or question', 'Trigger number'], ['1.003.wav', '3']] |
| acoustic csv sample | csv | 262144 | True | outputs/probe_artifacts/ds004718/05_acoustic_csv_sample.csv | columns=['time', 'f0', 'intensity']; preview_rows=[['1', '105.3891421', '0'], ['1.01', '185.7175368', '0.0035825']] |
| sentence wav header | wav | 4096 | True | outputs/probe_artifacts/ds004718/06_sentence_wav_header.wav.header.bin | sample_rate=44100; channels=2; bits=16; duration_sec_est=3.92 |
| raw EEG sidecar | json | 449 | False | outputs/probe_artifacts/ds004718/07_raw_EEG_sidecar.json | SamplingFrequency=1000; EEGReference=Average |
| raw EEG EEGLAB set bytes | binary | 512 | True | outputs/probe_artifacts/ds004718/08_raw_EEG_EEGLAB_set_bytes.bin | bytes=512; magic=4d41544c414220352e30204d41542d66; range=bytes 0-511/11911048 |

### ds006104 - EEG dataset for speech decoding

OpenNeuro S3 first page: `361` keys; truncated=`False`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| dataset_description | json | 2584 | False | outputs/probe_artifacts/ds006104/01_dataset_description.json | Name=EEG dataset for speech decoding; DatasetDOI=doi:10.18112/openneuro.ds006104.v1.0.1; License=CC0 |
| channels | tsv | 719 | False | outputs/probe_artifacts/ds006104/02_channels.tsv | columns=['name', 'type', 'units']; preview_rows=[['Fp1', 'EEG', 'uV'], ['Fpz', 'EEG', 'uV']] |
| events | tsv | 58838 | False | outputs/probe_artifacts/ds006104/03_events.tsv | columns=['onset', 'duration', 'trial_type', 'category', 'manner', 'phoneme1', 'phoneme2', 'place', 'tms_intensity', 'tms_target', 'trial', 'voicing']; preview_rows=[['7.154', '0', 'TMS', 'alveolar', 'stop', 'n/a', 'n/a', 'alveolar', '110', 'control_lip', '1', 'no'], ['7.204', '0', 'stimulus', 'n/a', 'n/a', 'i', 't', 'n/a', 'n/a', 'control_lip', 'n/a', 'n/a']] |
| EEG sidecar | json | 783 | False | outputs/probe_artifacts/ds006104/04_EEG_sidecar.json | SamplingFrequency=2000; EEGReference=CPz |
| raw EDF bytes | binary | 512 | True | outputs/probe_artifacts/ds006104/05_raw_EDF_bytes.bin | bytes=512; magic=30202020202020205820582031332d4d; range=bytes 0-511/815192128 |

### ds006434 - ABR to natural speech and selective attention

OpenNeuro S3 first page: `1000` keys; truncated=`True`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| dataset_description | json | 1319 | False | outputs/probe_artifacts/ds006434/01_dataset_description.json | Name=The auditory brainstem response to natural speech is not affected by selective attention; DatasetDOI=doi:10.18112/openneuro.ds006434.v1.2.0; License=CC0 |
| events | tsv | 18528 | False | outputs/probe_artifacts/ds006434/02_events.tsv | columns=['onset', 'duration', 'trial_type', 'value', 'sample', 'chapter_ind', 'att_story', 'att_side', 'EEG_trigger']; preview_rows=[['40.642', '64', 'Stimulus/S 4', '4', '20321', '0', '0', '0', '1'], ['40.672', '64', 'Stimulus/S 4', '4', '20336', '0', '0', '0', '1']] |
| channels | tsv | 1875 | False | outputs/probe_artifacts/ds006434/03_channels.tsv | columns=['\ufeffname', 'type', 'units', 'low_cutoff', 'high_cutoff', 'description', 'sampling_frequency', 'status', 'status_description']; preview_rows=[['Fp1', 'EEG', 'V', '0.1', '250.0', 'ElectroEncephaloGram', '500.0', 'good', 'n/a'], ['Fp2', 'EEG', 'V', '0.1', '250.0', 'ElectroEncephaloGram', '500.0', 'good', 'n/a']] |
| EEG sidecar | json | 945 | False | outputs/probe_artifacts/ds006434/04_EEG_sidecar.json | SamplingFrequency=500.0; EEGReference=TP9, TP10; Manufacturer=Brain Products |
| stim wav header | wav | 4096 | True | outputs/probe_artifacts/ds006434/05_stim_wav_header.wav.header.bin | sample_rate=24414; channels=2; bits=16; duration_sec_est=64.0 |
| raw EEG bytes | binary | 512 | True | outputs/probe_artifacts/ds006434/06_raw_EEG_bytes.bin | bytes=512; magic=00000000000000000000000000000000; range=bytes 0-511/296412800 |

### ds003774 - MUSIN-G music listening EEG

OpenNeuro S3 first page: `1000` keys; truncated=`True`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| dataset_description | json | 461 | False | outputs/probe_artifacts/ds003774/01_dataset_description.json | Name=Music Listening- Genre EEG dataset (MUSIN-G); DatasetDOI=10.18112/openneuro.ds003774.v1.0.0; License=CC0 |
| events | tsv | 7680 | False | outputs/probe_artifacts/ds003774/02_events.tsv | columns=['onset', 'duration', 'sample', 'trial_type', 'response_time', 'stim_file', 'value']; preview_rows=[['0.2126755497', '1000.0000000000', '53.1689', 'n/a', 'n/a', 'n/a', 'SESS'], ['0.4346719316', '1000.0000000000', '108.668', 'n/a', 'n/a', 'n/a', 'CELL']] |
| channels | tsv | 1972 | False | outputs/probe_artifacts/ds003774/03_channels.tsv | columns=['name', 'type', 'units']; preview_rows=[['E1', 'EEG', 'microV'], ['E2', 'EEG', 'microV']] |
| EEG sidecar | json | 2100 | False | outputs/probe_artifacts/ds003774/04_EEG_sidecar.json | SamplingFrequency=250; EEGReference=Cz; Manufacturer=MagstimEGI |
| song wav header | wav | 4096 | True | outputs/probe_artifacts/ds003774/05_song_wav_header.wav.header.bin | sample_rate=8000; channels=2; bits=16; duration_sec_est=125.993 |
| raw EEG set bytes | binary | 512 | True | outputs/probe_artifacts/ds003774/06_raw_EEG_set_bytes.bin | bytes=512; magic=4d41544c414220352e30204d41542d66; range=bytes 0-511/261068280 |

### ds007591 - Delineating neural contributions to EEG-based speech decoding

OpenNeuro S3 first page: `126` keys; truncated=`False`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| dataset_description | json | 1122 | False | outputs/probe_artifacts/ds007591/01_dataset_description.json | Name=Delineating neural contributions to EEG-based speech decoding; DatasetDOI=doi:10.18112/openneuro.ds007591.v1.0.1; License=CC0 |
| participants | tsv | 65 | False | outputs/probe_artifacts/ds007591/02_participants.tsv | columns=['participant_id', 'age', 'sex']; preview_rows=[['sub-1', 'n/a', 'n/a'], ['sub-2', 'n/a', 'n/a']] |
| events | tsv | 5456 | False | outputs/probe_artifacts/ds007591/03_events.tsv | columns=['onset', 'duration', 'trial_type', 'value', 'session_type', 'task_condition']; preview_rows=[['58.96484375', '6.25', 'yellow', '4', 'calibration', 'minimally overt'], ['68.859375', '6.25', 'green', '0', 'calibration', 'minimally overt']] |
| channels | tsv | 3135 | False | outputs/probe_artifacts/ds007591/04_channels.tsv | columns=['name', 'type', 'units', 'sampling_frequency', 'status']; preview_rows=[['EEG001', 'EEG', 'V', '256', 'good'], ['EEG002', 'EEG', 'V', '256', 'good']] |
| EEG sidecar | json | 823 | False | outputs/probe_artifacts/ds007591/05_EEG_sidecar.json | SamplingFrequency=256; EEGReference=n/a (raw, pre-reference); Manufacturer=g.tec medical engineering GmbH |
| raw EDF bytes | binary | 512 | True | outputs/probe_artifacts/ds007591/06_raw_EDF_bytes.bin | bytes=512; magic=30202020202020205820582058205820; range=bytes 0-511/82497276 |

### zenodo-4004271 - Auditory Attention Detection Dataset KULeuven

DOI: `10.5281/zenodo.4004271`
Zenodo file count: `19`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| README | text | 12732 | False | outputs/probe_artifacts/zenodo-4004271/01_README.txt | bytes=12732; magic=None; range=None |
| preprocess script | text | 7759 | False | outputs/probe_artifacts/zenodo-4004271/02_preprocess_script.txt | bytes=7759; magic=None; range=None |

### zenodo-1199011 - EEG and audio dataset for auditory attention decoding

DOI: `10.5281/zenodo.1199011`
Zenodo file count: `4`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| preproc script | text | 6252 | False | outputs/probe_artifacts/zenodo-1199011/01_preproc_script.txt | bytes=6252; magic=None; range=None |

### zenodo-4518754 - Ultra high-density 255-channel EEG-AAD dataset

DOI: `10.5281/zenodo.4518754`
Zenodo file count: `33`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| misc zip | zip | 2030221 | False | outputs/probe_artifacts/zenodo-4518754/01_misc_zip.zip | entries=3; first_entries=[{'filename': 'misc/', 'size': 0}, {'filename': 'misc/channel-layout.jpg', 'size': 2051348}, {'filename': 'misc/eeg255ch_locs.csv', 'size': 3943}] |
| scripts zip | zip | 3086 | False | outputs/probe_artifacts/zenodo-4518754/02_scripts_zip.zip | entries=3; first_entries=[{'filename': 'scripts/', 'size': 0}, {'filename': 'scripts/loadCurryData.m', 'size': 5469}, {'filename': 'scripts/sample_script.m', 'size': 1033}] |

### zenodo-7078451 - ESAA: an EEG-Speech auditory attention detection database

DOI: `10.5281/zenodo.7078451`
Zenodo file count: `22`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| readme | text | 974 | False | outputs/probe_artifacts/zenodo-7078451/01_readme.txt | bytes=974; magic=None; range=None |
| preprocess zip | zip | 17877 | False | outputs/probe_artifacts/zenodo-7078451/02_preprocess_zip.zip | entries=6; first_entries=[{'filename': 'preprocess/readme.txt', 'size': 111}, {'filename': 'preprocess/db/database.py', 'size': 1231}, {'filename': 'preprocess/db/SCUT.py', 'size': 2020}, {'filename': 'preprocess/preproc/__init__.py', 'size': 284}, {'filename': 'preprocess/preproc/preprocess.py', 'size': 7612}] |
| baseline zip | zip | 24337 | False | outputs/probe_artifacts/zenodo-7078451/03_baseline_zip.zip | entries=9; first_entries=[{'filename': 'cnn_database/cnn.py', 'size': 4017}, {'filename': 'cnn_database/db/database.py', 'size': 1231}, {'filename': 'cnn_database/db/SCUT.py', 'size': 2020}, {'filename': 'cnn_database/device/gpu.py', 'size': 269}, {'filename': 'cnn_database/eutils/kflod.py', 'size': 1743}] |

### openmiir - OpenMIIR music perception and imagination


| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| stimuli metadata | xlsx | 43929 | False | outputs/probe_artifacts/openmiir/01_stimuli_metadata.xlsx | sheets=['Sheet1']; shape=(19, 27); preview_rows=[['id', 'song', 'audio file', 'length of song+cue (sec)', 'length of cue (sec)', 'length of song (sec)', 'length of cue only (sec)', 'cue bpm'], ['1', 'Chim Chim Cheree (lyrics)', 'S01_Chim Chim Cheree_lyrics.wav', '14.9481', '1.632', '13.3161', '1.8913', '212']] |
| electrode info | xlsx | 46917 | False | outputs/probe_artifacts/openmiir/02_electrode_info.xlsx | sheets=['Sheet1']; shape=(65, 7); preview_rows=[['Channel', 'Electrode', 'θ (Inclination)', 'φ (Azimuth)', 'x = -r sinθ sinφ', 'y = r sinθ cosφ', 'z = r cosθ'], ['A1', 'Fp1', '-92', '-72', '-95.04771583621135', '-30.882874957133396', '-3.4899496702500956']] |
| beat annotations | text | 672 | False | outputs/probe_artifacts/openmiir/03_beat_annotations.txt | bytes=672; magic=None; range=None |

### zenodo-4537751 - MAD-EEG: an EEG dataset for decoding auditory attention to a target instrument in polyphonic music

DOI: `10.5281/zenodo.4537751`
Zenodo file count: `8`

| Target | Kind | Bytes | Partial | Artifact | Parsed evidence |
| --- | --- | ---: | --- | --- | --- |
| behavioral data | xlsx | 13693 | False | outputs/probe_artifacts/zenodo-4537751/01_behavioral_data.xlsx | sheets=['Sheet1']; shape=(9, 39); preview_rows=[['date', 'gender', 'writing hand', 'age', 'Nas_Occ_distance', 'studies', 'job', 'refXXXX'], ['07/21/2016 17:11:52:556', 'Male', 'Right', '23', '1', 'Master', 'other', '1']] |
| raw yaml | text | 65536 | True | outputs/probe_artifacts/zenodo-4537751/02_raw_yaml.txt | bytes=65536; magic=None; range=None |
| sequences yaml | text | 65536 | True | outputs/probe_artifacts/zenodo-4537751/03_sequences_yaml.txt | bytes=65536; magic=None; range=None |
