# EEG-to-speech factorized decoding: literature package manifest

This manifest accompanies `eeg_speech_factorized_decoding_literature_review_20260726.md`.
It records the **20 references actually cited in that review**, the saved filename, and a
lawful public PDF source.  Where the published paper is paywalled, the package uses an
author-hosted, PMC, or preprint version and states that explicitly.  The package is a
reading archive, not a redistribution of subscription-only versions.

## Contents

| ID | Saved PDF | Public source / version archived |
|---|---|---|
| S01 | `S01_Baevski_2020_wav2vec2.pdf` | [NeurIPS version of record](https://proceedings.neurips.cc/paper_files/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf) |
| S02 | `S02_Qian_2019_AutoVC.pdf` | [ICML/PMLR version of record](https://proceedings.mlr.press/v97/qian19c/qian19c.pdf) |
| S03 | `S03_Ao_2022_SpeechT5.pdf` | [ACL Anthology version of record](https://aclanthology.org/2022.acl-long.393.pdf) |
| S04 | `S04_Li_2023_StyleTTS2.pdf` | [NeurIPS version of record](https://proceedings.neurips.cc/paper_files/paper/2023/file/3eaad2a0b62b5ed7a2e66c2188bb1449-Paper-Conference.pdf) |
| S05 | `S05_Hsu_2021_HuBERT.pdf` | [arXiv author preprint](https://arxiv.org/pdf/2106.07447) for the TASLP article |
| S06 | `S06_Zeghidour_2022_SoundStream.pdf` | [arXiv author preprint](https://arxiv.org/pdf/2107.03312) for the TASLP article |
| S07 | `S07_Krishna_2020_EEG_speech_synthesis.pdf` | [arXiv author preprint](https://arxiv.org/pdf/2002.12756) for the ICASSP paper |
| S08 | `S08_Yin_2025_cross_subject_split.pdf` | [ACL Anthology version of record](https://aclanthology.org/2025.emnlp-main.289.pdf) |
| S09 | `S09_Khanday_2025_NeuroIncept.pdf` | [arXiv author preprint](https://arxiv.org/pdf/2501.03757) for the ICASSP paper |
| S10 | `S10_Defossez_2023_noninvasive_speech.pdf` | [arXiv author preprint](https://arxiv.org/pdf/2208.12266) for the Nature Machine Intelligence article |
| S11 | `S11_Anumanchipalli_2019_neural_speech_synthesis.pdf` | [eScholarship author manuscript](https://escholarship.org/content/qt1rz5r354/qt1rz5r354.pdf) for the Nature article |
| S12 | `S12_Chen_2024_neural_speech_framework.pdf` | [Nature Machine Intelligence open-access version of record](https://www.nature.com/articles/s42256-024-00824-8.pdf) |
| S13 | `S13_Pasley_2012_auditory_cortex.pdf` | [eScholarship public copy](https://escholarship.org/content/qt7h63h0kv/qt7h63h0kv.pdf) of the PLOS Biology article |
| S14 | `S14_Tang_2023_semantic_reconstruction_ACCESS.md` | [PMC public full text (author manuscript)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304553/) — the PMC OA API does not licence a redistributable PDF package, so the folder provides an access note rather than a copied PDF |
| S15 | `S15_Moses_2021_anarthria_neuroprosthesis.pdf` | [eScholarship author manuscript](https://escholarship.org/content/qt3pc4j4pn/qt3pc4j4pn.pdf) for the NEJM article |
| S16 | `S16_Willett_2023_high_performance_neuroprosthesis.pdf` | [public author-hosted PDF](https://sbgg.org.br/wp-content/uploads/2023/09/6-A-high-performance-speech-neuroprosthesis.pdf) of the Nature article |
| S17 | `S17_Metzger_2023_avatar_neuroprosthesis.pdf` | [eScholarship author manuscript](https://escholarship.org/content/qt5829g475/qt5829g475_noSplash_25cec01720c32c1ba4a4e783e588cdaf.pdf) for the Nature article |
| S18 | `S18_Wu_2021_EEG_spoken_speech_GPR.pdf` | [EUSIPCO proceedings version](https://eurasip.org/Proceedings/Eusipco/Eusipco2021/pdfs/0001140.pdf) |
| S19 | `S19_dAscoli_2025_individual_words.pdf` | [Nature Communications open-access version](https://www.nature.com/articles/s41467-025-65499-0.pdf) |
| S20 | `S20_Gao_2026_NeuroSonic.pdf` | [arXiv preprint](https://arxiv.org/pdf/2606.24087) |

## Version and integrity notes

- S05, S06, S07, S09, S10 and S20 are preprints; cite the peer-reviewed version where one is listed in the review.
- S11, S15 and S17 are author manuscripts; S16 is a public author-hosted copy and may not use publisher layout.
- S14 is available to read as a PMC author manuscript. Its PMCID is not in the PMC open-access package set, so a PDF was deliberately not copied; `S14_Tang_2023_semantic_reconstruction_ACCESS.md` preserves the legitimate full-text and DOI links.
- S12 and S19 are open-access articles. S01–S04, S07–S09, S13 and S18 are public proceedings/journal PDFs.
- After download, `validate_eeg_speech_factorized_literature_20260726.sh` checks whether each saved file identifies as a PDF. Any source that returns an access or bot-check HTML page is left out and called out in its terminal report; its citation and source URL remain in this manifest.
- The review itself is the authoritative record of how every paper was used. Do not infer comparable performance across EEG, MEG, ECoG, sEEG and intracortical recording modalities.
