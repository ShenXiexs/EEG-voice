# Internal DS006104 audio

Place the user-supplied 2021 trial-level original WAV files here. Keep a CSV
mapping EEG trial identity to audio path, for example:

```text
subject,session,task,run,trial_index,audio_path
S01,02,singlephoneme,01,0001,audio/S01_singlephoneme_0001.wav
```

The WAV files and mapping data are ignored by Git because they may contain
restricted data; this README remains trackable.
