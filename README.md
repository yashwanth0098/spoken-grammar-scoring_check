# Grammar Scoring Engine for Spoken English

Predicts a human grammar score (0–5) for 45–60 second recordings of people speaking English. Each recording is transcribed with Whisper, turned into text, fluency and audio features, and scored by a stacked regression model.

All the work is in one notebook, [`grammar_scoring_eda.ipynb`](grammar_scoring_eda.ipynb). Every section begins by explaining why it exists and what each code cell does, and ends with a "What this shows" summary.

## Result

Scores come from 5-fold cross-validation repeated 3 times, on the 732 training clips that have a real score (see [Validation](#validation) below for why the other 37 are left out).

| Model | RMSE | Pearson r |
|---|---|---|
| Always predict the mean | 1.01 | – |
| Whisper word confidence only | 0.83 | 0.58 |
| **Final model (stack of four)** | **0.63** | **0.78** |
| Final model, 45-second clips only | 0.68 | 0.66 |
| Final model, re-weighted to the test mix of clip lengths | 0.66 | – |

The final model's error is 38 % lower than always guessing the mean. **Expect about 0.66 on the test set**, not 0.63, because test is mostly 45-second clips and the model is weaker on those.

Predictions for the 216 test clips are in [`submission.csv`](submission.csv) (`filename,label`, same row order as `test.csv`).

## How it works

```
 raw .wav ──► clean audio ──► Whisper transcript ──► features ──► 4 models ──► stack ──► submission.csv
 (985)        trim edges,      medium.en, a time      confidence,   ridge,       ridge
              normalise to     stamp on every word    fluency,      boosted      on top
              -23 LUFS                                text, audio,  trees,
                                                      TF-IDF,       TF-IDF,
                                                      embedding     embedding
```

The notebook has three parts.

| Part | Sections | What it does |
|---|---|---|
| 1. Explore | 1–10 | Checks file integrity, labels, audio format, signal quality, pauses, the score-0 clips, and how far test differs from train |
| 2. Audio to text | 11–14 | Cleans the audio, transcribes every clip with Whisper, and checks the transcripts |
| 3. Model | 15–19 | Builds features, sets up validation, compares seven models, trains the final one and writes the submission |

### Features (section 15)

| Group | Why it is there |
|---|---|
| Whisper word confidence | Strongest single signal: correlation +0.62 with the score, and not explained by recording quality |
| Fluency from word times | Words per minute (+0.32) and the share of words lasting over 1 s, which marks hesitations |
| Text statistics | Vocabulary variety, word length, punctuation |
| Audio measures | Weak on their own (best +0.22); kept as controls for recording conditions |
| Clip type | 45-second clips score about half a point lower than 60-second clips |
| TF-IDF | Word and word-pair n-grams, plus character 2–5-grams |
| Text embedding | 384-dimensional `BAAI/bge-small-en-v1.5` embedding of each transcript |

Left out on purpose: filler counts (Whisper drops fillers), raw counts that grow with clip length, and the file number (test files are renumbered).

### Models (section 17)

Four base models (tabular ridge, tabular boosted trees, TF-IDF ridge, embedding ridge) are combined by a ridge stacker trained on out-of-fold predictions. The stack gives the four models similar weights (0.23 to 0.37), so each one adds information the others lack. The full comparison is saved in `working/features/cv_results.csv`.

## Findings from the data

These shaped the pipeline. The full tables with evidence are in sections 10, 14 and 19 of the notebook.

1. **The 37 clips labelled 0 are not "worst grammar".** They are one batch of recordings covered by loud constant noise, all in their own filename block (`audio_5037`–`audio_5073`), and found only in train. A two-threshold audio rule matches exactly those 37 and no test clip. Left in, they make any validation score look far better than it is: a "model" that only spots them reaches r = 0.60 without knowing anything about grammar.
2. **Test is mostly 45-second clips (65 %); train is mostly 60-second clips (73 %).** The two lengths are scored differently (mean 3.06 vs 3.58), so the expected test mean is lower (about 3.25). The final test predictions average 3.23.
3. **Audio statistics alone say little about grammar.** The best acoustic feature reaches a correlation of +0.22. The transcript has to carry the task.
4. **Whisper's confidence in its own words is the strongest signal**, but it mixes pronunciation, fluency and grammar. The model predicts the human score well; it does not detect grammar errors as such.
5. **Fillers and pauses cannot be read from the transcripts.** Whisper drops almost all "um"/"uh" (75 in 97,000 words), and gaps between word times mark punctuation, not hesitation.
6. **`sample_submission.csv` does not match `test.csv`** (204 vs 216 rows, 25 shared filenames). The submission is built from `test.csv`.
7. **Filenames repeat across train and test** (212 shared names, different audio). Every table is keyed on (`split`, `filename`).

## Validation

| Risk | How it is handled |
|---|---|
| The 37 noise-covered clips inflate any score | Train and validate only on the 732 clips with score > 0. A guard outputs 0 for any test clip that matches the noise signature (none does) |
| Test has a different mix of clip lengths | Folds are stratified by score and clip type. Error is reported for 45-second clips alone and re-weighted to the test mix |
| One split of 732 clips is noisy | 5-fold CV repeated 3 times |
| Leakage from held-out clips | Scaling, TF-IDF vocabulary and model weights are fitted inside the training folds only |

Predictions are clipped to the 0–5 scale.

## Project layout

```
├── grammar_scoring_eda.ipynb            # the whole project: exploration, transcription, model, submission
├── submission.csv                       # final predictions for the 216 test clips
├── grammar_scoring_preprocessing.ipynb  # earlier Kaggle draft of the preprocessing, superseded
├── Working_file.ipynb                   # earlier scratch notebook, superseded
└── shl-hiring-assessment-2026/          # NOT IN REPO: create it locally (see Setup)
    ├── Dataset_Final/                   # the dataset: train.csv, test.csv, train/*.wav (769), test/*.wav (216)
    └── working/                         # generated by the notebook: cleaned audio, transcripts, features, figures
```

The dataset folder is excluded by `.gitignore`: it is several GB, and the recordings are not ours to redistribute. Every chart and table it would hold is also shown in the notebook's saved outputs.

## Setup

Tested on Windows 11 with Python 3.13.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere

pip install numpy pandas scipy scikit-learn matplotlib joblib \
            librosa soundfile pyloudnorm \
            faster-whisper ctranslate2 \
            onnxruntime tokenizers huggingface-hub \
            ipykernel
python -m ipykernel install --user --name sshl --display-name "Python (SSHL .venv)"
```

Versions used for the submitted run: numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, librosa 1.0.0, soundfile 0.14.0, pyloudnorm 0.2.0, faster-whisper 1.2.1, ctranslate2 4.8.2, onnxruntime 1.30.0, tokenizers 0.23.2.

**GPU (optional, recommended for transcription).** With an NVIDIA GPU, also install the CUDA libraries that CTranslate2 needs:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12
```

The notebook finds them inside the venv automatically. With a GPU it uses Whisper `medium.en` (about 1 GB of GPU memory, safe on a 4 GB card). Without one it falls back to `small.en` on the CPU, which is slower and less accurate, so results will differ from the ones above.

**Data.** The dataset is not in this repository. Get the SHL Hiring Assessment 2026 dataset (`shl-hiring-assessment-2026.zip`) and unzip it in the project folder so that `shl-hiring-assessment-2026/Dataset_Final/train.csv` exists. The notebook stops with a clear message if it cannot find it.

The Whisper and embedding models are downloaded from Hugging Face on first use.

## Running

Open `grammar_scoring_eda.ipynb` from the project folder, choose the **Python (SSHL .venv)** kernel, and run all cells top to bottom.

Every slow step saves its result in `shl-hiring-assessment-2026/working/` and is skipped when that result already exists:

| Step | First run | Cached |
|---|---|---|
| Audio features (985 clips, 8 cores) | decodes all 15 h of audio | about a second |
| Audio cleaning | under 30 s | skipped |
| Whisper transcription | about 77 min on a laptop GPU | skipped |
| Text embedding (CPU) | about 30 s | skipped |
| Model comparison | about 4 min | runs every time |

A fresh clone has no cache, so the first run includes the transcription. After that, with everything cached, a full run takes about 5 minutes. If transcription is interrupted, running the cell again continues where it stopped. To redo a step from scratch, delete its output in `working/`.

## Limitations

- **The strongest features measure overall spoken proficiency, not grammar alone.** Whisper confidence mixes pronunciation, fluency and grammar.
- **No direct grammar-error feature yet.** LanguageTool needs Java, which was not installed.
- **Speaker overlap is unknown.** If one speaker recorded several clips, the cross-validated error is somewhat optimistic.
- **Labels are noisy.** 74 % are whole numbers, which suggests about half a point of rater disagreement, so an error far below 0.5 is probably out of reach.
- **45-second clips are the weak spot:** only 149 in training, but 65 % of test.
- **Predictions are pulled towards the middle.** Clips scored 2 are predicted around 2.7, clips scored 5 around 4.5.

## Next improvements

In order of expected value:

1. **Grammar-error features** from a grammar-correction model, such as corrections per 100 words. This is the most direct measure of what is being scored.
2. **Audio embeddings** from the Whisper encoder or wav2vec2, to capture pronunciation and fluency directly from the sound.
3. **Real pauses** from a voice-activity detector on the cleaned audio.
4. **More weight on 45-second clips** during training, since they dominate the test set.
