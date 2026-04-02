<h1 align="center">Forced Alignment with Hugging Face CTC Models</h1>

[![FreePalestine.Dev](https://freepalestine.dev/header/1)](https://freepalestine.dev)


<p align="center">
  <a href="https://github.com/MahmoudAshraf97/ctc-forced-aligner/actions/workflows/test_build.yml">
    <img src="https://github.com/MahmoudAshraf97/ctc-forced-aligner/actions/workflows/CI.yml/badge.svg"
         alt="Build Status">
  </a>
  <a href="https://github.com/MahmoudAshraf97/ctc-forced-aligner/stargazers">
    <img src="https://img.shields.io/github/stars/MahmoudAshraf97/ctc-forced-aligner.svg?colorA=orange&colorB=orange&logo=github"
         alt="GitHub stars">
  </a>
  <a href="https://github.com/MahmoudAshraf97/ctc-forced-aligner/issues">
        <img src="https://img.shields.io/github/issues/MahmoudAshraf97/ctc-forced-aligner.svg"
             alt="GitHub issues">
  </a>
  <a href="https://github.com/MahmoudAshraf97/ctc-forced-aligner/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/MahmoudAshraf97/ctc-forced-aligner.svg"
             alt="GitHub license">
  </a>
  <a href="https://twitter.com/intent/tweet?text=&url=https%3A%2F%2Fgithub.com%2FMahmoudAshraf97%2Fctc-forced-aligner">
  <img src="https://img.shields.io/twitter/url/https/github.com/MahmoudAshraf97/ctc-forced-aligner.svg?style=social" alt="Twitter">
  </a> 
  </a>
 
</p>

<img src="https://github.blog/wp-content/uploads/2020/09/github-stars-logo_Color.png" alt="drawing" width="25"/> **Please, star the project on github (see top-right corner) if you appreciate my contribution to the community!**

This Python package provides an efficient way to perform forced alignment between text and audio using Hugging Face's pretrained models. It leverages the power of Wav2Vec2, HuBERT, and MMS models for accurate alignment, making it a powerful tool for creating speech corpuses.

### Features
- **Atleast 5X less memory usage:** Improved implementation to use much less memory than TorchAudio forced alignment API.
- **Wide range of language support:** Works with multiple languages including English, Arabic, Russian, German, and 1126 more languages.
- **Flexibility in alignment granularity:** Choose between aligning on a sentence, word, or character level.
- **Customizable alignment parameters:** Control the frequency of `<star>` token insertion, merge threshold for segment merging, and more.
- **Integration with Hugging Face's models:** Leverage the power of pretrained Wav2Vec2, HuBERT, and MMS models for accurate alignment.
- **GPU acceleration:** Utilize your GPU for faster inference.
- **Output in JSON format:** Provides clear and structured alignment results for easy analysis and integration.


### Installation
`FFMPEG` is needed as prerequisite to use

#### Latest version from GitHub
```bash
pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
```
#### Installing locally from source
```bash
git clone https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
cd ctc-forced-aligner
pip install -e .[dev]
```

### Usage

```bash
ctc-forced-aligner --audio_path "path/to/audio.wav" --text_path "path/to/text.txt" --language "eng" --romanize
```

<details>
<summary>Terminal Usage</summary>


### Arguments

| Argument | Description | Default |
|---|---|---|
| `--audio_path` | Path to the audio file | Required |
| `--text_path` | Path to the text file | Required |
| `--language` | Language in ISO 639-3 code | Required |
| `--romanize` | Enable romanization for non-latin scripts or for multilingual models regardless of the language, required when using the default model| False |
| `--split_size` | Alignment granularity: "sentence", "word", or "char" | "word" |
| `--star_frequency` | Frequency of `<star>` token: "segment" or "edges" | "edges" |
| `--merge_threshold` | Merge threshold for segment merging | 0.00 |
| `--alignment_model` | Name of the alignment model | [MahmoudAshraf/mms-300m-1130-forced-aligner](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) |
| `--compute_dtype` | Compute dtype for inference | "float32" |
| `--batch_size` | Batch size for inference | 4 |
| `--window_size` | Window size in seconds for audio chunking | 30 |
| `--context_size` | Overlap between chunks in seconds | 2 |
| `--attn_implementation` | Attention implementation | "eager" |
| `--device` | Device to use for inference: "cuda" or "cpu" | "cuda" if available, else "cpu" |

### Examples

```bash
# Align an English audio file with the text file
ctc-forced-aligner --audio_path "english_audio.wav" --text_path "english_text.txt" --language "eng" --romanize

# Align a Russian audio file with romanized text
ctc-forced-aligner --audio_path "russian_audio.wav" --text_path "russian_text.txt" --language "rus" --romanize

# Align on a sentence level
ctc-forced-aligner --audio_path "audio.wav" --text_path "text.txt" --language "eng" --split_size "sentence" --romanize

# Align using a model with native vocabulary
ctc-forced-aligner --audio_path "audio.wav" --text_path "text.txt" --language "ara" --alignment_model "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
```

</details>


<details>
<summary>Python Usage</summary>
  
### Python Usage
```python
import torch
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

audio_path = "your/audio/path"
text_path = "your/text/path"
language = "iso" # ISO-639-3 Language code
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16


alignment_model, alignment_tokenizer = load_alignment_model(
    device,
    dtype=torch.float16 if device == "cuda" else torch.float32,
)

audio_waveform = load_audio(audio_path, alignment_model.dtype, alignment_model.device)


with open(text_path, "r") as f:
    lines = f.readlines()
text = "".join(line for line in lines).replace("\n", " ").strip()

emissions, stride = generate_emissions(
    alignment_model, audio_waveform, batch_size=batch_size
)

tokens_starred, text_starred = preprocess_text(
    text,
    romanize=True,
    language=language,
)

segments, scores, blank_token = get_alignments(
    emissions,
    tokens_starred,
    alignment_tokenizer,
)

spans = get_spans(tokens_starred, segments, blank_token)

word_timestamps = postprocess_results(text_starred, spans, stride, scores)
```

</details>

### Output

The alignment results will be saved to a file containing the following information in JSON format:

- **`text`:** The aligned text.
- **`segments`:** A list of segments, each containing the start and end time of the corresponding text segment.
<details>
<summary>JSON</summary>

```json
{
  "text": "This is a sample text to be aligned with the audio.",
  "segments": [
    {
      "start": 0.000,
      "end": 1.234,
      "text": "This"
    },
    {
      "start": 1.234,
      "end": 2.567,
      "text": "is"
    },
    {
      "start": 2.567,
      "end": 3.890,
      "text": "a"
    },
    {
      "start": 3.890,
      "end": 5.213,
      "text": "sample"
    },
    {
      "start": 5.213,
      "end": 6.536,
      "text": "text"
    },
    {
      "start": 6.536,
      "end": 7.859,
      "text": "to"
    },
    {
      "start": 7.859,
      "end": 9.182,
      "text": "be"
    },
    {
      "start": 9.182,
      "end": 10.405,
      "text": "aligned"
    },
    {
      "start": 10.405,
      "end": 11.728,
      "text": "with"
    },
    {
      "start": 11.728,
      "end": 13.051,
      "text": "the"
    },
    {
      "start": 13.051,
      "end": 14.374,
      "text": "audio."
    }
  ]
}
```
</details>

## FastAPI HTTP Service

This project now includes an HTTP service for forced alignment.

### Start Service Locally

```bash
pip install -e .
ctc-forced-aligner-api
```

The server starts at `http://0.0.0.0:8000`.

### API Endpoints

- `GET /health`: Service health and runtime config summary
- `POST /align`: Upload audio + text and get JSON alignment result

### `POST /align` Parameters (multipart/form-data)

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `audio` | file | yes | - | Input audio file |
| `text` | string | yes | - | Transcript text |
| `language` | string | no | `en` | `en` / `zh` or any ISO-639-3 code |
| `romanize` | bool | no | `false` | Enable uroman normalization |
| `split_size` | string | no | `word` | `sentence`, `word`, `char` |
| `star_frequency` | string | no | `edges` | `segment` or `edges` |
| `merge_threshold` | float | no | `0.0` | Merge close segment gaps |
| `batch_size` | int | no | env default | Per-request inference batch size |
| `window_size` | float | no | env default | Chunk size in seconds |
| `context_size` | float | no | env default | Chunk overlap in seconds |

### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/align" \
  -F "audio=@./audio.wav" \
  -F "text=This is a sample text to align." \
  -F "language=en" \
  -F "split_size=word" \
  -F "star_frequency=edges"
```

### Example Response

```json
{
  "text": "This is a sample text to align.",
  "language": "eng",
  "segments": [
    {
      "start": 0.02,
      "end": 0.61,
      "text": "This",
      "score": -1.23
    }
  ],
  "model": "MahmoudAshraf/mms-300m-1130-forced-aligner",
  "device": "cpu"
}
```

## Docker

The provided `Dockerfile` builds the C++ extension, installs `ffmpeg`, and pre-downloads
the alignment model into the image.

### Build

```bash
docker build \
  --build-arg ALIGN_MODEL=MahmoudAshraf/mms-300m-1130-forced-aligner \
  -t ctc-aligner-api:latest .
```

### Run

```bash
docker run --rm -p 8000:8000 \
  -e ALIGN_MAX_CONCURRENCY=2 \
  -e ALIGN_BATCH_SIZE=4 \
  ctc-aligner-api:latest
```

### Runtime Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ALIGN_MODEL` | `MahmoudAshraf/mms-300m-1130-forced-aligner` | Hugging Face model ID or local path |
| `ALIGN_DEVICE` | `cpu` in Dockerfile | `cpu` or `cuda` |
| `ALIGN_COMPUTE_DTYPE` | `float32` in Dockerfile | `float32`, `float16`, `bfloat16` |
| `ALIGN_MAX_CONCURRENCY` | `2` in Dockerfile | Maximum concurrent requests entering inference |
| `ALIGN_MAX_WORKERS` | same as concurrency | Python worker threads for blocking inference |
| `ALIGN_BATCH_SIZE` | `4` | Default inference batch size |
| `ALIGN_WINDOW_SIZE` | `30` | Audio chunk size (seconds) |
| `ALIGN_CONTEXT_SIZE` | `2` | Chunk overlap (seconds) |

### Notes for GPU Containers

- Use an NVIDIA CUDA base image if you want GPU inference.
- Set `ALIGN_DEVICE=cuda` and a compatible `ALIGN_COMPUTE_DTYPE` (`float16`/`bfloat16`).
- Ensure host has NVIDIA Container Toolkit enabled (`--gpus all`).

### Lightweight Benchmark Script

Use `scripts/benchmark_api.py` to quickly benchmark the `/align` endpoint with configurable
concurrency and request count.

```bash
python scripts/benchmark_api.py \
  --url http://127.0.0.1:8000/align \
  --audio ./audio.wav \
  --text "This is a sample benchmark text." \
  --language en \
  --requests 30 \
  --concurrency 4 \
  --warmup 2
```

Save a JSON report:

```bash
python scripts/benchmark_api.py \
  --url http://127.0.0.1:8000/align \
  --audio ./audio.wav \
  --text-file ./text.txt \
  --requests 50 \
  --concurrency 5 \
  --output ./bench_report.json
```



### Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

### License

This project is licensed under the BSD License, note that the default model has CC-BY-NC 4.0 License, so make sure to use a different model for commercial usage.

### Acknowledgements

This project is based on the work of FAIR MMS team.

