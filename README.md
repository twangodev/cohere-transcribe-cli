# cohere-transcribe-cli

[![PyPI](https://img.shields.io/pypi/v/cohere-transcribe-cli.svg?logo=pypi&logoColor=white)](https://pypi.org/project/cohere-transcribe-cli/)
[![Python](https://img.shields.io/pypi/pyversions/cohere-transcribe-cli.svg?logo=python&logoColor=white)](https://pypi.org/project/cohere-transcribe-cli/)
[![License](https://img.shields.io/pypi/l/cohere-transcribe-cli.svg)](LICENSE)

Run [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
locally on any PyAV/FFmpeg-compatible audio or video file. 14 languages,
GPU-aware, long-form audio handled automatically.

## Run

```bash
uvx --from cohere-transcribe-cli cohere MEDIA_FILE
```

## Install

```bash
uv tool install cohere-transcribe-cli
cohere MEDIA_FILE
```

The first transcription downloads the 2B model (~4 GB) into the Hugging Face
cache.

## More examples

```bash
cohere talk.m4a -l de -o talk.txt
cohere clip.wav -q > out.txt
```

Status (decode/load/inference time, real-time factor, chunk count) goes to
stderr. The transcript goes to stdout. Run `cohere --help` for every flag.

## License

MIT
