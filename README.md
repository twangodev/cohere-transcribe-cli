# cohere-transcribe-cli

[![PyPI](https://img.shields.io/pypi/v/cohere-transcribe-cli.svg)](https://pypi.org/project/cohere-transcribe-cli/)
[![Python](https://img.shields.io/pypi/pyversions/cohere-transcribe-cli.svg)](https://pypi.org/project/cohere-transcribe-cli/)
[![License](https://img.shields.io/pypi/l/cohere-transcribe-cli.svg)](LICENSE)

Run [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
locally on any audio or video file. 14 languages, GPU-aware, long-form audio
handled automatically.

## Install

```bash
uvx run cohere-transcribe-cli
```

First run downloads the 2B model (~4 GB) into the Hugging Face cache.

## Use

```bash
cohere talk.mp4                    # auto language=en, auto device, panel-free
cohere talk.m4a -l de -o talk.txt  # German, write transcript to file
cohere clip.wav -q > out.txt       # quiet: raw transcript on stdout only
```

Status (decode/load/inference time, real-time factor, chunk count) goes to
stderr. The transcript goes to stdout. `cohere --help` lists every flag.

## License

MIT
