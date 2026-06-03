# ComfyUI-Zegami

[![CI](https://github.com/zegami/comfyui-zegami/actions/workflows/ci.yml/badge.svg)](https://github.com/zegami/comfyui-zegami/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

> Push every ComfyUI generation — **image or video** — into a [Zegami](https://zegami.com)
> collection for visual triage, comparison, and curation.

Generating thousands of variations and triaging them by scrolling a folder? Drop
the **Zegami Batch Export** node onto any workflow and every output lands in a
Zegami collection — a filterable visual grid with similarity search and UMAP
clustering. Surface the best 50 of 5,000, audit a LoRA training set for
duplicates, compare samplers/CFG/seeds across a grid, or publish a shareable
gallery as the client deliverable.

![demo](assets/demo.gif)

## Install

**ComfyUI Manager** (recommended): search for "Zegami" and install.

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/zegami/comfyui-zegami
pip install -r comfyui-zegami/requirements.txt   # just `requests`
```
Video export needs **ffmpeg** on your PATH (a system binary, not a pip package).

## Authenticate

Create a collection-scoped API key in Zegami (**Collection → Settings → API access**)
— it can only write to that one collection, so it's safe to keep in a shared
workflow file. Provide it in priority order:

1. `ZEGAMI_API_KEY` environment variable (recommended)
2. `~/.zegami/config.json`: `{ "api_key": "zeg_…", "endpoint": "https://…" }`
3. the node's `api_key_override` input (last resort)

The key is **never** read from or written into a workflow JSON — published
workflows stay credential-free.

## The node

| Input | Notes |
|---|---|
| `images` | Standard ComfyUI IMAGE batch — **the primary path**: each item in the batch is a separate still. |
| `video` | Frames tensor `[frames,h,w,c]` (Wan / Hunyuan / LTX / Mochi) → one MP4, or a VHS_VIDEO passthrough. |
| `collection_id` | Target Zegami collection. |
| `tags`, `notes` | Free-form, attached to every output. |
| `fps` | Frame rate for video encoding (default 16). |
| `api_key_override`, `endpoint_override` | Fallbacks; prefer env / config. |
| `enabled` | Toggle upload off during iteration without removing the node. |

Outputs: `images` (pass-through, so it sits inline before a Save node) and
`upload_status` (JSON for chaining a notification node).

The node captures the **full prompt graph + workflow JSON + execution metadata**
as one opaque blob. In Zegami, the **Calculated Columns → From JSON path** tool
(with one-click smart defaults for KSampler / LoraLoader / CheckpointLoader /
CLIPTextEncode) turns `prompt.6.inputs.seed` into a `Seed` column, etc. — no need
for the node to keep pace with every node type.

## How it works

Per batch the node: encodes media locally first (the fail-soft anchor) → builds
a zip + a `metadata.csv` (with a `_comfy_json` column) → stages + enqueues via
Zegami's [ingest contract](https://github.com/zegami/zegami/blob/main/docs/zegami-ingest-contract-v1.md)
on a background queue. A Zegami failure **never** breaks your generation — on a
permanent failure it writes a `.zegami-pending` sidecar next to the media.

| Scenario | Behaviour |
|---|---|
| No API key found | Save locally, return error status. Generation completes. |
| Network failure | Retry with backoff; then a `.zegami-pending` sidecar. |
| Invalid collection / 4xx | Clear error in `upload_status`. Generation completes. |
| Video encode fails | Upload the first frame as a still instead. |
| Partial metadata | Upload whatever was captured — never fail on metadata. |

## Status & roadmap

- **Images: fully supported** end-to-end today.
- **Video: grid thumbnails + badges work** via the poster + `media_kind`/`duration`
  columns. Full in-app video *playback* (inspector + hover preview) needs the
  original MP4 served from the collection's `raw_assets/`, which the current
  zip-ingest path doesn't populate — tracked as a follow-up (an AARO-style video
  ingest or a dedicated raw-asset upload step).
- v1.1: `.zegami-pending` retry CLI; in-node progress.
- v1.2: a `ZegamiCollectionInput` node to pull triaged images back into a workflow.
- v1.3: deeper smart-flattening learned from real workflows.

MIT licensed. Built by Zegami.
