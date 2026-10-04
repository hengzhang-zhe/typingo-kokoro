# Typingo offline audio workflow

Typingo owns content IDs, source text and requested voice variants. Kokoro generates offline audio; no production connection to Kokoro is required.

## Export a task

In Typingo Admin, choose learning or preview items and select voices (including Select All). Learning tasks support selected items or a custom filtered count from 1 to 10000. Export the task ZIP. ZIP contains one compact JSON document; JSON and GZIP are also accepted, using the canonical v3 contract.

The v3 task contains `schemaVersion`, `outputMode`, `voices` and `items`. Top-level voice profiles contain only `voice` and `locale`; each item contains `contentId`, `text`, explicit voice IDs and a SHA-256 `snapshot`. Word targets additionally require `pronunciationId`, `locale` and definite dictionary `ipa`. No default voices are added.

## Generate

Upload the task in Batch Audio Studio at the configured service URL (default port 9000), or run:

```powershell
.\scripts\batch-generate.ps1 -InputFile ".\tasks\typingo-tts-export.zip"
```

The script uses the same import/generate/download batch API as the web UI. Results use ZIP. Reports and generated files remain under `output/batches/<jobId>/` on the generation server.

## Import the result

`outputMode=bundle` (default) produces `audio-manifest.json` plus `audio/`, with each audio file at its manifest object key. Import the ZIP directly in Typingo; it validates all audio files before uploading them to the current object storage and registering associations.

`outputMode=manifest` produces only `audio-manifest.json`, containing the successful updates from the current task. Put the local audio files into object storage manually, preserving object keys, or use the separate audio ZIP download and import that ZIP first. Then import the manifest ZIP. An audio-only ZIP uploads files without creating database associations; a manifest-only ZIP registers associations without uploading audio. The script can download the separate audio ZIP with `-DownloadAudio`.

Generated manifests omit deployment-specific storage fields. Typingo resolves those fields using its configured local/S3/OSS storage. Existing explicit storage fields remain supported. Partial failures are excluded from the manifest; reports remain local. Re-export missing tasks after importing successful results.

Object uploads and database registration are separate operations. If registration fails, uploaded files remain available for a corrected retry. Typingo uploads audio in 4MB chunks (256KB fallback after 413), accepts up to 10GB per file, then processes it as a background job. Manifests are parsed from disk and committed in 500-asset transactions; corrected retries reuse existing associations. Upload and database transactions are separate, so successful earlier batches remain if a later batch fails.

### Large batches

Task files and their expanded JSON are limited to 32MB, with up to 10000 distinct content items and 100000 reading targets; reduce the Typingo export count when text exceeds that bound. Studio supports 4MB task-upload chunks with a 256KB fallback after HTTP 413. Generation runs one batch at a time with bounded concurrent audio workers (calculated once at startup). Interrupted jobs can resume in Studio; intact successful files are reused after checksum validation. Long waveforms are appended to a temporary WAV rather than concatenated in memory; result assets and failures are staged as JSONL. Results split at approximately 256MB of audio or 5000 manifest entries. Every result part is an independent ZIP with its own manifest. Download and import all parts; never concatenate their ZIP bytes. Separate audio downloads also accept `?part=N`. The CLI downloads all result parts sequentially. A single generated audio must fit Typingo's 100MB object limit; the temporary WAV has a 2GB safety bound.

Word audio is synthesized from explicit IPA through a strict English Misaki mapping, never from guessed spelling. Unsupported symbols, ambiguous optional forms and absent relationships are reported in the Studio review list and local generation report; successful targets can continue. This is target identity validation, not a guarantee of perceptual quality. Inspect representative generated recordings before publishing. Manifests use `typingo-audio-manifest/v2`; object keys include a pronunciation snapshot so different readings of one word cannot overwrite each other. Resume validates target identity, locale and checksums against the task.
