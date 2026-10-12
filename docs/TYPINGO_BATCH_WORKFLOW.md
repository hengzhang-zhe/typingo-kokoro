# Typingo offline audio workflow

Typingo owns content IDs, source text and requested voice variants. Kokoro generates offline audio. Typingo only owns teaching data and final audio; no production Kokoro connection or standalone phoneme exchange is required.

## Export a task

In Typingo Admin, choose learning or preview items and select voices (including Select All). Learning tasks support selected items or a custom filtered count from 1 to 10000. Export the task ZIP. ZIP contains one compact JSON document; JSON and GZIP are also accepted, using the current unversioned `format` contract; older task files must be re-exported from Typingo.

The v5 task contains `format`, `outputMode`, `voices` and `items`. Top-level voice profiles contain only `voice` and `locale`; each item contains `contentId`, `text`, explicit voice IDs and a SHA-256 `snapshot`. Word targets additionally require `pronunciationId`, `locale` and definite dictionary `ipa`. No default voices are added.

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

Word audio uses explicit IPA through the strict English mapping to control the intended reading. Misaki processes normal text and contextual pronunciation inside Kokoro. Unsupported symbols, ambiguous optional forms and absent relationships are reported in the Studio review list and local generation report; successful targets can continue. This is target identity validation, not a guarantee of perceptual quality. Inspect representative generated recordings before publishing. Manifests use `typingo-audio-manifest`; object keys distinguish synthesis identity and runtime version so different readings or configurations cannot overwrite each other. Resume validates target identity, locale and checksums against the task.

## Explicit synthesis controls

Each v5 item requires `synthesis: {mode, speed, overrides}`. `mode` is `ipa` for a word reading, `text` for normal text, or `mixed` for contextual text with pronunciation overrides. Speed is canonicalized to a decimal string between 0.5 and 2.0; the admin export controls default to 1.0. Every learning voice has an ID-independent synthesis identity; previews use their own target paths.

Typingo automatically exports confirmed word annotations with a unique matching pronunciation per locale. Overrides use `{start,end,text,ipa,locale}` with source UTF-16 offsets and an exclusive end, matching the content annotation contract. Misaki first processes the complete text, then only exact whole-token spans are replaced. Overrides affect only their own locale. Invalid spans, unsupported IPA, or tokenizer alignment failures fail explicitly rather than falling back to guessed spelling. Phrase overrides are not automatically exported.

The v2 result manifest carries `identity` and `synthesis`. Synthesis records the actual phoneme chunks, controls, model/config/voice file hashes, package/adapter versions and encoder settings. Stable identity ignores content and pronunciation IDs; cache and object keys also include the loaded runtime fingerprint. Resume validates parameters, runtime provenance and file checksums. Cache hits retain the original synthesis metadata. Audio is MP3 at 24 kHz; model and encoding settings are owned by the generation service rather than asserted by the client. Changing speed or pronunciation overrides creates a distinct variant.

语法学习项使用 `contentType=grammar`，以主练习例句作为任务 `text`；语法结构只用于讲解，不进入音频。语法与句子、长文一样采用 text 或 mixed 生成模式，音频身份保留 grammar 类型，不能改写为 sentence。
