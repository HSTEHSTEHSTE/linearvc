# cf_tts/dataset.py
import io
import os
import json, yaml
import queue
import random
import re
import tarfile
import threading
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch, torchaudio
from torch.utils.data import Dataset, IterableDataset, Sampler, get_worker_info


class TTSDatum:
    """
    One training example
    """
    def __init__(
        self,
        wav_path: str,
        text: str,
        speaker_id: str,
        num_frames: int,
        accent: str = None,
        prompt_wav_path: str = None,
    ):
        self.wav_path = wav_path
        self.text = text
        self.speaker_id = speaker_id
        self.num_frames = num_frames
        self.accent = accent
        self.prompt_wav_path = prompt_wav_path


class TTSDataset(Dataset):
    """
    TTS dataset used by CF-TTS.
    """

    def __init__(
        self,
        config_file_path: str,
        split: str = 'train', # train, dev, test
        data: list[TTSDatum] = None,
    ):
        self.resamplers = {}

        # read config
        with open(config_file_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        if data is not None:
            self.data = data

        else:
            self.data: List[TTSDatum] = []

            # load data
            print("Reading Data")
            if self.config['data'][split]['dataset'] == 'librispeech':
                for subset in self.config['data'][split]['librispeech_subsets']:
                    print("Processing ", subset)
                    transcript_files = list((Path(self.config['data']['librispeech_transcript_path']) / subset).glob('*.txt'))
                    spks = []
                    for transcript_file in tqdm(transcript_files):
                        spk = transcript_file.stem
                        spks.append(spk)
                        with open(transcript_file, 'r') as file:
                            for line in file:
                                line = line.strip().split('|')
                                text = line[1].strip()
                                filename = line[0].strip()
                                filename_elements = filename.split('-')
                                wav_path = Path(self.config['data']['librispeech_audio_path']) / subset / spk / filename_elements[1] / (filename + '.flac')
                                num_frames = torchaudio.info(wav_path, backend='soundfile').num_frames
                                self.data.append(TTSDatum(
                                    wav_path=wav_path,
                                    text=text,
                                    speaker_id=spk,
                                    num_frames=num_frames
                                ))
            elif self.config['data'][split]['dataset'] == 'commonvoice':
                audio_root = Path(self.config['data']['commonvoice_root_path']) / 'clips'
                for subset_file in self.config['data'][split]['commonvoice_subsets']:
                    print("Processing ", subset_file)
                    subset_df = pd.read_csv(subset_file, sep='|', header=0, index_col=None, quoting=3)
                    for entry in tqdm(subset_df.iterrows(), total=subset_df.shape[0]):
                        if self.config['data']['accent']['use_accent']:
                            self.data.append(TTSDatum(
                                wav_path=audio_root / entry[1].path,
                                text=entry[1]['sentence'],
                                speaker_id=entry[1]['client_id'],
                                num_frames=int(entry[1]['duration'] * 16000),
                                accent=entry[1]['accents']
                            ))
                        else:
                            self.data.append(TTSDatum(
                                wav_path=audio_root / entry[1].path,
                                text=entry[1]['sentence'],
                                speaker_id=entry[1]['client_id'],
                                num_frames=int(entry[1]['duration'] * 16000)
                            ))
            self.data.sort(key=lambda x: x.num_frames)
            if self.config['data'][split]['dataset'] == 'commonvoice':
                self._assign_same_speaker_prompt_paths()

    def _assign_same_speaker_prompt_paths(self):
        """Attach a distinct >=3-second utterance from the same speaker.

        These paths are used only by validation. Keeping the lookup on the
        datum avoids ever reusing the target utterance as its style prompt.
        """
        prompt_num_samples = int(3.0 * self.config['data']['sampling_rate'])
        candidates_by_speaker = {}
        for datum in self.data:
            if datum.num_frames >= prompt_num_samples:
                candidates_by_speaker.setdefault(datum.speaker_id, []).append(datum)

        for datum in self.data:
            candidates = sorted(
                candidates_by_speaker.get(datum.speaker_id, []),
                key=lambda candidate: str(candidate.wav_path),
            )
            for candidate in candidates:
                if candidate.wav_path != datum.wav_path:
                    datum.prompt_wav_path = candidate.wav_path
                    break

    # -------------------------
    # torch dataset
    # -------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if item.wav_path.suffix == '.mp3':
            wav, sr = torchaudio.load(item.wav_path)
        else:
            wav, sr = torchaudio.load(item.wav_path)
        if sr != 16000:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = self.resamplers[sr](wav)

        wav = wav.squeeze(0)

        wav = wav[: self.config['data']['max_audio_len']]

        return {
            "wav": wav,
            "text": item.text,
            "speaker": item.speaker_id,
            "accent": item.accent,
            "wav_path": item.wav_path,
            "prompt_wav_path": item.prompt_wav_path,
        }


class EmiliaStreamingDataset(IterableDataset):
    """Stream Emilia WebDataset shards from the Hugging Face Hub.

    The Emilia repository is gated.  Authenticate once with ``huggingface-cli
    login`` or set the token named by ``data.<split>.emilia.hf_token_env``
    (``HF_TOKEN`` by default) before starting training.  Audio and matching
    JSON metadata are read progressively from the tar shards; no local copy of
    the corpus or precomputed manifest is required.
    """

    _AUDIO_EXTENSIONS = ("flac", "wav", "mp3", "ogg", "opus", "m4a")

    def __init__(self, config_file_path: str, split: str = "train"):
        super().__init__()
        with open(config_file_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

        self.split = split
        split_config = self.config["data"][split]
        self.emilia_config = split_config.get("emilia", {})
        self.repo_id = self.emilia_config.get("repo_id", "amphion/Emilia-Dataset")
        self.revision = self.emilia_config.get("revision", "main")
        self.directory = self.emilia_config.get("directory", "Emilia").strip("/")
        self.languages = set(self.emilia_config.get("languages", []))
        self.max_shards = self.emilia_config.get("max_shards")
        self.shuffle_seed = int(self.emilia_config.get("shuffle_seed", self.config["training"]["random_seed"]))
        self.token = os.environ.get(self.emilia_config.get("hf_token_env", "HF_TOKEN"))
        self.text_key = self.emilia_config.get("text_key", "text")
        self.speaker_key = self.emilia_config.get("speaker_key", "speaker_id")
        self.json_key = self.emilia_config.get("json_key", "json")
        self.min_audio_samples = int(
            float(self.emilia_config.get("min_audio_seconds", 0.0))
            * self.config["data"]["sampling_rate"]
        )
        self.max_audio_len = self.config["data"]["max_audio_len"]
        self.shard_max_retries = int(split_config.get("streaming_shard_max_retries", 8))
        self.shard_retry_backoff_seconds = float(
            split_config.get("streaming_shard_retry_backoff_seconds", 5.0)
        )
        self.shard_retry_max_backoff_seconds = float(
            split_config.get("streaming_shard_retry_max_backoff_seconds", 60.0)
        )
        # A complete tar is staged locally before WebDataset reads it.  In a
        # Slurm job the launcher places this under node-local TMPDIR; direct
        # launches can override it explicitly with the same environment var.
        shard_cache_dir = os.environ.get(
            "CF_TTS_LOCAL_SHARD_CACHE_DIR",
            split_config.get("streaming_local_shard_cache_dir"),
        )
        if not shard_cache_dir:
            shard_cache_dir = os.path.join(
                os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                "cf_tts_emilia_shards",
            )
        self.local_shard_cache_dir = Path(shard_cache_dir).expanduser().resolve()
        self.resume_cursor = None
        self.shards = self._list_shards()

    def _list_shards(self) -> List[str]:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError(
                "Emilia streaming requires huggingface_hub. Install it in the training environment."
            ) from exc

        prefix = f"{self.directory}/"
        api = HfApi(token=self.token)
        shards = []
        for entry in api.list_repo_tree(
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            recursive=True,
        ):
            path = getattr(entry, "path", "")
            if not path.startswith(prefix) or not path.endswith((".tar", ".tar.gz", ".tgz")):
                continue
            relative_path = path[len(prefix):]
            language = relative_path.split("/", 1)[0]
            if self.languages and language not in self.languages:
                continue
            shards.append(path)

        shards.sort()
        if self.max_shards is not None:
            shards = shards[: int(self.max_shards)]
        if not shards:
            languages = ", ".join(sorted(self.languages)) or "all languages"
            raise RuntimeError(
                f"No Emilia shards found under '{self.directory}' for {languages}. "
                "Check the Hugging Face access token and Emilia configuration."
            )
        print(f"Streaming {len(shards)} Emilia shards from {self.repo_id}:{self.directory}", flush=True)
        return shards

    def set_resume_cursor(self, cursor: Optional[Dict[str, Any]]):
        """Resume at the next deterministic batch emitted by the batch dataset."""
        if cursor is None:
            self.resume_cursor = None
            return
        required = {"shard_index", "batch_index", "shard_path"}
        missing = required.difference(cursor)
        if missing:
            raise ValueError(f"Invalid Emilia resume cursor; missing {sorted(missing)}")
        self.resume_cursor = {
            "shard_index": int(cursor["shard_index"]),
            "batch_index": int(cursor["batch_index"]),
            "shard_path": str(cursor["shard_path"]),
        }

    def _local_shard_path(self, shard_path: str) -> Path:
        """Return a safe node-local target path for one repository filename."""
        local_path = (self.local_shard_cache_dir / shard_path).resolve()
        if not local_path.is_relative_to(self.local_shard_cache_dir):
            raise ValueError(f"Invalid Emilia shard path: {shard_path!r}")
        return local_path

    def _stage_shard_file(self, shard_path: str) -> Path:
        """Reliably download one complete shard before any tar iteration.

        ``hf_hub_download`` downloads through an on-disk cache and exposes the
        requested local path only after the transfer completes.  Thus a read
        timeout cannot surface halfway through the subsequent tar traversal.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "Emilia streaming requires huggingface_hub. Install it in the training environment."
            ) from exc

        local_path = self._local_shard_path(shard_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Staging Emilia shard locally: {shard_path}", flush=True)
        downloaded_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=shard_path,
            repo_type="dataset",
            revision=self.revision,
            token=self.token,
            local_dir=str(self.local_shard_cache_dir),
        )
        downloaded_path = Path(downloaded_path).resolve()
        if downloaded_path != local_path:
            raise RuntimeError(
                f"Hugging Face staged {shard_path} at unexpected path {downloaded_path}; "
                f"expected {local_path}."
            )
        if not local_path.is_file() or local_path.stat().st_size == 0:
            raise RuntimeError(f"Emilia staging did not produce a complete file: {local_path}")
        return local_path

    def _open_shard_stream(self, shard_path: str):
        """Create a local WebDataset stream for one fully staged tar shard."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Emilia streaming requires the Hugging Face datasets package. Install it in the training environment."
            ) from exc

        local_path = self._stage_shard_file(shard_path)
        stream = load_dataset(
            "webdataset",
            data_files=[str(local_path)],
            split="train",
            streaming=True,
        ).decode(False)
        return stream, local_path

    @staticmethod
    def _is_transient_shard_error(error: BaseException) -> bool:
        if isinstance(error, (tarfile.ReadError, TimeoutError, ConnectionError, EOFError, OSError)):
            return True
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "timed out",
                "timeout",
                "unexpected end of data",
                "incomplete read",
                "remote data host",
                "connection reset",
                "connection aborted",
                "chunkedencodingerror",
                "temporarily unavailable",
                "name or service not known",
                "temporary failure in name resolution",
                "network is unreachable",
                "readtimeout",
                "connecterror",
            )
        )

    def _iter_shard(self, shard_path: str, shard_index: int) -> Iterator[Dict[str, Any]]:
        """Yield one shard, retrying a broken HTTP/tar stream without duplicates.

        A retry reopens the tar and skips raw WebDataset records already seen in
        this shard. This is slower than a clean transfer but preserves the
        deterministic sample order and avoids replaying completed records.
        """
        next_raw_sample_index = 0
        for attempt in range(self.shard_max_retries + 1):
            staged_path = None
            try:
                opened_stream = self._open_shard_stream(shard_path)
                # Retain support for the focused recovery test's lightweight
                # stream override while production returns (stream, path).
                if isinstance(opened_stream, tuple):
                    stream, staged_path = opened_stream
                else:
                    stream = opened_stream
                for raw_sample_index, sample in enumerate(stream):
                    if raw_sample_index < next_raw_sample_index:
                        continue
                    next_raw_sample_index = raw_sample_index + 1
                    try:
                        item = self._prepare_sample(sample)
                    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
                        continue
                    if item is not None:
                        item["_shard_index"] = shard_index
                        item["_shard_path"] = shard_path
                        item["_raw_sample_index"] = raw_sample_index
                        yield item
                if staged_path is not None and staged_path.exists():
                    staged_path.unlink()
                return
            except Exception as exc:
                # A completed local tar which fails to parse is corrupt; drop
                # it before retrying.  Failed downloads leave Hugging Face's
                # resumable partial cache intact because staged_path is None.
                if staged_path is not None and staged_path.exists():
                    staged_path.unlink()
                if not self._is_transient_shard_error(exc) or attempt >= self.shard_max_retries:
                    raise RuntimeError(
                        f"Emilia shard {shard_path} failed after raw sample {next_raw_sample_index}"
                    ) from exc
                delay = min(
                    self.shard_retry_max_backoff_seconds,
                    self.shard_retry_backoff_seconds * (2 ** attempt),
                )
                print(
                    f"Retrying Emilia shard {shard_path} from raw sample {next_raw_sample_index} "
                    f"after {type(exc).__name__}: {exc}. Retry {attempt + 1}/{self.shard_max_retries} "
                    f"in {delay:.1f}s.",
                    flush=True,
                )
                time.sleep(delay)

    def _audio_and_metadata(self, sample: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
        audio_bytes = None
        metadata = sample.get(self.json_key)
        for key, value in sample.items():
            extension = key.rsplit(".", 1)[-1].lower()
            if extension in self._AUDIO_EXTENSIONS and value is not None:
                if isinstance(value, dict):
                    audio_bytes = value.get("bytes")
                elif isinstance(value, (bytes, bytearray)):
                    audio_bytes = bytes(value)
                if audio_bytes is not None:
                    break

        if isinstance(metadata, (bytes, bytearray)):
            metadata = json.loads(metadata.decode("utf-8"))
        elif isinstance(metadata, str):
            metadata = json.loads(metadata)
        if not isinstance(metadata, dict):
            metadata = {}
        if audio_bytes is None:
            raise ValueError("Emilia WebDataset sample has no supported audio member")
        return audio_bytes, metadata

    def _prepare_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        audio_bytes, metadata = self._audio_and_metadata(sample)
        text = metadata.get(self.text_key, sample.get(self.text_key))
        if not isinstance(text, str) or not text.strip():
            return None

        audio_info = torchaudio.info(io.BytesIO(audio_bytes))
        num_samples = round(
            audio_info.num_frames
            * self.config["data"]["sampling_rate"]
            / audio_info.sample_rate
        )
        # Emilia is segmented to 3--30 seconds. Reject anomalous long clips
        # instead of truncating them, so text and waveform stay aligned.
        if num_samples > self.max_audio_len:
            return None
        if num_samples < self.min_audio_samples:
            return None

        speaker = metadata.get(self.speaker_key, sample.get(self.speaker_key, sample.get("__key__", "emilia")))
        return {
            # Keep compressed audio until the length-sorted batch is assembled.
            # Decoding occurs in TTS_Collate inside the sole DataLoader worker.
            "audio_bytes": audio_bytes,
            "num_samples": num_samples,
            "_shard_url": sample.get("__url__", ""),
            "text": text.strip(),
            "speaker": str(speaker),
            "accent": None,
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_shards = list(self.shards[worker_id::num_workers])
        if not worker_shards:
            return
        random.Random(self.shuffle_seed + worker_id).shuffle(worker_shards)

        resume_cursor = self.resume_cursor
        self.resume_cursor = None
        start_shard_index = 0
        if resume_cursor is not None:
            if num_workers != 1:
                raise RuntimeError("Emilia cursor resume requires a single DataLoader worker.")
            start_shard_index = resume_cursor["shard_index"]
            if start_shard_index < 0 or start_shard_index >= len(worker_shards):
                raise ValueError(f"Emilia resume shard index is out of range: {start_shard_index}")
            if worker_shards[start_shard_index] != resume_cursor["shard_path"]:
                raise ValueError(
                    "Emilia shard order changed since checkpoint; refusing to resume a different stream."
                )

        for shard_index, shard_path in enumerate(worker_shards):
            if shard_index < start_shard_index:
                continue
            yield from self._iter_shard(shard_path, shard_index)


class EmiliaLengthSortedBatchDataset(IterableDataset):
    """Sort one streamed shard by length, then form dynamic padded batches.

    Batches use a linear duration budget: if ``reference_batch_size`` clips of
    ``reference_audio_seconds`` are safe, a batch whose longest clip is ``L``
    seconds has at most ``floor(reference_batch_size * reference_seconds / L)``
    elements.  This deliberately overestimates memory for short clips instead
    of relying on the model's more complex length scaling.
    """

    def __init__(self, dataset: EmiliaStreamingDataset, config_file_path: str, split: str = "train"):
        super().__init__()
        self.dataset = dataset
        with open(config_file_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        split_config = config["data"][split]
        self.reference_audio_samples = int(
            float(split_config["streaming_reference_audio_seconds"])
            * config["data"]["sampling_rate"]
        )
        self.reference_batch_size = int(
            os.environ.get(
                "CF_TTS_STREAMING_REFERENCE_BATCH_SIZE",
                split_config["streaming_reference_batch_size"],
            )
        )
        if self.reference_batch_size < 1:
            raise ValueError("streaming_reference_batch_size must be positive")
        self.max_batch_size = int(
            os.environ.get(
                "CF_TTS_STREAMING_MAX_BATCH_SIZE",
                split_config["streaming_max_batch_size"],
            )
        )
        if self.max_batch_size < 1:
            raise ValueError("streaming_max_batch_size must be positive")
        self.max_shard_buffer_bytes = int(split_config["streaming_shard_buffer_bytes"])
        self.shuffle_seed = int(config["training"]["random_seed"])
        self.resume_cursor = None

    def set_resume_cursor(self, cursor: Optional[Dict[str, Any]]):
        self.resume_cursor = cursor
        self.dataset.set_resume_cursor(cursor)

    def _batch_capacity(self, longest_num_samples: int) -> int:
        capacity = self.reference_batch_size * self.reference_audio_samples // longest_num_samples
        return max(1, min(self.max_batch_size, capacity))

    def _batches_from_shard(
        self,
        samples: List[Dict[str, Any]],
        shard_index: int,
        first_batch_index: int = 0,
    ) -> Iterator[List[Dict[str, Any]]]:
        samples.sort(key=lambda sample: sample["num_samples"])
        batches = []
        batch = []
        longest_num_samples = 0
        for sample in samples:
            prospective_longest = max(longest_num_samples, sample["num_samples"])
            if batch and len(batch) + 1 > self._batch_capacity(prospective_longest):
                batches.append(batch)
                batch = []
                longest_num_samples = 0
            batch.append(sample)
            longest_num_samples = max(longest_num_samples, sample["num_samples"])
        if batch:
            batches.append(batch)

        random.Random(self.shuffle_seed + shard_index).shuffle(batches)
        if first_batch_index < 0 or first_batch_index > len(batches):
            raise ValueError(
                f"Emilia resume batch index {first_batch_index} is invalid for shard {shard_index} "
                f"with {len(batches)} batches."
            )
        shard_path = samples[0]["_shard_path"]
        for batch_index, batch in enumerate(batches):
            if batch_index < first_batch_index:
                continue
            next_cursor = {
                "shard_index": shard_index,
                "batch_index": batch_index + 1,
                "shard_path": shard_path,
            }
            for sample in batch:
                sample["_stream_cursor"] = next_cursor
            yield batch

    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        current_shard = None
        shard_samples = []
        shard_buffer_bytes = 0
        resume_cursor = self.resume_cursor
        self.resume_cursor = None

        for sample in self.dataset:
            sample_shard = sample["_shard_index"]
            if current_shard is None:
                current_shard = sample_shard
            elif sample_shard != current_shard:
                first_batch_index = (
                    resume_cursor["batch_index"]
                    if resume_cursor is not None and current_shard == resume_cursor["shard_index"]
                    else 0
                )
                yield from self._batches_from_shard(
                    shard_samples, current_shard, first_batch_index
                )
                current_shard = sample_shard
                shard_samples = []
                shard_buffer_bytes = 0

            sample_bytes = len(sample["audio_bytes"])
            shard_buffer_bytes += sample_bytes
            if shard_buffer_bytes > self.max_shard_buffer_bytes:
                raise RuntimeError(
                    f"Emilia shard exceeds streaming_shard_buffer_bytes "
                    f"({self.max_shard_buffer_bytes} bytes). Increase the buffer limit "
                    "or select smaller shards."
                )
            shard_samples.append(sample)

        if shard_samples:
            first_batch_index = (
                resume_cursor["batch_index"]
                if resume_cursor is not None and current_shard == resume_cursor["shard_index"]
                else 0
            )
            yield from self._batches_from_shard(shard_samples, current_shard, first_batch_index)


class BackgroundPrefetchDataset(IterableDataset):
    """Prefetch iterable items with one in-process background thread.

    This avoids PyTorch DataLoader worker processes and their POSIX semaphore
    queues, which can be unavailable on shared Slurm nodes. Items are batches
    in the Emilia path, so the bounded queue controls the additional buffered
    memory beyond the current shard buffer.
    """

    def __init__(self, dataset: IterableDataset, prefetch_items: int = 1):
        super().__init__()
        if prefetch_items < 1:
            raise ValueError("prefetch_items must be at least 1")
        self.dataset = dataset
        self.prefetch_items = prefetch_items

    def set_resume_cursor(self, cursor: Optional[Dict[str, Any]]):
        if not hasattr(self.dataset, "set_resume_cursor"):
            raise TypeError("Wrapped dataset does not support Emilia stream cursors.")
        self.dataset.set_resume_cursor(cursor)

    def __iter__(self):
        item_queue = queue.Queue(maxsize=self.prefetch_items)
        stop_event = threading.Event()
        done = object()

        def put(item):
            while not stop_event.is_set():
                try:
                    item_queue.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def producer():
            try:
                for item in self.dataset:
                    if not put(("item", item)):
                        return
            except BaseException as exc:
                put(("error", exc))
            finally:
                put(("done", done))

        thread = threading.Thread(
            target=producer,
            name="emilia-stream-prefetch",
            daemon=True,
        )
        thread.start()
        try:
            while True:
                kind, payload = item_queue.get()
                if kind == "item":
                    yield payload
                elif kind == "error":
                    raise RuntimeError("Emilia background streaming failed") from payload
                else:
                    return
        finally:
            stop_event.set()
            thread.join(timeout=1)


# -------------------------
# sampler
# -------------------------
class FrameBatchSampler(Sampler):
    """
    Groups samples so that the total number of audio frames
    per batch does not exceed max_frames.

    This gives:
        short utterances → large batches
        long utterances  → small batches
    """

    def __init__(
        self,
        dataset,
        config_file_path: str,
        split: str = 'train'
    ):
        self.dataset = dataset

        # read config
        with open(config_file_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        self.max_frames = self.config['training']['max_frames_per_batch']

        self.indices = list(range(len(dataset)))

        self.batches = []
        batch = []
        total_frames = 0
        for idx in self.indices:
            frames = min(self.dataset.data[idx].num_frames, self.config['data']['max_audio_len'])

            # if this utterance alone is too big, force it into its own batch
            if frames > self.max_frames:
                if batch:
                    self.batches.append(batch)
                    batch = []
                    total_frames = 0
                self.batches.append([idx])
                continue

            if total_frames + frames > self.max_frames and batch:
                self.batches.append(batch)
                batch = []
                total_frames = 0

            batch.append(idx)
            total_frames += frames

        if batch:
            self.batches.append(batch)
        if self.config['data'][split]['shuffle_batches']:
            random.seed(self.config['training']['random_seed'])
            random.shuffle(self.batches)
        self.shuffle = self.config['data'][split]['shuffle_batches_between_epochs']


    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch
        

    def __len__(self):
        # PyTorch allows this to be approximate
        return len(self.batches)


# -------------------------
# batching
# -------------------------

class TTS_Collate:
    """
    Pads variable-length audio and text for batching.
    """
    def __init__(self, cfg):
        self.sampling_rate = cfg['data']['sampling_rate']
        self.max_audio_len = cfg['data']['max_audio_len']
        self.style_prompt_samples = int(3.0 * self.sampling_rate)
        self.resamplers = {}
        self.tokenizer_type = cfg['data']['text_tokenizer']['type']
        self.pre_phonemized = cfg['data']['text_tokenizer']['pre_phonemized']
        if self.tokenizer_type == 'spm':
            import sentencepiece as spm
            self.text_tokenizer = spm.SentencePieceProcessor()
            self.text_tokenizer.load(cfg['data']['text_tokenizer']['tokenizer_file'])
        elif self.tokenizer_type == 'phone':
            import json
            with open(cfg['data']['text_tokenizer']['tokenizer_file'], 'r') as phone_json_file:
                tokens = json.load(phone_json_file)
                self.text_tokenizer = {}
                punctuation_marks = ''
                for category in tokens.keys():
                    category_tokens = tokens[category]
                    for token_id, token in enumerate(category_tokens):
                        self.text_tokenizer[token] = token_id
                        if category == 'punctuations':
                            punctuation_marks += token
            if not self.pre_phonemized:
                from phonemizer.backend import EspeakBackend
                from phonemizer.separator import Separator
                self.separator = Separator(phone='-', word=' ')
                if len(punctuation_marks) > 0:
                    backend = EspeakBackend(
                        language='en-us', 
                        preserve_punctuation=True, 
                        punctuation_marks=punctuation_marks,
                        words_mismatch='ignore'
                    )
                else:
                    backend = EspeakBackend(
                        language='en-us', 
                        preserve_punctuation=True, 
                        words_mismatch='ignore'
                    )
                self.phonemize = backend.phonemize
        
        # accent tokenizer
        self.use_accent = cfg['data']['accent']['use_accent']
        if self.use_accent:
            with open(cfg['data']['accent']['accent_file'], 'r') as file:
                self.accents = json.load(file)['accents']
            self.accent_to_id = {}
            for index, accent in enumerate(self.accents):
                self.accent_to_id[accent] = index

    def _load_wav(self, item: Dict) -> torch.Tensor:
        if "audio_bytes" not in item:
            return item["wav"]

        wav, sampling_rate = torchaudio.load(io.BytesIO(item["audio_bytes"]))
        if wav.size(0) > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)
        if sampling_rate != self.sampling_rate:
            if sampling_rate not in self.resamplers:
                self.resamplers[sampling_rate] = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate,
                    new_freq=self.sampling_rate,
                )
            wav = self.resamplers[sampling_rate](wav)
        if wav.numel() > self.max_audio_len:
            raise ValueError("Emilia clip exceeded max_audio_len after decoding")
        return wav

    def _load_prompt_wav(self, wav_path: Path) -> torch.Tensor:
        wav, sampling_rate = torchaudio.load(wav_path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)
        if sampling_rate != self.sampling_rate:
            if sampling_rate not in self.resamplers:
                self.resamplers[sampling_rate] = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate,
                    new_freq=self.sampling_rate,
                )
            wav = self.resamplers[sampling_rate](wav)
        return wav

    def __call__(self, batch: List[Dict]):
        wavs = [self._load_wav(b) for b in batch]
        speakers = [b["speaker"] for b in batch]
        texts_raw = [str(b["text"]) for b in batch]
        texts = []
        if self.tokenizer_type == 'spm':
            for text_raw in texts_raw:
                texts.append(self.text_tokenizer.encode_as_ids(text_raw))
        elif self.tokenizer_type == 'phone':
            if self.pre_phonemized:
                for text_raw in texts_raw:
                    text_raw = text_raw.split(' ')
                    texts.append([self.text_tokenizer[phone] for phone in text_raw if phone in self.text_tokenizer])
            else:
                phones = self.phonemize(texts_raw, separator=self.separator, strip=True)
                phones = [re.sub(r"""([;:,.!?¡¿—…"«»“”\(\)\{\}\[\]])""", r"-\1", phone) for phone in phones]
                phones = [phone.replace('- ', '-').replace(' ', '-').split('-') for phone in phones]
                for text in phones:
                    texts.append([self.text_tokenizer[token] for token in text if token in self.text_tokenizer])

        if self.use_accent:
            accents_raw = [b["accent"] for b in batch]
            accents = [self.accent_to_id[accent] for accent in accents_raw]
        else:
            accents = []

        lengths = torch.tensor([w.size(0) for w in wavs])
        max_len = max(lengths)

        padded = torch.zeros(len(wavs), max_len)
        for i, w in enumerate(wavs):
            padded[i, : w.size(0)] = w

        prompt_paths = [b.get('prompt_wav_path') for b in batch]
        stream_cursors = [b.get('_stream_cursor') for b in batch]
        stream_cursor = None
        if any(cursor is not None for cursor in stream_cursors):
            if any(cursor != stream_cursors[0] for cursor in stream_cursors):
                raise RuntimeError("A streamed Emilia batch contains inconsistent resume cursors.")
            stream_cursor = stream_cursors[0]
        prompt_wavs = None
        prompt_available = None
        if any(path is not None for path in prompt_paths):
            prompt_wavs = torch.zeros(len(wavs), self.style_prompt_samples)
            prompt_available = torch.zeros(len(wavs), dtype=torch.bool)
            for i, prompt_path in enumerate(prompt_paths):
                if prompt_path is None:
                    continue
                prompt_wav = self._load_prompt_wav(prompt_path)
                if prompt_wav.numel() < self.style_prompt_samples:
                    continue
                prompt_wavs[i] = prompt_wav[: self.style_prompt_samples]
                prompt_available[i] = True

        return {
            "wav": padded,
            "wav_lengths": lengths,
            "text": texts,
            "speaker": speakers,
            "accent": accents,
            "wav_path": [b.get('wav_path') for b in batch],
            "prompt_wav": prompt_wavs,
            "prompt_available": prompt_available,
            "prompt_wav_path": prompt_paths,
            "stream_cursor": stream_cursor,
        }
    
    

if __name__ == "__main__":
    # test dataset object
    dataset = TTSDataset(config_file_path='cf_tts/config/config.yaml')
    print(len(dataset))
    print(dataset.__getitem__(0))

    # test sampler
    sampler = FrameBatchSampler(dataset, config_file_path='cf_tts/config/config.yaml')
    print(len(sampler))
    for batch in sampler:
        print(batch)
