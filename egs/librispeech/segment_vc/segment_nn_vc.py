import argparse
from collections import defaultdict, Counter
import math
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio
import torch.nn.functional as F
import json

device = "cuda"


def save_seg(out_dir: Path, idx: int, wav_1ch: torch.Tensor, sr: int = 16000):
    out_dir.mkdir(parents=True, exist_ok=True)
    # wav_1ch: expected shape [1, N] on CPU
    torchaudio.save(str(out_dir / f"{idx:06d}.wav"), wav_1ch, sr)


def check_argv():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--unit_distance_type",
        type=str,
        choices=["mean", "dtw", "mean_first_last"],
        help="mean, dtw, mean_first_last",
        default="mean_first_last",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="mean pooled length scaling",
        default=0.0,
    )
    parser.add_argument(
        "--lam",
        type=float,
        help="DTW length penalty",
        default=0.1,
    )
    parser.add_argument(
        "--bag_silence",
        action="store_true",
        help="put silence into the bag of hypotheses",
    )
    parser.add_argument(
        "--segment_splits",
        type=int,
        default=1,
        help="Split each segment into this many equal portions. Default: 1.",
    )
    parser.add_argument(
        "--max-segment-length",
        type=int,
        default=0,
        help=(
            "If >0, also add ALL contiguous segments from each target-speaker audio "
            "whose length is <= this many frames."
        ),
    )
    parser.add_argument(
        "--min-segment-length",
        type=int,
        default=1,
        help=(
            "Minimum segment length in feature frames. Segments shorter than this are skipped. "
            "Does not apply to aligned segments."
        ),
    )
    parser.add_argument(
        "--source-segment-length",
        type=int,
        default=0,
        help=(
            "If >0, ignore forced alignment for the source and instead use sliding "
            "windows of this many feature frames."
        ),
    )
    parser.add_argument(
        "--source-stride",
        type=int,
        default=0,
        help=(
            "Stride in feature frames between consecutive source windows when "
            "--source-segment-length > 0."
        ),
    )
    parser.add_argument(
        "--target_use_alignment",
        action="store_true",
        help=(
            "If set, build target bag from forced-aligned L2-ARCTIC segments and optional silence. "
            "If not set, do not use alignments for target bag."
        ),
    )

    parser.add_argument(
        "--w_mean",
        type=float,
        default=5.0,
        help="Weight for mean/pooled distance",
    )
    parser.add_argument(
        "--w_first",
        type=float,
        default=1.0,
        help="Weight for first-frame distance",
    )
    parser.add_argument(
        "--w_last",
        type=float,
        default=1.0,
        help="Weight for last-frame distance",
    )
    parser.add_argument(
        "--label_bonus",
        type=float,
        default=0.05,
        help="Small distance bonus subtracted when source/target labels match.",
    )

    return parser.parse_args()


def overlap_add_average(chunks, starts, total_len, device=None):
    """
    chunks: list of [Li, D] tensors
    starts: list of start indices in output-frame coordinates
    total_len: total number of frames in final sequence
    returns: [total_len, D]
    """
    assert len(chunks) == len(starts)

    if len(chunks) == 0:
        return None

    D = chunks[0].shape[1]
    dev = device if device is not None else chunks[0].device

    acc = torch.zeros((total_len, D), device=dev)
    cnt = torch.zeros((total_len, 1), device=dev)

    for seg, s in zip(chunks, starts):
        if seg is None or seg.numel() == 0:
            continue

        L = seg.shape[0]
        e = min(total_len, s + L)
        L_eff = e - s

        if L_eff <= 0:
            continue

        acc[s:e] += seg[:L_eff]
        cnt[s:e] += 1.0

    return acc / torch.clamp(cnt, min=1.0)


def length_scaled_pool(X, alpha):
    n = X.shape[0]
    return X.mean(axis=0) * (n ** alpha)


def dtw(X, Y, lam=0.0):
    n, m = len(X), len(Y)

    X_norm = F.normalize(X, p=2, dim=1)
    Y_norm = F.normalize(Y, p=2, dim=1)
    cost_matrix = 1 - X_norm @ Y_norm.T

    D = torch.full((n + 1, m + 1), float("inf"), device=X.device)
    D[0, 0] = 0.0
    lam = torch.tensor(lam, device=X.device)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i - 1, j - 1]
            D[i, j] = cost + min(
                D[i - 1, j - 1],
                D[i - 1, j] + lam,
                D[i, j - 1] + lam,
            )

    return D[n, m] / 1000


def split_into_equal_portions(frames, k: int):
    """Split [T, D] into k contiguous chunks of nearly equal length."""
    if frames is None or len(frames) == 0:
        return []

    k = max(1, int(k))

    if k == 1:
        return [frames]

    chunks = torch.tensor_split(frames, k, dim=0)
    return [c for c in chunks if c.numel() > 0]


def _print_len_stats(name, lens):
    if len(lens) == 0:
        print(f"{name}: no segments")
        return

    mn = min(lens)
    mx = max(lens)
    avg = sum(lens) / len(lens)

    print(f"{name}: min={mn} max={mx} avg={avg:.2f} (n={len(lens)})")


def all_subsegments_leq(x: torch.Tensor, max_len: int):
    """
    x: [T, D]
    yields [L, D] for all start i and length L where:
        1 <= L <= max_len
        i + L <= T
    """
    T = x.shape[0]
    max_len = min(max_len, T)

    for i in range(T):
        max_L = min(max_len, T - i)

        for L in range(1, max_L + 1):
            yield x[i : i + L]


def infer_l2arctic_speaker(path: Path):
    """
    Infer L2-ARCTIC speaker ID from a file path.

    Expected examples:
        .../ABA/wav/arctic_a0001.wav
        .../ABA/textgrid/arctic_a0001.TextGrid
        .../ABA/arctic_a0001.wav

    Returns:
        ABA
    """
    path = Path(path)
    parts = path.parts

    for i, part in enumerate(parts):
        if part.lower() == "wav" and i > 0:
            return parts[i - 1]

    return path.parent.name


def infer_librispeech_speaker(path: Path):
    """
    Infer LibriSpeech speaker ID from a path like:
        .../LibriSpeech/test-clean/61/70968/61-70968-0000.flac

    Returns:
        61

    This is only used for printing/debugging. The LibriSpeech alignment lookup is by wav stem.
    """
    path = Path(path)
    stem = path.stem

    if "-" in stem:
        return stem.split("-")[0]

    return path.parent.parent.name


def get_seg_label(phoneme_info):
    """
    Extract a phone/segment label from an alignment item.

    Supports common keys:
        text, label, phone, phoneme

    Returns None if no label is found.
    """
    for key in ("text", "label", "phone", "phoneme"):
        if key in phoneme_info:
            lab = phoneme_info[key]
            if lab is None:
                return None
            lab = str(lab).strip()
            return lab if lab != "" else None
    return None


def get_l2arctic_alignment(alignments, speaker: str, wav_stem: str, tier: str = "phones"):
    """
    Get nested L2-ARCTIC alignment from:
        alignments[speaker][wav_stem][tier]

    Example:
        alignments["ABA"]["arctic_a0001"]["phones"]
    """
    if speaker not in alignments:
        raise KeyError(
            f"Speaker '{speaker}' not found in L2-ARCTIC alignments. "
            f"Available speakers include: {list(alignments.keys())[:20]}"
        )

    if wav_stem not in alignments[speaker]:
        raise KeyError(
            f"Wav stem '{wav_stem}' not found for L2-ARCTIC speaker '{speaker}'. "
            f"Available stems include: {list(alignments[speaker].keys())[:20]}"
        )

    if tier not in alignments[speaker][wav_stem]:
        raise KeyError(
            f"Tier '{tier}' not found for speaker='{speaker}', wav_stem='{wav_stem}'. "
            f"Available tiers: {list(alignments[speaker][wav_stem].keys())}"
        )

    return alignments[speaker][wav_stem][tier]


def get_librispeech_alignment(alignments, wav_stem: str):
    """
    Get flat LibriSpeech alignment from:
        alignments[wav_stem]

    Example:
        alignments["61-70968-0000"]
    """
    if wav_stem not in alignments:
        raise KeyError(
            f"Wav stem '{wav_stem}' not found in LibriSpeech alignments. "
            f"Available stems include: {list(alignments.keys())[:20]}"
        )

    return alignments[wav_stem]


def main(args):
    wavlm = torch.hub.load(
        "bshall/knn-vc",
        "wavlm_large",
        trust_repo=True,
        device=device,
    )

    hifigan, _ = torch.hub.load(
        "bshall/knn-vc",
        "hifigan_wavlm",
        trust_repo=True,
        device=device,
        prematched=True,
    )

    alpha = args.alpha
    lam = args.lam
    k = args.segment_splits
    min_len = int(args.min_segment_length or 1)

    seg_src_dir = Path("segments/src")
    seg_tgt_dir = Path("segments/tgt")
    dump_idx = 0

    # -------------------------------------------------------------------------
    # Source: LibriSpeech
    # Target: L2-ARCTIC
    # -------------------------------------------------------------------------

    src_wav_path = Path(
        "/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/61/70968/61-70968-0000.flac"
    )

    l2arctic_root = Path(
        "/home/hltcoe/xli/ARTS/corpora/l2arctic"
    )

    tgt_speaker_path = l2arctic_root / "BWC" / "wav"

    l2arctic_alignments_path = Path(
        "/home/hltcoe/xli/ARTS/linearvc/egs/l2arctic/alignments.json"
    )

    librispeech_alignments_path = Path(
        "/home/hltcoe/xli/ARTS/linearvc/exp/asr/LibriSpeech/alignments/alignments.json"
    )

    tgt_speaker_wavs = (
        list(tgt_speaker_path.rglob("*.wav"))
        + list(tgt_speaker_path.rglob("*.flac"))
        + list(tgt_speaker_path.rglob("*.mp3"))
    )
    tgt_speaker_wavs = sorted(tgt_speaker_wavs)

    if not src_wav_path.exists():
        raise FileNotFoundError(f"Source wav does not exist: {src_wav_path}")

    if not tgt_speaker_path.exists():
        raise FileNotFoundError(f"Target speaker wav dir does not exist: {tgt_speaker_path}")

    if len(tgt_speaker_wavs) == 0:
        raise ValueError(f"No target audio files found under: {tgt_speaker_path}")

    if not l2arctic_alignments_path.exists():
        raise FileNotFoundError(f"L2-ARCTIC alignments JSON does not exist: {l2arctic_alignments_path}")

    if not librispeech_alignments_path.exists():
        raise FileNotFoundError(f"LibriSpeech alignments JSON does not exist: {librispeech_alignments_path}")

    src_speaker_name = infer_librispeech_speaker(src_wav_path)
    tgt_speaker_name = infer_l2arctic_speaker(tgt_speaker_path / "dummy.wav")

    print(f"Source wav: {src_wav_path}")
    print(f"Source corpus: LibriSpeech")
    print(f"Source speaker: {src_speaker_name}")
    print(f"Target corpus: L2-ARCTIC")
    print(f"Target speaker path: {tgt_speaker_path}")
    print(f"Target speaker: {tgt_speaker_name}")
    print(f"L2-ARCTIC alignments: {l2arctic_alignments_path}")
    print(f"LibriSpeech alignments: {librispeech_alignments_path}")

    with l2arctic_alignments_path.open("r", encoding="utf-8") as file:
        l2arctic_alignments = json.load(file)

    with librispeech_alignments_path.open("r", encoding="utf-8") as file:
        librispeech_alignments = json.load(file)

    bag = []
    bag_labels = []

    silence_lens = []
    voiced_lens = []

    # -------------------------------------------------------------------------
    # Build target bag from L2-ARCTIC target speaker
    # -------------------------------------------------------------------------
    for tgt_speaker_wav in tqdm(tgt_speaker_wavs, desc="building target bag"):
        wav, sr = torchaudio.load(tgt_speaker_wav)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000

        with torch.no_grad():
            x, _ = wavlm.extract_features(wav.to(device), output_layer=6)

        x = x.squeeze(0)

        if args.target_use_alignment:
            current_frame = 0

            tgt_speaker = infer_l2arctic_speaker(tgt_speaker_wav)
            alignment = get_l2arctic_alignment(
                l2arctic_alignments,
                speaker=tgt_speaker,
                wav_stem=tgt_speaker_wav.stem,
                tier="phones",
            )

            for phoneme_info in alignment:
                start = math.floor(phoneme_info["start"] * 50)
                end = math.ceil(phoneme_info["end"] * 50)

                if start > current_frame:
                    silence_frames = x[current_frame:start]

                    # Preserve original behavior:
                    # internal silence is added when using alignment.
                    for chunk in split_into_equal_portions(silence_frames, k):
                        silence_lens.append(int(chunk.shape[0]))
                        bag.append(chunk)
                        bag_labels.append("<sil>")

                if start >= end:
                    continue

                frames = x[start:end]
                current_frame = end
                target_label = get_seg_label(phoneme_info)

                for chunk in split_into_equal_portions(frames, k):
                    voiced_lens.append(int(chunk.shape[0]))
                    bag.append(chunk)
                    bag_labels.append(target_label)

            if args.bag_silence and current_frame < x.shape[0]:
                for chunk in split_into_equal_portions(x[current_frame:], k):
                    silence_lens.append(int(chunk.shape[0]))
                    bag.append(chunk)
                    bag_labels.append("<sil>")

        if args.max_segment_length and args.max_segment_length > 0:
            for seg in all_subsegments_leq(x, args.max_segment_length):
                if seg.shape[0] >= min_len:
                    bag.append(seg)
                    bag_labels.append(None)

    if len(bag) == 0:
        raise ValueError(
            "Built empty target bag. Use --target_use_alignment and/or --max-segment-length > 0."
        )

    if len(bag) != len(bag_labels):
        raise ValueError(
            f"Internal error: len(bag)={len(bag)} but len(bag_labels)={len(bag_labels)}"
        )

    bag_means = None
    bag_first = None
    bag_last = None

    if args.unit_distance_type in ("mean", "mean_first_last"):
        with torch.no_grad():
            bag_means = torch.stack(
                [length_scaled_pool(seg, alpha) for seg in bag],
                dim=0,
            ).to(device)

    if args.unit_distance_type == "mean_first_last":
        with torch.no_grad():
            bag_first = torch.stack([seg[0] for seg in bag], dim=0).to(device)
            bag_last = torch.stack([seg[-1] for seg in bag], dim=0).to(device)

    def make_label_match_mask(query_label):
        """
        Returns a float tensor [B] where 1.0 means target label matches query label.
        """
        if query_label is None:
            return None

        query_label = str(query_label).strip()

        if query_label == "":
            return None

        mask = [
            1.0 if (lab is not None and str(lab).strip() == query_label) else 0.0
            for lab in bag_labels
        ]

        return torch.tensor(mask, device=device, dtype=torch.float32)

    def pick_from_bag(query_frames, query_label=None, use_first=True, use_last=True):
        """Return best-matching segment from bag according to unit_distance_type."""
        if len(query_frames) == 0:
            return query_frames

        if args.unit_distance_type == "dtw":
            min_dist = float("inf")
            target_seq = None
            target_label = None

            for item, lab in zip(bag, bag_labels):
                dist = dtw(query_frames, item, lam)

                if (
                    query_label is not None
                    and lab is not None
                    and str(lab).strip() == str(query_label).strip()
                    and args.label_bonus != 0.0
                ):
                    dist = dist - args.label_bonus

                if dist < min_dist:
                    min_dist = dist
                    target_seq = item
                    target_label = lab

            print(f"segment label: input={query_label} output={target_label}")

            return target_seq

        elif args.unit_distance_type == "mean":
            q = length_scaled_pool(query_frames, alpha).to(device)
            dists = torch.sum((bag_means - q.unsqueeze(0)) ** 2, dim=1)

            label_mask = make_label_match_mask(query_label)
            if label_mask is not None and args.label_bonus != 0.0:
                dists = dists - args.label_bonus * label_mask

            idx = torch.argmin(dists).item()
            target_label = bag_labels[idx]

            print(f"segment label: input={query_label} output={target_label}")

            return bag[idx]

        elif args.unit_distance_type == "mean_first_last":
            w_mean = args.w_mean
            w_first = args.w_first
            w_last = args.w_last

            q_mean = length_scaled_pool(query_frames, alpha).to(device)
            dist = w_mean * torch.sum((bag_means - q_mean.unsqueeze(0)) ** 2, dim=1)

            if use_first and w_first != 0.0:
                q_first = query_frames[0].to(device)
                dist = dist + w_first * torch.sum(
                    (bag_first - q_first.unsqueeze(0)) ** 2,
                    dim=1,
                )

            if use_last and w_last != 0.0:
                q_last = query_frames[-1].to(device)
                dist = dist + w_last * torch.sum(
                    (bag_last - q_last.unsqueeze(0)) ** 2,
                    dim=1,
                )

            label_mask = make_label_match_mask(query_label)
            if label_mask is not None and args.label_bonus != 0.0:
                dist = dist - args.label_bonus * label_mask

            idx = torch.argmin(dist).item()
            target_label = bag_labels[idx]

            print(f"segment label: input={query_label} output={target_label}")

            return bag[idx]

        else:
            raise ValueError(f"Unknown unit_distance_type: {args.unit_distance_type}")

    # -------------------------------------------------------------------------
    # Source features from LibriSpeech
    # -------------------------------------------------------------------------
    wav, sr = torchaudio.load(src_wav_path)

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    with torch.no_grad():
        x, _ = wavlm.extract_features(wav.to(device), output_layer=6)

    x = x.squeeze(0)

    src_len_to_tgt_len_counts = defaultdict(Counter)

    seg_len = int(args.source_segment_length or 0)
    stride = int(args.source_stride or 0)

    if seg_len <= 0:
        # ---------------------------------------------------------------------
        # Source uses LibriSpeech flat alignments, not L2-ARCTIC nested alignments.
        # ---------------------------------------------------------------------
        src_alignment = get_librispeech_alignment(
            librispeech_alignments,
            wav_stem=src_wav_path.stem,
        )

        new_seq_list = []
        current_frame = 0
        num_segments = len(src_alignment)

        for seg_idx, phoneme_info in enumerate(tqdm(src_alignment, desc="source phones")):
            is_first_seg = seg_idx == 0
            is_last_seg = seg_idx == num_segments - 1
            source_label = get_seg_label(phoneme_info)

            start = math.floor(phoneme_info["start"] * 50)
            end = math.ceil(phoneme_info["end"] * 50)

            if start > current_frame:
                silence_frames = x[current_frame:start]

                if args.bag_silence:
                    for chunk in split_into_equal_portions(silence_frames, k):
                        matched = pick_from_bag(
                            chunk,
                            query_label="<sil>",
                            use_first=not is_first_seg,
                            use_last=not is_last_seg,
                        )
                        new_seq_list.append(matched)

                        with torch.no_grad():
                            src_w = (
                                hifigan(chunk.unsqueeze(0).to(device))
                                .squeeze(0)
                                .detach()
                                .cpu()
                                .unsqueeze(0)
                            )
                            tgt_w = (
                                hifigan(matched.unsqueeze(0).to(device))
                                .squeeze(0)
                                .detach()
                                .cpu()
                                .unsqueeze(0)
                            )

                        save_seg(seg_src_dir, dump_idx, src_w, 16000)
                        save_seg(seg_tgt_dir, dump_idx, tgt_w, 16000)
                        dump_idx += 1
                else:
                    new_seq_list.append(silence_frames)

            if start >= end:
                continue

            frames = x[start:end]

            for chunk in split_into_equal_portions(frames, k):
                matched = pick_from_bag(
                    chunk,
                    query_label=source_label,
                    use_first=not is_first_seg,
                    use_last=not is_last_seg,
                )

                src_len = int(chunk.shape[0])
                tgt_len = int(matched.shape[0]) if matched is not None else 0
                src_len_to_tgt_len_counts[src_len][tgt_len] += 1

                new_seq_list.append(matched)

                with torch.no_grad():
                    src_w = (
                        hifigan(chunk.unsqueeze(0).to(device))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .unsqueeze(0)
                    )
                    tgt_w = (
                        hifigan(matched.unsqueeze(0).to(device))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .unsqueeze(0)
                    )

                save_seg(seg_src_dir, dump_idx, src_w.squeeze(0), 16000)
                save_seg(seg_tgt_dir, dump_idx, tgt_w.squeeze(0), 16000)
                dump_idx += 1

            current_frame = end

        if current_frame < x.shape[0]:
            tail = x[current_frame:]

            if args.bag_silence:
                for chunk in split_into_equal_portions(tail, k):
                    matched = pick_from_bag(
                        chunk,
                        query_label="<sil>",
                        use_first=True,
                        use_last=False,
                    )
                    new_seq_list.append(matched)

                    with torch.no_grad():
                        src_w = (
                            hifigan(chunk.unsqueeze(0).to(device))
                            .squeeze(0)
                            .detach()
                            .cpu()
                            .unsqueeze(0)
                        )
                        tgt_w = (
                            hifigan(matched.unsqueeze(0).to(device))
                            .squeeze(0)
                            .detach()
                            .cpu()
                            .unsqueeze(0)
                        )

                    save_seg(seg_src_dir, dump_idx, src_w, 16000)
                    save_seg(seg_tgt_dir, dump_idx, tgt_w, 16000)
                    dump_idx += 1
            else:
                new_seq_list.append(tail)

        new_seq = torch.cat(new_seq_list, dim=0)

    else:
        if stride <= 0:
            raise ValueError("Meow: set --source-stride > 0 when --source-segment-length > 0.")

        T = x.shape[0]

        if T == 0:
            raise ValueError("Meow: source features have length 0 frames.")

        last = max(0, T - seg_len)
        starts = list(range(0, last + 1, stride))

        if len(starts) == 0:
            starts = [0]
        elif starts[-1] != last:
            starts.append(last)

        matched_chunks = []

        for i, s in enumerate(tqdm(starts, desc="source windows")):
            e = min(T, s + seg_len)
            query = x[s:e]

            is_first = i == 0
            is_last = i == len(starts) - 1

            matched = pick_from_bag(
                query,
                query_label=None,
                use_first=not is_first,
                use_last=not is_last,
            )

            src_len = int(query.shape[0])
            tgt_len = int(matched.shape[0]) if matched is not None else 0
            src_len_to_tgt_len_counts[src_len][tgt_len] += 1

            matched_chunks.append(matched)

            with torch.no_grad():
                src_w = (
                    hifigan(query.unsqueeze(0).to(device))
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .unsqueeze(0)
                )
                tgt_w = (
                    hifigan(matched.unsqueeze(0).to(device))
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .unsqueeze(0)
                )

            save_seg(seg_src_dir, dump_idx, src_w, 16000)
            save_seg(seg_tgt_dir, dump_idx, tgt_w, 16000)
            dump_idx += 1

        if len(matched_chunks) == 0:
            raise ValueError("Meow: no matched chunks produced; check seg_len/stride.")

        overlap_src = max(0, seg_len - stride)
        r = 0.0 if seg_len <= 0 else overlap_src / float(seg_len)

        matched_starts = [0]
        prev_start = 0
        prev_len = (
            int(matched_chunks[0].shape[0])
            if matched_chunks[0] is not None and matched_chunks[0].numel() > 0
            else 0
        )

        for i in range(1, len(matched_chunks)):
            cur = matched_chunks[i]
            cur_len = int(cur.shape[0]) if cur is not None and cur.numel() > 0 else 0

            ov = int(round(r * min(prev_len, cur_len)))
            ov = max(0, min(ov, prev_len, cur_len))

            prev_end = prev_start + prev_len
            next_start = prev_end - ov

            matched_starts.append(next_start)
            prev_start = next_start
            prev_len = cur_len

        total_len = prev_start + prev_len

        new_seq = overlap_add_average(
            matched_chunks,
            matched_starts,
            total_len=total_len,
            device=device,
        )

    with torch.no_grad():
        wav_hat = hifigan(new_seq.unsqueeze(0).to(device)).squeeze(0).detach().cpu()

    torchaudio.save("output.wav", wav_hat, 16000)

    _print_len_stats("Voiced target segments (frames)", voiced_lens)
    _print_len_stats("Silence target segments (frames)", silence_lens)

    out_path = Path("length_bins.csv")

    src_lens = sorted(src_len_to_tgt_len_counts.keys())
    all_tgt_lens = sorted(
        {t for c in src_len_to_tgt_len_counts.values() for t in c.keys()}
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("split=" + str(k))

        for t in all_tgt_lens:
            f.write("\t" + str(t))

        f.write("\n")

        for s in src_lens:
            f.write(str(s))
            row = src_len_to_tgt_len_counts[s]

            for t in all_tgt_lens:
                val = row.get(t, 0)
                f.write("\t" + ("" if val == 0 else str(val)))

            f.write("\n")


if __name__ == "__main__":
    args = check_argv()

    print("Parsed command-line args:")
    for k, v in vars(args).items():
        print(f"  --{k} = {v}")

    main(args)