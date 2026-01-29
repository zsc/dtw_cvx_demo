import argparse
import json
import sys

import numpy as np

from audio_warp import load_audio_mono, save_wav, warp_audio_to_match
from pipeline import run_alignment


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Warp wav2 into wav1 timeline using Whisper DTW alignment."
    )
    parser.add_argument("wav1", type=str, help="Reference wav (A)")
    parser.add_argument("wav2", type=str, help="Source wav to warp (B)")
    parser.add_argument("output_wav", type=str, help="Warped output wav path")

    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--feature-mode", type=str, default="whisper_encoder")
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--dist", type=str, default="cosine", choices=["cosine", "l2sq"])
    parser.add_argument("--gamma-time", type=float, default=0.1)
    parser.add_argument("--band-radius", type=float, default=0.08)
    parser.add_argument("--no-band", action="store_true", help="Disable band constraint")

    parser.add_argument("--step-penalty-diag", type=float, default=0.0)
    parser.add_argument("--step-penalty-horiz", type=float, default=0.2)
    parser.add_argument("--step-penalty-vert", type=float, default=0.2)
    parser.add_argument("--cost-scale", type=float, default=1_000_000)

    parser.add_argument("--qp-alpha", type=float, default=1e-2)
    parser.add_argument("--qp-beta", type=float, default=1e-2)
    parser.add_argument("--slope-min", type=float, default=None)
    parser.add_argument("--slope-max", type=float, default=None)

    parser.add_argument("--max-band-tries", type=int, default=4)
    parser.add_argument("--band-expand", type=float, default=1.5)

    parser.add_argument(
        "--trim-silence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove long silence segments before alignment and warping.",
    )
    parser.add_argument("--trim-top-db", type=float, default=40.0)
    parser.add_argument("--trim-min-silence-sec", type=float, default=0.5)
    parser.add_argument("--trim-frame-length", type=int, default=2048)
    parser.add_argument("--trim-hop-length", type=int, default=512)

    parser.add_argument("--save-mapping", type=str, default=None)
    return parser


def main(argv=None):
    parser = build_argparser()
    args = parser.parse_args(argv)

    band_radius = None if args.no_band else args.band_radius
    step_penalty = {
        "diag": args.step_penalty_diag,
        "horiz": args.step_penalty_horiz,
        "vert": args.step_penalty_vert,
    }

    result = run_alignment(
        args.wav1,
        args.wav2,
        feature_mode=args.feature_mode,
        model_name=args.model,
        device=args.device,
        dist=args.dist,
        gamma_time=args.gamma_time,
        band_radius=band_radius,
        step_penalty=step_penalty,
        cost_scale=args.cost_scale,
        qp_alpha=args.qp_alpha,
        qp_beta=args.qp_beta,
        slope_min=args.slope_min,
        slope_max=args.slope_max,
        max_band_tries=args.max_band_tries,
        band_expand=args.band_expand,
        trim_silence=args.trim_silence,
        trim_top_db=args.trim_top_db,
        trim_min_silence_sec=args.trim_min_silence_sec,
        trim_frame_length=args.trim_frame_length,
        trim_hop_length=args.trim_hop_length,
    )

    if args.save_mapping:
        with open(args.save_mapping, "w", encoding="utf-8") as f:
            json.dump(result.mapping, f, ensure_ascii=False)

    audio_b, _, _ = load_audio_mono(
        args.wav2,
        sample_rate=args.sr,
        trim_silence=args.trim_silence,
        trim_top_db=args.trim_top_db,
        trim_min_silence_sec=args.trim_min_silence_sec,
        trim_frame_length=args.trim_frame_length,
        trim_hop_length=args.trim_hop_length,
    )

    u = np.asarray(result.mapping["u"], dtype=np.float32)
    v = np.asarray(result.mapping["v"], dtype=np.float32)
    D1 = float(result.mapping["durations"]["D1"])
    D2 = float(result.mapping["durations"]["D2"])

    warped = warp_audio_to_match(audio_b, args.sr, u, v, D1, D2)
    save_wav(args.output_wav, warped, args.sr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
