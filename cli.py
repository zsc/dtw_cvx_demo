import argparse
import json
import sys
from typing import Dict

from pipeline import run_alignment


def _parse_step_penalty(args) -> Dict[str, float]:
    return {
        "diag": args.step_penalty_diag,
        "horiz": args.step_penalty_horiz,
        "vert": args.step_penalty_vert,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convex DTW alignment using Whisper features (no silence trimming)."
    )
    parser.add_argument("wav1", type=str)
    parser.add_argument("wav2", type=str)

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

    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    return parser


def main(argv=None):
    parser = build_argparser()
    args = parser.parse_args(argv)

    band_radius = None if args.no_band else args.band_radius
    step_penalty = _parse_step_penalty(args)

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
    )

    payload = result.mapping
    payload["flow_cost"] = result.flow_cost

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    else:
        json.dump(payload, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
