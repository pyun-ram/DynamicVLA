#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Sequentially run simulations/evaluate.py for every tests/*.json.
#
# How to run (from anywhere; paths are resolved from this repo):
#
#   # Recommended: Isaac Lab env + auto-start inference.py per test case
#   conda activate isaaclab
#   python3 scripts/batch_evaluate_tests.py \
#       --with-inference -p /path/to/vla-checkpoint -r euler -d -s
#
#   # Evaluate only (you must run scripts/inference.py yourself in another shell)
#   conda activate isaaclab
#   python3 scripts/batch_evaluate_tests.py
#
#   # Dry-run: print commands without executing
#   python3 scripts/batch_evaluate_tests.py --dry-run
#
# Failure policy (default): continue after a failed case; exit code is the number
# of failures (0 if all succeeded). Use --stop-on-failure to abort on first error.
#
# Note: evaluate.py is a long-running ZMQ server and does not exit after one
# suite when used alone. This script terminates each evaluate.py process after
# inference finishes (--with-inference) or after --timeout-per-run seconds.

from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = REPO_ROOT / "simulations"
EVALUATE_PY = SIM_DIR / "evaluate.py"
INFERENCE_PY = REPO_ROOT / "scripts" / "inference.py"
TESTS_DIR = REPO_ROOT / "tests"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "evaluation"


@dataclass
class RunResult:
    name: str
    index: int
    total: int
    ok: bool
    evaluate_rc: int | None = None
    inference_rc: int | None = None
    message: str = ""


@dataclass
class BatchSummary:
    results: list[RunResult] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.results if not r.ok)


def discover_test_jsons(tests_dir: Path) -> list[Path]:
    if not tests_dir.is_dir():
        raise FileNotFoundError("Tests directory not found: %s" % tests_dir)
    files = sorted(tests_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError("No tests/*.json files under %s" % tests_dir)
    return files


def wait_for_tcp(host: str, port: int, timeout_s: float, poll_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            time.sleep(poll_s)
    return False


def build_evaluate_cmd(
    env_cfg: Path,
    scene_dir: Path,
    object_dir: Path,
    output_dir: Path,
    n_tests: int,
    num_envs: int,
    host: str,
    img_port: int,
    act_port: int,
    extra: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(EVALUATE_PY),
        "--scene_dir",
        str(scene_dir),
        "--object_dir",
        str(object_dir),
        "--env_cfg",
        str(env_cfg),
        "--enable_cameras",
        "--num_envs",
        str(num_envs),
        "-n",
        str(n_tests),
        "--output_dir",
        str(output_dir),
        "--headless",
        "--save",
        "--host",
        host,
        "--img_port",
        str(img_port),
        "--act_port",
        str(act_port),
    ]
    cmd.extend(extra)
    return cmd


def build_inference_cmd(
    weights: Path,
    host: str,
    img_port: int,
    act_port: int,
    rotation: str,
    delta: bool,
    streaming: bool,
    alias: str | None,
    epoch: int,
    output_dir: Path | None,
    extra: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(INFERENCE_PY),
        "--host",
        host,
        "--img_port",
        str(img_port),
        "--act_port",
        str(act_port),
        "-p",
        str(weights),
        "-r",
        rotation,
        "-i",
        str(epoch),
    ]
    if delta:
        cmd.append("-d")
    if streaming:
        cmd.append("-s")
    if alias:
        cmd.extend(["-a", alias])
    if output_dir is not None:
        cmd.extend(["-o", str(output_dir)])
    cmd.extend(extra)
    return cmd


def terminate_process(proc: subprocess.Popen | None, grace_s: float = 15.0) -> int | None:
    if proc is None:
        return None
    if proc.poll() is not None:
        return proc.returncode
    proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    return proc.returncode


def run_one(
    env_cfg: Path,
    index: int,
    total: int,
    args: argparse.Namespace,
    log_path: Path | None,
) -> RunResult:
    name = env_cfg.name
    logging.info("[%d/%d] %s", index, total, name)

    eval_cmd = build_evaluate_cmd(
        env_cfg=env_cfg,
        scene_dir=args.scene_dir,
        object_dir=args.object_dir,
        output_dir=args.output_dir,
        n_tests=args.n_tests,
        num_envs=args.num_envs,
        host=args.host,
        img_port=args.img_port,
        act_port=args.act_port,
        extra=args.evaluate_extra,
    )

    if args.dry_run:
        logging.info("DRY-RUN evaluate: %s", " ".join(eval_cmd))
        if args.with_inference:
            inf_cmd = build_inference_cmd(
                weights=args.weights,
                host=args.host,
                img_port=args.img_port,
                act_port=args.act_port,
                rotation=args.rotation,
                delta=args.delta,
                streaming=args.streaming,
                alias=args.alias,
                epoch=args.epoch,
                output_dir=args.inference_output_dir,
                extra=args.inference_extra,
            )
            logging.info("DRY-RUN inference: %s", " ".join(inf_cmd))
        return RunResult(name=name, index=index, total=total, ok=True, message="dry-run")

    stdout_target = None
    log_fp = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(log_path, "w", encoding="utf-8")
        stdout_target = log_fp

    evaluate_proc: subprocess.Popen | None = None
    inference_proc: subprocess.Popen | None = None
    ok = False
    message = ""

    try:
        evaluate_proc = subprocess.Popen(
            eval_cmd,
            cwd=str(SIM_DIR),
            stdout=stdout_target,
            stderr=subprocess.STDOUT,
        )

        if not wait_for_tcp(args.host, args.img_port, args.server_startup_timeout):
            message = "evaluate.py ZMQ port not ready in time"
            return RunResult(
                name=name,
                index=index,
                total=total,
                ok=False,
                evaluate_rc=terminate_process(evaluate_proc),
                message=message,
            )

        if args.with_inference:
            inf_cmd = build_inference_cmd(
                weights=args.weights,
                host=args.host,
                img_port=args.img_port,
                act_port=args.act_port,
                rotation=args.rotation,
                delta=args.delta,
                streaming=args.streaming,
                alias=args.alias,
                epoch=args.epoch,
                output_dir=args.inference_output_dir,
                extra=args.inference_extra,
            )
            # Inference uses the PyTorch env; run from repo root.
            inference_proc = subprocess.Popen(
                inf_cmd,
                cwd=str(REPO_ROOT),
                stdout=stdout_target,
                stderr=subprocess.STDOUT,
            )
            try:
                inference_rc = inference_proc.wait(timeout=args.timeout_per_run)
            except subprocess.TimeoutExpired:
                message = "inference.py timed out"
                terminate_process(inference_proc, grace_s=5.0)
                return RunResult(
                    name=name,
                    index=index,
                    total=total,
                    ok=False,
                    inference_rc=-1,
                    evaluate_rc=terminate_process(evaluate_proc),
                    message=message,
                )

            evaluate_rc = terminate_process(evaluate_proc)
            ok = inference_rc == 0
            if not ok:
                message = "inference.py exited with code %d" % inference_rc
            return RunResult(
                name=name,
                index=index,
                total=total,
                ok=ok,
                evaluate_rc=evaluate_rc,
                inference_rc=inference_rc,
                message=message,
            )

        # Evaluate-only: wait until timeout, then stop Isaac.
        try:
            evaluate_rc = evaluate_proc.wait(timeout=args.timeout_per_run)
            ok = evaluate_rc == 0
            if not ok:
                message = "evaluate.py exited with code %d" % evaluate_rc
        except subprocess.TimeoutExpired:
            evaluate_rc = terminate_process(evaluate_proc)
            ok = False
            message = (
                "timed out after %ds; use --with-inference or run inference.py "
                "in another terminal before the timeout"
                % args.timeout_per_run
            )
        return RunResult(
            name=name,
            index=index,
            total=total,
            ok=ok,
            evaluate_rc=evaluate_rc,
            message=message,
        )
    finally:
        terminate_process(evaluate_proc)
        terminate_process(inference_proc)
        if log_fp is not None:
            log_fp.close()


def print_summary(summary: BatchSummary) -> None:
    logging.info("=" * 60)
    logging.info(
        "Done: %d/%d succeeded, %d failed.",
        summary.n_ok,
        len(summary.results),
        summary.n_fail,
    )
    if summary.n_fail:
        logging.info("Failures:")
        for r in summary.results:
            if not r.ok:
                extra = []
                if r.inference_rc is not None:
                    extra.append("inference=%s" % r.inference_rc)
                if r.evaluate_rc is not None:
                    extra.append("evaluate=%s" % r.evaluate_rc)
                suffix = (" (%s)" % ", ".join(extra)) if extra else ""
                detail = (" — %s" % r.message) if r.message else ""
                logging.info("  - %s%s%s", r.name, suffix, detail)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluate.py sequentially for all tests/*.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=TESTS_DIR,
        help="Directory containing per-env *.json configs",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=REPO_ROOT / "scenes",
        help="DOM 3D scenes directory",
    )
    parser.add_argument(
        "--object-dir",
        type=Path,
        default=REPO_ROOT / "objects",
        help="DOM 3D objects directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Evaluation video output directory",
    )
    parser.add_argument(
        "-n",
        "--n-tests",
        type=int,
        default=1,
        help="Trials per environment (evaluate.py --n_tests)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Parallel Isaac envs (evaluate.py --num_envs)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--img-port", type=int, default=3186)
    parser.add_argument("--act-port", type=int, default=3188)
    parser.add_argument(
        "--server-startup-timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for evaluate.py ZMQ ports",
    )
    parser.add_argument(
        "--timeout-per-run",
        type=float,
        default=7200.0,
        help="Max seconds per test JSON (evaluate and/or inference)",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop the batch on the first failed case (default: continue)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="If set, write per-case logs under this directory",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--evaluate-extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed to evaluate.py (prefix with --)",
    )
    parser.add_argument(
        "--with-inference",
        action="store_true",
        help="Start scripts/inference.py for each case (PyTorch env)",
    )
    parser.add_argument(
        "-p",
        "--weights",
        type=Path,
        default=None,
        help="VLA checkpoint path (required with --with-inference)",
    )
    parser.add_argument("-r", "--rotation", default="euler")
    parser.add_argument("-d", "--delta", action="store_true")
    parser.add_argument("-s", "--streaming", action="store_true")
    parser.add_argument("-a", "--alias", default=None)
    parser.add_argument("-i", "--epoch", type=int, default=0)
    parser.add_argument(
        "--inference-output-dir",
        type=Path,
        default=None,
        help="Optional -o for inference.py",
    )
    parser.add_argument(
        "--inference-extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed to inference.py (prefix with --)",
    )
    args = parser.parse_args(argv)

    if args.with_inference and args.weights is None:
        parser.error("--with-inference requires -p/--weights")
    if not EVALUATE_PY.is_file():
        parser.error("evaluate.py not found: %s" % EVALUATE_PY)
    if args.with_inference and not INFERENCE_PY.is_file():
        parser.error("inference.py not found: %s" % INFERENCE_PY)

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )

    if not args.with_inference and not args.dry_run:
        logging.warning(
            "Running evaluate.py only. You need scripts/inference.py connected "
            "via ZMQ (ports %d/%d) in another terminal, or use --with-inference.",
            args.img_port,
            args.act_port,
        )
        logging.warning(
            "evaluate.py does not exit after a suite; each run is stopped after "
            "--timeout-per-run (%ds) unless --with-inference is set.",
            int(args.timeout_per_run),
        )

    test_files = discover_test_jsons(args.tests_dir)
    total = len(test_files)
    logging.info("Found %d test configs under %s", total, args.tests_dir)

    summary = BatchSummary()
    for index, env_cfg in enumerate(test_files, start=1):
        log_path = None
        if args.log_dir is not None:
            log_path = args.log_dir / ("%s.log" % env_cfg.stem)

        result = run_one(env_cfg, index, total, args, log_path)
        summary.results.append(result)

        status = "OK" if result.ok else "FAIL"
        logging.info(
            "[%d/%d] %s -> %s%s",
            index,
            total,
            env_cfg.name,
            status,
            (" (%s)" % result.message) if result.message else "",
        )

        if not result.ok and args.stop_on_failure:
            logging.error("Stopping batch due to --stop-on-failure.")
            break

    print_summary(summary)
    return summary.n_fail


if __name__ == "__main__":
    sys.exit(main())
