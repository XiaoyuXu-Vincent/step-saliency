#!/usr/bin/env python3
"""
LiveCodeBench code-generation evaluation driver for GPT-OSS.
Features:
  * Local dataset loading (release tags v1~v6)
  * Automatic prompt construction for stdin/functional tasks
  * Execution harness for stdin + LeetCode-style function tests
  * Optional interventions (SMI/OEB) sharing switches with GPQA scripts
  * Resume / overwrite friendly JSONL logs + summary metrics
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import math
import pickle
import re
import subprocess
import sys
import textwrap
import time
import traceback
import zlib
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interventions import (  # noqa: E402
    BridgeGuardOEBWrapper,
    StepMomentumInjectionWrapper,
)
from src.model_config import (  # noqa: E402
    available_model_types,
    collect_eos_token_ids,
    resolve_model_config,
)

FUNCTIONAL_RUNNER_SOURCE = """#!/usr/bin/env python3
import ast
import base64
import importlib.util
import json
import pickle
import sys
import traceback


def _parse_scalar(text: str):
    text = text.strip()
    if text == "":
        return ""
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue
    return text


def _parse_args(blob: str):
    lines = [ln for ln in blob.splitlines() if ln.strip() != ""]
    return [_parse_scalar(ln) for ln in lines]


def _encode_payload(obj):
    return base64.b64encode(pickle.dumps(obj)).decode("utf-8")


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"status": "error", "error": "missing argv"}))
        return 1

    solution_path = sys.argv[1]
    method_name = sys.argv[2]
    payload_text = sys.stdin.read()
    try:
        payload = json.loads(payload_text or "{}")
    except json.JSONDecodeError:
        payload = {}
    args_blob = payload.get("input_blob", "")

    try:
        spec = importlib.util.spec_from_file_location("candidate_solution", solution_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        solution_cls = getattr(module, "Solution")
        method = getattr(solution_cls(), method_name)
        args = _parse_args(args_blob)
        result = method(*args)
        print(json.dumps({"status": "ok", "payload": _encode_payload(result)}))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
"""


@dataclass
class TestCase:
    idx: int
    origin: str  # "public" or "private"
    testtype: str
    input: str
    output: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LiveCodeBench evaluation with GPT-OSS + interventions"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="data/models/gpt-oss-20b",
        help="HF model path or local checkpoint",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", *available_model_types()],
        help="Model family (auto, gpt-oss, deepseek-qwen)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/datasets/live_code_bench",
        help="Directory containing LiveCodeBench jsonl files",
    )
    parser.add_argument(
        "--version-tag",
        type=str,
        default="release_v1",
        help="Version tag (release_v1~release_v6, v1, v1_v3, release_latest, ...)",
    )
    parser.add_argument(
        "--test-mode",
        type=str,
        default="all",
        choices=("all", "public", "private"),
        help="Which tests to execute for scoring",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of problems to evaluate",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum generation tokens per sample",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: model-specific)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p for sampling (only if temperature > 0)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Chat template reasoning_effort knob",
    )
    parser.add_argument(
        "--execution-timeout",
        type=float,
        default=10.0,
        help="Seconds per test case before timing out",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/livecodebench_eval",
        help="Where to save details_*.jsonl + summary.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing details file (auto enabled if file exists)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previous outputs before starting",
    )
    parser.add_argument(
        "--use-smi",
        action="store_true",
        help="Enable Step Momentum Injection",
    )
    parser.add_argument(
        "--smi-strength",
        type=float,
        default=0.2,
        help="SMI strength parameter",
    )
    parser.add_argument(
        "--use-oeb",
        action="store_true",
        help="Enable BridgeGuard OEB",
    )
    parser.add_argument(
        "--oeb-max-layer",
        type=int,
        default=None,
        help="Highest layer index (inclusive) for OEB auto-range (default: auto)",
    )
    parser.add_argument(
        "--oeb-layers",
        type=str,
        default="",
        help='Comma separated explicit layers list, e.g. "1,3,5,7"',
    )
    return parser.parse_args()


def load_allowed_files(dataset_dir: Path) -> Dict[str, List[str]]:
    """Import the dataset script to reuse ALLOWED_FILES."""
    module_path = dataset_dir / "code_generation_lite.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing dataset helper: {module_path}")
    import importlib.util

    spec = importlib.util.spec_from_file_location("lcb_codegen_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    allowed = getattr(module, "ALLOWED_FILES", None)
    if not allowed:
        raise RuntimeError("Failed to read ALLOWED_FILES from dataset script")
    return allowed


def decode_private_cases(raw: str | None) -> List[Dict[str, Any]]:
    if not raw or raw.strip().lower() in {"", "null"}:
        return []
    try:
        data = pickle.loads(zlib.decompress(base64.b64decode(raw)))
        if isinstance(data, list):
            return [
                {
                    "input": item.get("input", ""),
                    "output": item.get("output", ""),
                    "testtype": item.get("testtype", "stdin"),
                }
                for item in data
                if isinstance(item, dict)
            ]
    except Exception:
        traceback.print_exc()
    return []


def load_livecodebench(
    dataset_dir: Path, version_tag: str, max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    allowed_map = load_allowed_files(dataset_dir)
    if version_tag not in allowed_map:
        raise ValueError(f"Unknown version_tag '{version_tag}', choose from {sorted(allowed_map)}")
    samples: List[Dict[str, Any]] = []
    files = allowed_map[version_tag]
    running_idx = 0
    for fname in files:
        path = dataset_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset shard: {path}")
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                public_cases = json.loads(record.get("public_test_cases") or "[]")
                private_cases = decode_private_cases(record.get("private_test_cases"))
                record["_public_cases"] = public_cases
                record["_private_cases"] = private_cases
                record["_idx"] = running_idx
                samples.append(record)
                running_idx += 1
                if max_samples and len(samples) >= max_samples:
                    return samples
    return samples


def detect_mode(record: Dict[str, Any]) -> str:
    for coll in (record.get("_public_cases") or [], record.get("_private_cases") or []):
        for case in coll:
            ttype = case.get("testtype")
            if ttype in ("stdin", "functional"):
                return ttype
    # Default fallback
    return "stdin"


def build_user_prompt(record: Dict[str, Any], mode: str) -> str:
    title = record.get("question_title", "Unknown Problem")
    platform = (record.get("platform") or "Unknown platform").title()
    difficulty = record.get("difficulty", "unknown")
    body = record.get("question_content", "").strip()
    header = f"{title} ({platform}, difficulty: {difficulty})"
    instructions = [
        (
            "You will be given a competitive programming problem. "
            "Write a correct and efficient solution in Python 3."
        ),
        "Return exactly one ```python``` code block containing the final program with no extra commentary.",
    ]
    if mode == "stdin":
        instructions.append(
            "The program must read from standard input and write to standard output without prompts or debug prints."
        )
    else:
        starter = record.get("starter_code", "").rstrip()
        instructions.append(
            "This is a LeetCode-style class interface. Implement the required method inside `class Solution` without using input()/print()."
        )
        if starter:
            instructions.append("Starter code:\n```python\n" + starter + "\n```")
    instructions.append("Problem statement:\n" + body)
    prompt = header + "\n\n" + "\n\n".join(instructions)
    return textwrap.dedent(prompt).strip()


CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(response: str) -> Optional[str]:
    matches = CODE_BLOCK_PATTERN.findall(response)
    if matches:
        return matches[-1].strip()
    # fallback: try to detect from triple backticks without language
    if "```" in response:
        parts = response.split("```")
        if len(parts) >= 3:
            return parts[-2].strip()
    return response.strip() if response.strip() else None


def normalize_stdout(text: str) -> str:
    return text.replace("\r", "").rstrip()


def parse_value(serialized: str) -> Any:
    serialized = serialized.strip()
    if serialized == "":
        return ""
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(serialized)
        except Exception:
            continue
    return serialized


FLOAT_REL_TOL = 1e-6
FLOAT_ABS_TOL = 1e-6


def _try_parse_structured(text: str) -> Tuple[bool, Any]:
    text = text.strip()
    if text == "":
        return True, ""
    for parser in (json.loads, ast.literal_eval):
        try:
            return True, parser(text)
        except Exception:
            continue
    return False, text


def _tokens_close(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        return math.isclose(float(a), float(b), rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL)
    except ValueError:
        return False


def values_close(expected: Any, actual: Any, rel_tol: float = FLOAT_REL_TOL, abs_tol: float = FLOAT_ABS_TOL) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=rel_tol, abs_tol=abs_tol)
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False
        return all(values_close(e, a, rel_tol, abs_tol) for e, a in zip(expected, actual))
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False
        return all(values_close(expected[k], actual[k], rel_tol, abs_tol) for k in expected)
    if isinstance(expected, set) and isinstance(actual, set):
        return expected == actual
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip() == actual.strip()
    if type(expected) != type(actual):
        try:
            return values_close(float(expected), float(actual), rel_tol, abs_tol)  # type: ignore[arg-type]
        except Exception:
            return str(expected).strip() == str(actual).strip()
    return expected == actual


def compare_stdout_outputs(expected: str, actual: str) -> bool:
    if expected == actual:
        return True

    exp_lines = [ln.rstrip() for ln in expected.splitlines() if ln.strip() != ""]
    act_lines = [ln.rstrip() for ln in actual.splitlines() if ln.strip() != ""]
    if exp_lines == act_lines:
        return True

    exp_tokens = expected.split()
    act_tokens = actual.split()
    if not exp_tokens and not act_tokens:
        return True
    if exp_tokens and len(exp_tokens) == len(act_tokens):
        if all(_tokens_close(e, a) for e, a in zip(exp_tokens, act_tokens)):
            return True

    exp_ok, exp_val = _try_parse_structured(expected)
    act_ok, act_val = _try_parse_structured(actual)
    if exp_ok and act_ok and values_close(exp_val, act_val):
        return True

    return False


def build_test_suite(record: Dict[str, Any], mode: str, test_mode: str) -> List[TestCase]:
    cases: List[TestCase] = []
    idx = 0
    if test_mode in ("all", "public"):
        for case in record.get("_public_cases", []):
            cases.append(
                TestCase(
                    idx=idx,
                    origin="public",
                    testtype=case.get("testtype", mode),
                    input=case.get("input", ""),
                    output=case.get("output", ""),
                )
            )
            idx += 1
    if test_mode in ("all", "private"):
        for case in record.get("_private_cases", []):
            cases.append(
                TestCase(
                    idx=idx,
                    origin="private",
                    testtype=case.get("testtype", mode),
                    input=case.get("input", ""),
                    output=case.get("output", ""),
                )
            )
            idx += 1
    return cases


def infer_method_name(starter_code: str) -> Optional[str]:
    match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", starter_code)
    if match:
        name = match.group(1)
        if name != "self":
            return name
    return None


def run_stdin_suite(
    code_text: str, cases: List[TestCase], timeout: float, python_exec: str
) -> Tuple[int, List[Dict[str, Any]]]:
    logs: List[Dict[str, Any]] = []
    if not cases:
        return 0, logs

    with TemporaryDirectory() as tmpdir:
        code_path = Path(tmpdir) / "solution.py"
        code_path.write_text(code_text, encoding="utf-8")
        passed = 0
        for case in cases:
            case_log: Dict[str, Any] = {
                "case_idx": case.idx,
                "origin": case.origin,
                "testtype": case.testtype,
            }
            try:
                proc = subprocess.run(
                    [python_exec, str(code_path)],
                    input=case.input,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                stdout = normalize_stdout(proc.stdout)
                expected = normalize_stdout(case.output)
                case_log["stdout"] = stdout
                case_log["expected"] = expected
                if proc.returncode != 0:
                    case_log["passed"] = False
                    case_log["stderr"] = proc.stderr.strip()
                    case_log["error_type"] = "runtime_error"
                elif compare_stdout_outputs(expected, stdout):
                    case_log["passed"] = True
                    passed += 1
                else:
                    case_log["passed"] = False
                    case_log["error_type"] = "wrong_answer"
            except subprocess.TimeoutExpired as exc:
                case_log["passed"] = False
                case_log["error_type"] = "timeout"
                case_log["stdout"] = (exc.stdout or "").strip()
                case_log["stderr"] = (exc.stderr or "").strip()
            logs.append(case_log)
    return passed, logs


def run_functional_suite(
    code_text: str,
    cases: List[TestCase],
    timeout: float,
    python_exec: str,
    method_name: str,
) -> Tuple[int, List[Dict[str, Any]]]:
    logs: List[Dict[str, Any]] = []
    if not cases:
        return 0, logs

    with TemporaryDirectory() as tmpdir:
        code_path = Path(tmpdir) / "solution.py"
        runner_path = Path(tmpdir) / "functional_runner.py"
        code_path.write_text(code_text, encoding="utf-8")
        runner_path.write_text(FUNCTIONAL_RUNNER_SOURCE, encoding="utf-8")

        passed = 0
        for case in cases:
            case_log: Dict[str, Any] = {
                "case_idx": case.idx,
                "origin": case.origin,
                "testtype": case.testtype,
            }
            expected_value = parse_value(case.output)
            payload = json.dumps({"input_blob": case.input})

            try:
                proc = subprocess.run(
                    [python_exec, str(runner_path), str(code_path), method_name],
                    input=payload,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                case_log["passed"] = False
                case_log["error_type"] = "timeout"
                case_log["stdout"] = (exc.stdout or "").strip()
                case_log["stderr"] = (exc.stderr or "").strip()
                logs.append(case_log)
                continue

            stdout = proc.stdout.strip()
            case_log["stdout"] = stdout
            if proc.returncode != 0:
                case_log["passed"] = False
                case_log["stderr"] = proc.stderr.strip()
                case_log["error_type"] = "runtime_error"
                logs.append(case_log)
                continue

            try:
                resp = json.loads(stdout or "{}")
            except json.JSONDecodeError:
                case_log["passed"] = False
                case_log["error_type"] = "bad_runner_output"
                logs.append(case_log)
                continue

            if resp.get("status") != "ok":
                case_log["passed"] = False
                case_log["error_type"] = resp.get("error") or "execution_failed"
                case_log["traceback"] = resp.get("traceback")
                logs.append(case_log)
                continue

            payload_base64 = resp.get("payload", "")
            try:
                actual_value = pickle.loads(base64.b64decode(payload_base64))
            except Exception:
                case_log["passed"] = False
                case_log["error_type"] = "decode_failure"
                logs.append(case_log)
                continue

            if values_close(expected_value, actual_value):
                case_log["passed"] = True
                passed += 1
            else:
                case_log["passed"] = False
                case_log["error_type"] = "wrong_answer"
                case_log["expected"] = expected_value
                case_log["actual"] = actual_value
            logs.append(case_log)
    return passed, logs


def load_completed_results(path: Path) -> Tuple[List[Dict[str, Any]], set[int]]:
    if not path.exists():
        return [], set()
    results: List[Dict[str, Any]] = []
    seen: set[int] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            results.append(record)
            seen.add(record["idx"])
    return results, seen


def main():
    args = parse_args()
    explicit_type = None if args.model_type == "auto" else args.model_type
    model_config = resolve_model_config(
        model_path=args.model_path,
        explicit_type=explicit_type,
    )
    dataset_dir = Path(args.dataset_dir)
    samples = load_livecodebench(dataset_dir, args.version_tag, args.max_samples)

    mode_bits = ["LCB"]
    if args.use_smi:
        mode_bits.append("SMI")
    if args.use_oeb:
        mode_bits.append("OEB")
    mode_name = "+".join(mode_bits)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    smi_wrapper = None
    if args.use_smi:
        smi_wrapper = StepMomentumInjectionWrapper(
            tokenizer,
            strength=args.smi_strength,
            model_config=model_config,
        )
        smi_wrapper.apply(model)

    oeb_wrapper = None
    oeb_layers = None
    if args.oeb_layers.strip():
        try:
            oeb_layers = sorted(
                {
                    int(piece.strip())
                    for piece in args.oeb_layers.split(",")
                    if piece.strip()
                }
            )
        except ValueError:
            raise ValueError(f"Invalid --oeb-layers value: {args.oeb_layers}")

    resolved_oeb_max_layer = args.oeb_max_layer
    if resolved_oeb_max_layer is None:
        resolved_oeb_max_layer = model_config.default_oeb_max_layer

    if args.use_oeb:
        oeb_wrapper = BridgeGuardOEBWrapper(
            tokenizer,
            max_layer=resolved_oeb_max_layer,
            layers=oeb_layers,
            model_config=model_config,
        )
        oeb_wrapper.apply(model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix_bits = [
        "lcb",
        args.version_tag.replace("/", "_"),
        args.test_mode,
    ]
    if args.use_smi:
        suffix_bits.append(f"smi{str(args.smi_strength).replace('.', 'p')}")
    if args.use_oeb:
        if oeb_layers:
            suffix_bits.append("oebl" + "".join(map(str, oeb_layers)))
        else:
            suffix_bits.append(f"oebl{resolved_oeb_max_layer}")
    suffix_bits.append(f"re{args.reasoning_effort}")
    details_path = output_dir / f"details_{'_'.join(suffix_bits)}.jsonl"
    summary_path = output_dir / f"summary_{'_'.join(suffix_bits)}.json"

    if details_path.exists():
        if args.overwrite:
            details_path.unlink()
            if summary_path.exists():
                summary_path.unlink()
            existing_results: List[Dict[str, Any]] = []
            completed = set()
        else:
            print(f"Resume detected: {details_path}")
            args.resume = True
            existing_results, completed = load_completed_results(details_path)
    else:
        existing_results, completed = [], set()

    python_exec = sys.executable
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_token_ids = collect_eos_token_ids(tokenizer, model_config)
    temperature = (
        args.temperature
        if args.temperature is not None
        else model_config.default_temperature
    )
    sampling = temperature > 0
    total_cases = total_pass = 0
    public_total = public_pass = 0
    private_total = private_pass = 0

    # Seed stats with resumed results
    for record in existing_results:
        total_cases += record.get("num_tests", 0)
        total_pass += record.get("tests_passed", 0)
        public_total += record.get("public_cases_total", 0)
        public_pass += record.get("public_cases_passed", 0)
        private_total += record.get("private_cases_total", 0)
        private_pass += record.get("private_cases_passed", 0)

    start_from = len(existing_results)
    progress = tqdm(total=len(samples), desc="LiveCodeBench", initial=len(completed))

    for sample in samples:
        idx = sample["_idx"]
        if idx in completed:
            continue

        mode = detect_mode(sample)
        test_suite = build_test_suite(sample, mode, args.test_mode)
        prompt = build_user_prompt(sample, mode)
        messages = []
        if model_config.allow_system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": "You are an elite competitive programming assistant. Always return ONLY python code inside a single code block.",
                }
            )
        messages.append({"role": "user", "content": prompt})
        template_kwargs = {
            "conversation": messages,
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if model_config.supports_reasoning_effort:
            template_kwargs["reasoning_effort"] = args.reasoning_effort
        chat_prompt = tokenizer.apply_chat_template(**template_kwargs)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": sampling,
        }
        if sampling:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = args.top_p
        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id
        if eos_token_ids:
            gen_kwargs["eos_token_id"] = (
                eos_token_ids if len(eos_token_ids) > 1 else eos_token_ids[0]
            )

        try:
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            response = tokenizer.decode(
                outputs[0][prompt_len:], skip_special_tokens=False
            )
        except Exception as exc:
            traceback.print_exc()
            progress.update(1)
            result = {
                "idx": idx,
                "question_id": sample.get("question_id"),
                "title": sample.get("question_title"),
                "platform": sample.get("platform"),
                "solved": False,
                "error": f"generation_failed: {exc}",
            }
            with details_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            continue

        code_text = extract_python_code(response)
        if not code_text:
            result = {
                "idx": idx,
                "question_id": sample.get("question_id"),
                "title": sample.get("question_title"),
                "platform": sample.get("platform"),
                "solved": False,
                "error": "no_code_block_found",
                "raw_response": response,
            }
            with details_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            progress.update(1)
            continue

        eval_start = time.time()
        if mode == "stdin":
            passed, logs = run_stdin_suite(code_text, test_suite, args.execution_timeout, python_exec)
        else:
            starter = sample.get("starter_code", "")
            method_name = infer_method_name(starter) or infer_method_name(code_text)
            if not method_name:
                logs = [
                    {
                        "case_idx": 0,
                        "origin": "n/a",
                        "testtype": "functional",
                        "passed": False,
                        "error_type": "method_name_not_found",
                    }
                ]
                passed = 0
            else:
                passed, logs = run_functional_suite(
                    code_text, test_suite, args.execution_timeout, python_exec, method_name
                )
        eval_end = time.time()

        total = len(test_suite)
        solved = total > 0 and passed == total
        public_cases_total = sum(1 for c in test_suite if c.origin == "public")
        private_cases_total = sum(1 for c in test_suite if c.origin == "private")
        public_cases_passed = sum(1 for log in logs if log.get("origin") == "public" and log.get("passed"))
        private_cases_passed = sum(1 for log in logs if log.get("origin") == "private" and log.get("passed"))

        total_cases += total
        total_pass += passed
        public_total += public_cases_total
        public_pass += public_cases_passed
        private_total += private_cases_total
        private_pass += private_cases_passed

        result = {
            "idx": idx,
            "question_id": sample.get("question_id"),
            "title": sample.get("question_title"),
            "platform": sample.get("platform"),
            "difficulty": sample.get("difficulty"),
            "mode": mode,
            "num_tests": total,
            "tests_passed": passed,
            "solved": solved,
            "test_logs": logs,
            "response": response,
            "code": code_text,
            "elapsed_sec": eval_end - eval_start,
            "public_cases_total": public_cases_total,
            "public_cases_passed": public_cases_passed,
            "private_cases_total": private_cases_total,
            "private_cases_passed": private_cases_passed,
        }

        if smi_wrapper:
            result["smi_stats"] = smi_wrapper.collect_stats()
        if oeb_wrapper:
            result["oeb_stats"] = oeb_wrapper.collect_stats()

        with details_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")

        progress.update(1)

    progress.close()

    total_problems = len(existing_results) + (len(samples) - len(completed))
    solved_count = sum(1 for rec in load_completed_results(details_path)[0] if rec.get("solved"))
    summary = {
        "mode": mode_name,
        "version_tag": args.version_tag,
        "test_mode": args.test_mode,
        "total_problems": total_problems,
        "solved": solved_count,
        "pass_at_1": solved_count / total_problems if total_problems else 0.0,
        "total_cases": total_cases,
        "case_pass_rate": total_pass / total_cases if total_cases else 0.0,
        "public_cases": {
            "total": public_total,
            "passed": public_pass,
            "pass_rate": public_pass / public_total if public_total else 0.0,
        },
        "private_cases": {
            "total": private_total,
            "passed": private_pass,
            "pass_rate": private_pass / private_total if private_total else 0.0,
        },
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        "execution_timeout": args.execution_timeout,
    }

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if smi_wrapper:
        smi_wrapper.remove()
    if oeb_wrapper:
        oeb_wrapper.remove()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

