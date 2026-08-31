import asyncio
import csv
import datetime
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from PIL import Image, ImageFile

# be tolerant to slightly malformed streams
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Optional, forgiving fallback reader
try:
    import imageio.v3 as iio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

# Optional ultra-low-memory decoder/resizer (pyvips)
try:
    import pyvips  # type: ignore
    _HAS_VIPS = True
except Exception:
    _HAS_VIPS = False

# Optional nice progress bar
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


class Imager:
    def __init__(
        self,
        input_path: str,
        base_output_dir: str = ".",
        max_image_size: int = 1280,
        jpeg_quality: int = 85,
        delete_originals: bool = False,
        progress_refresh_interval: float = 0.5,
        per_image_jobs_concurrency: int = 1,
        convert_to_excel: bool = True,
        prompt_cost_per_1k: Optional[float] = None,
        completion_cost_per_1k: Optional[float] = None,
        write_experiment_metadata: bool = False,
        experiment_name: Optional[str] = None,
        experiment_notes: Optional[str] = None,
        fail_fast: bool = True,
        resume: bool = False,
    ):
        self.input_path = input_path
        self.base_output_dir = base_output_dir
        self.output_path = ""
        self.max_image_size = max(0, int(max_image_size)) if max_image_size is not None else 0
        self.jpeg_quality = int(jpeg_quality)
        self.delete_originals = bool(delete_originals)
        self.progress_refresh_interval = float(progress_refresh_interval)
        self.per_image_jobs_concurrency = max(1, int(per_image_jobs_concurrency))
        self.convert_to_excel = bool(convert_to_excel)
        self.prompt_cost_per_1k = prompt_cost_per_1k
        self.completion_cost_per_1k = completion_cost_per_1k
        self.write_experiment_metadata = bool(write_experiment_metadata)
        self.experiment_name = experiment_name
        self.experiment_notes = experiment_notes

        self.fail_fast = bool(fail_fast)
        self.resume = bool(resume)

        self.base_columns = ["PictureId", "Folder", "TimeCreated", "TimeDigitized", "TimeModified"]
        self.logger = logging.getLogger("app")

        self._tokens = {
            "jobs": {},
            "overall": {"prompt": 0, "completion": 0, "total": 0},
        }

        self._t_start = None
        self._t_end = None

    def use_output_dir(self, path: str):
        """
        Use an explicit output directory (for RESUME_DIR).
        Creates it if missing. Does not delete existing contents.
        """
        run_dir = os.path.abspath(path)
        os.makedirs(run_dir, exist_ok=True)
        self.output_path = run_dir
        self.logger.info(f"Using explicit output dir: {self.output_path}")

    # ---------- basic IO ----------

    # Imager.py (add inside class Imager)
    def _save_debug_image(self, folder: str, picture_id: str, jpeg_bytes: bytes):
        try:
            out_dir = os.path.join(self.output_path, "_debug_compressed", folder)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{Path(picture_id).stem}.jpg")
            with open(out_path, "wb") as f:
                f.write(jpeg_bytes)
            self.logger.info(f"Saved debug compressed JPEG -> {out_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save debug image for {picture_id}: {e}")

    def _load_done_keys(self, csv_path: str) -> set[str]:
        """
        Resume keys must be unique across folders.
        We use Folder§PictureId as the unique key.
        """
        if not os.path.exists(csv_path):
            return set()
        try:
            df = pd.read_csv(csv_path, usecols=["Folder", "PictureId"])
            df = df.dropna(subset=["Folder", "PictureId"])
            return set((df["Folder"].astype(str) + "§" + df["PictureId"].astype(str)).tolist())
        except Exception:
            return set()

    def getInputFolder(self):
        return self.input_path

    def createFolder(self, title=datetime.datetime.now().strftime("%Y%m%d%H%M%S")):
        run_dir = os.path.join(self.base_output_dir, title)
        try:
            os.makedirs(run_dir, exist_ok=False)
            self.output_path = run_dir
        except OSError as e:
            print(f"Creation of the directory {run_dir} failed: {e}")
            self.logger.exception(f"Failed to create output dir {run_dir}")
            fallback = os.path.join(".", title)
            os.makedirs(fallback, exist_ok=True)
            self.output_path = fallback

    def removeFile(self, filename):
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception as e:
                print(f"Failed to remove file {filename}: {e}")
                self.logger.warning(f"Failed to remove file {filename}: {e}")

    def getOutputFolder(self):
        return self.output_path

    def get_all_picture_paths(self) -> List[str]:
        picture_list: List[str] = []
        for subfolder in sorted(Path(self.input_path).iterdir()):
            if subfolder.is_dir():
                image_files = chain(
                    subfolder.glob("*.[jJ][pP][gG]"),
                    subfolder.glob("*.[pP][nN][gG]")
                )
                for image_path in sorted(image_files):
                    picture_list.append(f'{str(image_path)}§{subfolder.name}')
        return picture_list

    # ---------- robust image helpers ----------

    def _read_file_bytes_stable(self, path: Path, attempts: int = 4, sleep_s: float = 0.25) -> bytes:
        """
        Read file bytes with a quick 'stable size' check. Helps on network filesystems.
        """
        last_size = -1
        data = b""
        for _ in range(attempts):
            data = path.read_bytes()
            size = len(data)
            if size == last_size or size == 0:
                break
            last_size = size
            time.sleep(sleep_s)
        return data

    def _open_pil_image(self, data: bytes) -> Image.Image:
        im = Image.open(BytesIO(data))
        im.load()
        return im

    def _open_image_anyhow(self, path: Path) -> Image.Image:
        """
        Best-effort opener: PIL -> imageio (if available).
        """
        data = self._read_file_bytes_stable(path)
        last_err = None

        try:
            return self._open_pil_image(data)
        except Exception as e:
            last_err = e

        if _HAS_IMAGEIO:
            try:
                arr = iio.imread(data, extension=path.suffix.lower() or None)
                return Image.fromarray(arr).convert("RGB")
            except Exception as e:
                last_err = e

        raise last_err

    def _downscale_to_jpeg_pil(self, image_path: Path) -> bytes:
        img = self._open_image_anyhow(image_path)
        try:
            img = img.convert("RGB")
            if self.max_image_size > 0:
                w, h = img.size
                longest = max(w, h)
                if longest > self.max_image_size:
                    scale = self.max_image_size / float(longest)
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    img = img.resize((new_w, new_h), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=self.jpeg_quality, optimize=True)
            return buf.getvalue()
        finally:
            img.close()

    def _downscale_to_jpeg_vips(self, image_path: Path) -> bytes:
        img = pyvips.Image.new_from_file(str(image_path), access="sequential")
        if self.max_image_size > 0:
            longest = max(img.width, img.height)
            if longest > self.max_image_size:
                scale = self.max_image_size / float(longest)
                img = img.resize(scale)  # streaming resize
        buf = img.jpegsave_buffer(Q=self.jpeg_quality)
        return bytes(buf)

    def _downscale_to_jpeg(self, image_path: Path) -> bytes:
        if _HAS_VIPS:
            try:
                return self._downscale_to_jpeg_vips(image_path)
            except Exception:
                return self._downscale_to_jpeg_pil(image_path)
        return self._downscale_to_jpeg_pil(image_path)

    def readImageTime(self, image) -> Dict[str, Any]:
        exifdata = image.getexif()
        data = {'TO': '-', 'TD': '-', 'TM': '-'}
        for tag_id in exifdata:
            if tag_id == 36867:
                data['TO'] = exifdata[tag_id]
            elif tag_id == 36868:
                data['TD'] = exifdata[tag_id]
            elif tag_id == 306:
                data['TM'] = exifdata[tag_id]
        return data

    # ---------- progress ----------

    async def _progress_reporter(self, total: int, queue: "asyncio.Queue[str]"):
        processed = 0
        failures = 0
        start = time.monotonic()
        bar = tqdm(total=total, unit="task", desc="Progress", smoothing=0.1) if _HAS_TQDM else None

        try:
            while processed < total:
                try:
                    token = await asyncio.wait_for(queue.get(), timeout=self.progress_refresh_interval)
                except asyncio.TimeoutError:
                    token = None

                if token is not None:
                    processed += 1
                    if token == "failure":
                        failures += 1

                elapsed = max(1e-6, time.monotonic() - start)
                rate = processed / elapsed
                remaining = max(0, total - processed)
                eta_sec = remaining / rate if rate > 0 else 0

                if bar:
                    bar.n = processed
                    bar.set_postfix_str(f"fail={failures}")
                    bar.refresh()
                else:
                    width = 30
                    filled = int((processed / total) * width)
                    bar_txt = "[" + "#" * filled + "-" * (width - filled) + "]"
                    eta_txt = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
                    print(f"\r{bar_txt} {processed}/{total} | ETA ~ {eta_txt} | failures={failures}", end="", flush=True)

            if not bar:
                print()
        finally:
            if bar:
                bar.close()

    # ---------- streaming CSV writers ----------

    def _ensure_csv_header(self, csv_path: str, columns: List[str]):
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.base_columns + columns)

    def _append_csv_row(self, csv_path: str, columns: List[str], row_dict: Dict[str, Any]):
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([row_dict.get(c, "") for c in (self.base_columns + columns)])

    # ---------- tokens tracking ----------

    def _accumulate_tokens(self, job_name: str, usage: Dict[str, Optional[int]]):
        if not usage:
            return
        job_rec = self._tokens["jobs"].setdefault(job_name, {"prompt": 0, "completion": 0, "total": 0})
        p = usage.get("prompt_tokens") or 0
        c = usage.get("completion_tokens") or 0
        t = usage.get("total_tokens") or (p + c)
        job_rec["prompt"] += p
        job_rec["completion"] += c
        job_rec["total"] += t
        self._tokens["overall"]["prompt"] += p
        self._tokens["overall"]["completion"] += c
        self._tokens["overall"]["total"] += t

    def _save_token_report(self):
        # enrich with costs if provided
        tokens_out = dict(self._tokens)  # shallow copy
        if self.prompt_cost_per_1k is not None or self.completion_cost_per_1k is not None:
            def cost_for(p, c):
                cost = 0.0
                if self.prompt_cost_per_1k is not None:
                    cost += (p / 1000.0) * self.prompt_cost_per_1k
                if self.completion_cost_per_1k is not None:
                    cost += (c / 1000.0) * self.completion_cost_per_1k
                return cost

            costs = {"jobs": {}, "overall": 0.0}
            for j, rec in tokens_out["jobs"].items():
                costs["jobs"][j] = cost_for(rec["prompt"], rec["completion"])
            costs["overall"] = cost_for(tokens_out["overall"]["prompt"], tokens_out["overall"]["completion"])
            tokens_out["cost_usd"] = costs

        # JSON
        json_path = os.path.join(self.output_path, "tokens.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(tokens_out, f, indent=2)
            self.logger.info(f"Wrote token report -> {json_path}")
        except Exception as e:
            self.logger.exception(f"Failed to write tokens.json: {e}")

        # CSV (per-job rows + overall)
        csv_path = os.path.join(self.output_path, "tokens.csv")
        try:
            rows = []
            for j, rec in self._tokens["jobs"].items():
                rows.append({"Scope": f"job:{j}", "PromptTokens": rec["prompt"], "CompletionTokens": rec["completion"], "TotalTokens": rec["total"]})
            o = self._tokens["overall"]
            rows.append({"Scope": "overall", "PromptTokens": o["prompt"], "CompletionTokens": o["completion"], "TotalTokens": o["total"]})
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            self.logger.info(f"Wrote token CSV -> {csv_path}")
        except Exception as e:
            self.logger.exception(f"Failed to write tokens.csv: {e}")

    # ---------- main runner (worker pool; memory-safe) ----------
    async def run_all(self, middle, jobs, picture_paths: List[str], max_concurrency=12):
        """
        Process ALL pictures with a bounded worker pool and streaming CSV output.
        Fail-fast: first definitively failed job (after retries) aborts the run.
        Resume: skip (job, PictureId) already present in CSVs.
        """
        num_pictures = len(picture_paths)
        total_tasks = num_pictures * len(jobs)
        print(f"Preparing {len(jobs)} jobs for each of {num_pictures} pictures.")
        print(f"Total API tasks: {total_tasks} (max {max_concurrency} workers; per-image jobs concurrency={self.per_image_jobs_concurrency})")
        if self.resume:
            print("RESUME=True: will skip already-completed rows found in existing CSVs.")
        if self.fail_fast:
            print("FAIL_FAST=True: will abort on first job that fails after retries.")

        self._t_start = time.time()

        # Prepare CSV headers and load resume state
        self._done_by_job: Dict[str, set[str]] = {}
        for job in jobs:
            csv_path = os.path.join(self.output_path, job["csv_file"])
            self._ensure_csv_header(csv_path, job["columns"])
            self._done_by_job[job["name"]] = self._load_done_keys(csv_path) if self.resume else set()

        self._failures: List[Dict[str, str]] = []

        progress_queue: asyncio.Queue[str] = asyncio.Queue()
        reporter = asyncio.create_task(self._progress_reporter(total_tasks, progress_queue))

        work_q: asyncio.Queue[Optional[str]] = asyncio.Queue()
        for p in picture_paths:
            await work_q.put(p)
        for _ in range(max_concurrency):
            await work_q.put(None)

        workers = [
            asyncio.create_task(self._worker(middle, jobs, work_q, progress_queue))
            for _ in range(max_concurrency)
        ]

        try:
            # If any worker raises (fail-fast), gather will raise immediately.
            await asyncio.gather(*workers)
        finally:
            # Reporter will stop once it has seen total_tasks signals.
            # If we fail-fast early, the reporter might otherwise wait forever.
            # So we should drain remaining "expected" progress signals by cancelling it.
            if not reporter.done():
                reporter.cancel()
                try:
                    await reporter
                except Exception:
                    pass

        print(f"\nRun complete. Streaming results written to CSVs. Converting to Excel = {self.convert_to_excel}")

        if self.convert_to_excel:
            self._convert_csvs_to_excel(jobs)

        self._save_failures_csv()
        self._save_token_report()

        self._t_end = time.time()
        if self.write_experiment_metadata:
            self._save_experiment_metadata(jobs, num_pictures, total_tasks)
    
    async def _worker(self, middle, jobs, work_q: "asyncio.Queue[Optional[str]]", progress_queue: "asyncio.Queue[str]"):
        while True:
            item = await work_q.get()
            if item is None:
                break
            image_path_full = item
            await self._process_one_picture(middle, image_path_full, jobs, progress_queue)

    async def _process_one_picture(self, middle, image_path_full: str, jobs, progress_queue: "asyncio.Queue[str]"):
        image_path_str = ""
        folder = "UNKNOWN"
        try:
            image_path_str, folder = image_path_full.split('§')
            image_path = Path(image_path_str)
            picture_id = image_path.name
            done_key = f"{folder}§{picture_id}"

            # Robust open once to read EXIF timestamps
            read_image = await asyncio.to_thread(self._open_image_anyhow, image_path)
            try:
                image_time = await asyncio.to_thread(self.readImageTime, read_image)
            finally:
                read_image.close()

            # Downscale + JPEG compress once in memory
            compressed_bytes = await asyncio.to_thread(self._downscale_to_jpeg, image_path)

            base_data = {
                "PictureId": picture_id,
                "Folder": folder,
                "TimeCreated": image_time['TO'],
                "TimeDigitized": image_time['TD'],
                "TimeModified": image_time['TM']
            }

            sem = asyncio.Semaphore(self.per_image_jobs_concurrency)

            async def run_one_job(job):
                job_name = job["name"]

                # RESUME: if already done, skip API call but count progress
                if self.resume and done_key in self._done_by_job.get(job_name, set()):
                    await progress_queue.put("success")
                    return

                async with sem:
                    result = await self._run_job(middle, job, compressed_bytes, base_data, progress_queue)

                    # success: stream row immediately
                    if result is not None:
                        for j in jobs:
                            if j["name"] == result["job_name"]:
                                csv_path = os.path.join(self.output_path, j["csv_file"])
                                self._append_csv_row(csv_path, j["columns"], result["data"])
                                self._done_by_job[job_name].add(done_key)
                                break

            # If any job raises (fail-fast), this gather raises and aborts the worker
            await asyncio.gather(*[run_one_job(job) for job in jobs])

            del compressed_bytes

            if self.delete_originals:
                try:
                    await asyncio.to_thread(self.removeFile, image_path_str)
                except Exception as e:
                    self.logger.warning(f"Could not remove file {image_path_str}: {e}")

        except Exception as e:
            self.logger.exception(f"Parent task failed for {image_path_str}: {e}")
            print(f"--- PARENT TASK FAILED: {image_path_str} ---")

            for job in jobs:
                await progress_queue.put("failure")
                self._record_failure(
                    picture_id=image_path_str or "UNKNOWN",
                    folder=folder,
                    job_name=job.get("name", "UNKNOWN"),
                    stage="decode",
                    error=str(e),
                )

            if self.fail_fast:
                raise

    async def _run_job(self, middle, job, image_bytes: bytes, base_data: Dict[str, Any], progress_queue: "asyncio.Queue[str]"):
        job_name = job.get('name', 'UNKNOWN')
        picture_id = base_data.get('PictureId', 'UNKNOWN')

        max_attempts = 6
        attempt = 0
        base_delay = 3.0

        while True:
            try:
                parsed, usage = await middle.sendImageRequestAsync(image_bytes, job["prompt"])
                if isinstance(parsed, str) and parsed.startswith("Error processing image:"):
                    raise RuntimeError(parsed)
                # track tokens if present
                if usage:
                    self._accumulate_tokens(job_name, usage)
                break  # success
            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    msg = f"Job '{job_name}' for {picture_id} failed after {attempt} attempts: {e}"
                    print(f"--- {msg}")
                    self.logger.error(msg)
                    await progress_queue.put("failure")
                    self._record_failure(
                        picture_id=picture_id,
                        folder=base_data.get("Folder", "UNKNOWN"),
                        job_name=job_name,
                        stage="request",
                        error=str(e),
                    )
                    return None
                sleep_s = min(60.0, base_delay * (2 ** (attempt - 1)))
                sleep_s *= (0.85 + (0.3 * (hash((job_name, picture_id, attempt)) % 1000) / 1000.0))
                self.logger.warning(f"Retry {attempt}/{max_attempts-1} for {job_name}/{picture_id} in {sleep_s:.1f}s: {e}")
                await asyncio.sleep(sleep_s)

        # --- Structured parse to row ---
        try:
            data_for_excel = base_data.copy()

            # job["columns"] == [cat, f"{cat}Confidence", "ImageDescription"]
            cat_col, conf_col, desc_col = job["columns"]

            # `parsed` can be a Pydantic model or a dict (fallback path)
            if hasattr(parsed, "presence"):
                presence = int(parsed.presence)
                confidence = int(parsed.confidence)
                description = str(parsed.description)
            else:
                presence = int(parsed["presence"])
                confidence = int(parsed["confidence"])
                description = str(parsed["description"])

            data_for_excel[cat_col] = presence
            data_for_excel[conf_col] = confidence
            data_for_excel[desc_col] = description

            await progress_queue.put("success")
            return {"job_name": job_name, "data": data_for_excel}

        except Exception as e_parse:
            self.logger.error(
                f"Parsing failed for job '{job_name}' / {picture_id}: {e_parse}. Raw structured: {parsed}"
            )
            await progress_queue.put("failure")
            self._record_failure(
                picture_id=picture_id,
                folder=base_data.get("Folder", "UNKNOWN"),
                job_name=job_name,
                stage="parse",
                error=str(e_parse),
            )
            return None

    # ---------- CSV→Excel (optional) ----------

    def _convert_csvs_to_excel(self, jobs):
        for job in jobs:
            csv_path = os.path.join(self.output_path, job["csv_file"])
            xlsx_path = os.path.join(self.output_path, job["xlsx_file"])
            if not os.path.exists(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
                df.to_excel(xlsx_path, index=False)
                self.logger.info(f"Converted {csv_path} -> {xlsx_path} ({len(df)} rows)")
            except Exception as e:
                self.logger.exception(f"Failed to convert {csv_path} to Excel: {e}")

    # ---------- failures CSV ----------

    def _record_failure(self, picture_id: str, folder: str, job_name: str, stage: str, error: str):
        if not hasattr(self, "_failures"):
            self._failures = []
        self._failures.append({
            "PictureId": picture_id,
            "Folder": folder,
            "Job": job_name,
            "Stage": stage,          # 'decode' | 'request' | 'parse'
            "Error": error,
        })

    def _save_failures_csv(self):
        if not getattr(self, "_failures", None):
            print("No failures to write. Great!")
            return
        out = os.path.join(self.output_path, "failures.csv")
        try:
            pd.DataFrame(self._failures, columns=["PictureId", "Folder", "Job", "Stage", "Error"]).to_csv(out, index=False)
            print(f"Wrote failures CSV -> {out}  (rows: {len(self._failures)})")
        except Exception as e:
            self.logger.exception(f"Failed to write failures.csv: {e}")

    # ---------- experiment metadata (optional) ----------

    def _safe_git_commit(self) -> Optional[str]:
        try:
            rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=1.0)
            return rev.decode("utf-8").strip()
        except Exception:
            return None

    def _list_existing_outputs(self, jobs) -> dict:
        out = {"csv": [], "xlsx": []}
        for j in jobs:
            csv_path = os.path.join(self.output_path, j["csv_file"])
            xlsx_path = os.path.join(self.output_path, j["xlsx_file"])
            if os.path.exists(csv_path):
                out["csv"].append(os.path.basename(csv_path))
            if os.path.exists(xlsx_path):
                out["xlsx"].append(os.path.basename(xlsx_path))
        for extra in ("failures.csv", "tokens.csv", "tokens.json", "run.log"):
            p = os.path.join(self.output_path, extra)
            if os.path.exists(p):
                out.setdefault("other", []).append(extra)
        return out

    def _save_experiment_metadata(self, jobs, num_images: int, total_tasks: int):
        try:
            duration = round((self._t_end - self._t_start), 3) if (self._t_start and self._t_end) else None
            failures = getattr(self, "_failures", []) or []
            tokens = getattr(self, "_tokens", {"overall": {"prompt": 0, "completion": 0, "total": 0}, "jobs": {}})

            # Optional cost calc
            def _cost(p, c):
                cost = 0.0
                if self.prompt_cost_per_1k is not None:
                    cost += (p / 1000.0) * self.prompt_cost_per_1k
                if self.completion_cost_per_1k is not None:
                    cost += (c / 1000.0) * self.completion_cost_per_1k
                return round(cost, 6)

            overall_cost = _cost(tokens["overall"]["prompt"], tokens["overall"]["completion"]) \
                if (self.prompt_cost_per_1k is not None or self.completion_cost_per_1k is not None) else None

            meta = {
                "experiment": {
                    "name": self.experiment_name,
                    "notes": self.experiment_notes,
                },
                "run": {
                    "output_dir": self.output_path,
                    "start_iso": datetime.datetime.fromtimestamp(self._t_start).isoformat() if self._t_start else None,
                    "end_iso": datetime.datetime.fromtimestamp(self._t_end).isoformat() if self._t_end else None,
                    "duration_seconds": duration,
                },
                "data": {
                    "num_images": num_images,
                    "total_tasks": total_tasks,
                    "failures": len(failures),
                },
                "tokens": {
                    "overall": tokens.get("overall", {}),
                    "by_job": tokens.get("jobs", {}),
                    "overall_cost_usd": overall_cost,
                    "unit_costs_per_1k": {
                        "prompt": self.prompt_cost_per_1k,
                        "completion": self.completion_cost_per_1k,
                    },
                },
                "env": {
                    "FOLDERPATH": os.getenv("FOLDERPATH"),
                    "OUTPUT_DIR": os.getenv("OUTPUT_DIR"),
                    "BASE_URL": os.getenv("BASE_URL"),
                    "MODEL": os.getenv("MODEL"),
                    "MAX_CONCURRENCY": os.getenv("MAX_CONCURRENCY"),
                    "PER_IMAGE_JOBS_CONCURRENCY": os.getenv("PER_IMAGE_JOBS_CONCURRENCY"),
                    "MAX_IMAGE_SIZE": os.getenv("MAX_IMAGE_SIZE"),
                    "JPEG_QUALITY": os.getenv("JPEG_QUALITY"),
                    "CONVERT_TO_EXCEL": os.getenv("CONVERT_TO_EXCEL"),
                    "DELETE_ORIGINALS": os.getenv("DELETE_ORIGINALS"),
                    "PROGRESS_REFRESH_INTERVAL": os.getenv("PROGRESS_REFRESH_INTERVAL"),
                    # New optional generation controls:
                    "MAX_COMPLETION_TOKENS": os.getenv("MAX_COMPLETION_TOKENS"),
                    "TEMPERATURE": os.getenv("TEMPERATURE"),
                    "SEED": os.getenv("SEED"),
                },
                "host": {
                    "hostname": socket.gethostname(),
                    "platform": platform.platform(),
                    "python": sys.version.split()[0],
                    "has_imageio": _HAS_IMAGEIO,
                    "has_pyvips": _HAS_VIPS,
                    "has_tqdm": _HAS_TQDM,
                    "git_commit": self._safe_git_commit(),
                },
                "outputs": self._list_existing_outputs(jobs),
                "jobs": [
                    {
                        "name": j["name"],
                        "columns": j["columns"],
                        "csv_file": j["csv_file"],
                        "xlsx_file": j["xlsx_file"],
                    } for j in jobs
                ],
            }

            out_json = os.path.join(self.output_path, "experiment.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            self.logger.info(f"Wrote experiment metadata -> {out_json}")

            # also a tiny 1-row CSV for quick grepping
            out_csv = os.path.join(self.output_path, "experiment_summary.csv")
            pd.DataFrame([{
                "ExperimentName": self.experiment_name or "",
                "DurationSec": duration,
                "Images": num_images,
                "Tasks": total_tasks,
                "Failures": len(failures),
                "PromptTokens": meta["tokens"]["overall"].get("prompt", 0),
                "CompletionTokens": meta["tokens"]["overall"].get("completion", 0),
                "TotalTokens": meta["tokens"]["overall"].get("total", 0),
                "CostUSD": overall_cost if overall_cost is not None else "",
                "Model": os.getenv("MODEL"),
                "BaseURL": os.getenv("BASE_URL"),
                "MaxConc": os.getenv("MAX_CONCURRENCY"),
                "PerImageConc": os.getenv("PER_IMAGE_JOBS_CONCURRENCY"),
                "MaxImageSize": os.getenv("MAX_IMAGE_SIZE"),
                "JPEGQuality": os.getenv("JPEG_QUALITY"),
            }]).to_csv(out_csv, index=False)
            self.logger.info(f"Wrote experiment summary CSV -> {out_csv}")

        except Exception as e:
            self.logger.exception(f"Failed to write experiment metadata: {e}")
