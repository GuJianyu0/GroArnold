#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transform unified_settings.yaml (user-friendly) into the legacy settings files:
  - user_settings_multi.txt
  - IC_setting_list.txt
and copy:
  - unified_settings.yaml
  - run.param
  - IC_DICE_manucraft.params
Also write per-model IC_DICE_{model}.params into:
  GDDFAA/step1_galaxy_IC_preprocess/step1_set_IC_DDFA/

Usage:
  python3 transform_yaml_settings.py
  python3 transform_yaml_settings.py /path/to/unified_settings.yaml
"""

import sys, re, shutil, glob
from pathlib import Path

# ---- YAML loader (PyYAML required) ----
try:
    import yaml
except Exception as e:
    print("[transform] ERROR: PyYAML is required: pip install pyyaml", file=sys.stderr)
    raise

HERE = Path(__file__).resolve().parent           # install_and_run/
ROOT = HERE.parent                                # .../GroArnold_framework/
IC_SETTING_DIR = ROOT / "GDDFAA/step1_galaxy_IC_preprocess/step1_set_IC_DDFA/"

# ---- small helpers -----------------------------------------------------------
def _safe_get(d, path, default=None):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def _num(s):
    try:
        if isinstance(s, (int, float)):
            return s
        if isinstance(s, str):
            s2 = s.strip()
            if re.fullmatch(r"[-+]?\d+", s2):
                return int(s2)
            if re.fullmatch(r"[-+]?\d*\.\d+(e[-+]?\d+)?", s2, flags=re.I) or re.fullmatch(r"[-+]?\d+\.?(e[-+]?\d+)?", s2, flags=re.I):
                return float(s2)
        return s
    except Exception:
        return s

def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[transform] wrote {path.relative_to(ROOT)}")

def _copy_if_exists_else_exit(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[transform] copied {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    else:
        print(f"[transform] Not found: {src}. Cannot copy, exit.")
        sys.exit(2) #debug

def _autodetect_yaml():
    # Search in initial_conditions/settings_*/unified_settings.yaml
    cands = list((HERE / "initial_conditions").glob("settings_*/unified_settings.yaml"))
    if len(cands) == 1:
        return cands[0]
    return cands[0] if cands else None

#----------------------------------------------------------
# ---- simple public helpers for other modules ----
from pathlib import Path
import sys, yaml
def detect_unified_yaml(settings_dir=None):
    """
    Find unified_settings.yaml. If settings_dir is given, use it; otherwise
    reuse this script's autodetection. Exit(2) on failure.
    """
    if settings_dir:
        y = Path(settings_dir).expanduser().resolve() / "unified_settings.yaml"
        if not y.exists():
            print(f"[settings] ERROR: unified_settings.yaml not found: {y}", file=sys.stderr)
            sys.exit(2)
        return y

    return Path(y)

def get_fit_choice(yaml_path, particle_type):
    """
    Return (fitting_model_name, is_fit_1d) for the requested particle_type.
    Order of precedence: fit.components entry matching `type` -> fit.defaults.
    Exit(2) if keys are missing.
    """
    p = Path(yaml_path)
    if not p.exists():
        print(f"[settings] ERROR: YAML not found: {p}", file=sys.stderr)
        sys.exit(2)

    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    fit = (cfg.get("fit") or {})
    defaults = (fit.get("defaults") or {})
    comps = (fit.get("components") or [])

    # pick per-type override first
    entry = None
    for c in comps:
        if int(c.get("type", -999)) == int(particle_type):
            entry = c
            break
    if entry is None:
        entry = defaults
    print(entry)

    if "fitting_model" not in entry or "is_fit_1d" not in entry:
        print("[settings] ERROR: missing 'fitting_model' or 'is_fit_1d' under fit/defaults or fit/components", file=sys.stderr)
        sys.exit(2)

    return str(entry["fitting_model"]), bool(entry["is_fit_1d"])

# -----------------------------------------------------------------------------
def main():
    # 0) pick YAML
    print("[transform] sys.argv: ", sys.argv)
    yml_path = Path(sys.argv[1]).resolve()
    if not yml_path.exists():
        print(f"[transform] ERROR: {yml_path} not found.", file=sys.stderr)
        sys.exit(2)
    print(f"[transform] using YAML: {yml_path.relative_to(ROOT)}")

    cfg = yaml.safe_load(yml_path.read_text(encoding="utf-8"))

    # Support both placements:
    defaults = cfg.get("defaults", {})
    models_block = cfg.get("models")
    if models_block is None:
        models_block = defaults.get("models", [])

    # 1) Build model names and values for IC_setting_list.txt
    # Schema (your YAML):
    # - each item has modelnumber; first usually has models_name; others may omit -> inherit
    base_name = None
    model_names = []
    flatz_vals = []
    for i, m in enumerate(models_block):
        if "models_name" in m:
            base_name = m["models_name"]
        if base_name is None:
            # fallback: derive from YAML folder name settings_{models_name}
            try:
                base_folder = yml_path.parent.name  # settings_{models_name}
                base_name = base_folder.split("settings_", 1)[1]
            except Exception:
                base_name = "Model"
        num = int(m.get("modelnumber", i))
        model_names.append(f"{base_name}{num}")
        flatz_vals.append(str(_num(m.get("flatz1", 0.5))))  # default 0.5 if absent

    # # user_settings_multi.txt
    # t_start = _num(_safe_get(defaults, "preparation.t_start", 0.0))
    # t_end   = _num(_safe_get(defaults, "preparation.t_end_inclusive", 1.0))
    # dt_act  = _num(_safe_get(defaults, "preparation.time_snapshot_target_interval", 0.1))
    # dt_tp   = _num(_safe_get(defaults, "preparation.time_snapshot_each", 0.1))
    # n_cut   = int(_safe_get(defaults, "preparation.n_cut_snapshot", 10))
    # snap_cmp= int(_safe_get(defaults, "preparation.snapshot_to_compare", 10))
    # n_mpi   = int(_safe_get(defaults, "preparation.n_mpi", 4))
    # mod_ic  = int(_safe_get(defaults, "preparation.is_modify_IC", 0))
    # mpirun  = _safe_get(defaults, "angle_action.mpirun_line",
    #                     "mpirun -np 4 mains/./data.exe 0.02 0.03 0.01 0.01 0. 0 0 5 4 -1 -1 -1 0.03 0.031 1. 1 -1 0 -1 0 -3. -2. 1 1 1. 1. 1. 1.")
    # fit_flag= int(_safe_get(defaults, "preparation.fit_model", 1))

    # usm = []
    # usm.append("#" * 101)
    # usm.append("#### User settings for running prog GDDFAA.")
    # usm.append("#### NOTE: Auto-generated from unified_settings.yaml. Edit the YAML, not this file.")
    # usm.append("#" * 101)
    # usm.append("#### Begin")
    # usm.append("# --- MODELS LINE (space-separated model names) ---")
    # usm.append(" ".join(model_names))
    # usm.append("# --- SNAPSHOTS SETTINGS (8 numbers):")
    # usm.append("# t_start  t_end_inclusive  time_snapshot_target_interval  time_snapshot_each  n_cut_snapshot  snapshot_to_compare  n_mpi  is_modify_IC")
    # usm.append(f"{t_start} {t_end} {dt_act} {dt_tp} {n_cut} {snap_cmp} {n_mpi} {mod_ic}")
    # usm.append("# --- ACTIONS AA mpirun line (kept verbatim) ---")
    # usm.append(mpirun.strip())
    # usm.append("# --- FIT ENABLE FLAG (0/1) ---")
    # usm.append(str(fit_flag))
    # usm.append("#### End")
    # _write_text(HERE / "user_settings_multi.txt", "\n".join(usm) + "\n")

    # IC_setting_list.txt
    ics = []
    ics.append(f"{base_name}")
    ics.append("flatz1")
    ics.append(" ".join(flatz_vals))
    ics.append("1 -1 0")
    ics.append("#line1: model name; line2: various parameter; line3: value of various parameter; line4: fit profile in powerlaw or Sersic; line5: comment.")
    _write_text(HERE / "IC_setting_list.txt", "\n".join(ics) + "\n")

    # 2) Copy run.param and IC_DICE_manucraft.params (and optional galaxy_general.config)
    settings_dir = yml_path.parent
    _copy_if_exists_else_exit(settings_dir / "unified_settings.yaml", HERE / "unified_settings.yaml")
    _copy_if_exists_else_exit(settings_dir / "run.param", HERE / "run.param")
    _copy_if_exists_else_exit(settings_dir / "IC_DICE_manucraft.params", HERE / "IC_DICE_manucraft.params")
    _copy_if_exists_else_exit(settings_dir / "galaxy_general.config", HERE / "galaxy_general.config")

    # 3) Place per-model IC_DICE_{model}.params where DICE expects them
    IC_SETTING_DIR.mkdir(parents=True, exist_ok=True)
    src_dice = settings_dir / "IC_DICE_manucraft.params"
    for name in model_names:
        dst = IC_SETTING_DIR / f"IC_DICE_{name}.params"
        _copy_if_exists_else_exit(src_dice, dst)

    # 4) Copy galaxy_general.config next to those needed by your DICE runner
    cfg_src = settings_dir / "galaxy_general.config"
    if cfg_src.exists():
        _copy_if_exists_else_exit(cfg_src, IC_SETTING_DIR / "galaxy_general.config")

    print("[transform] done.")
    # sys.exit(2) #debug



if __name__ == "__main__":

    main()
