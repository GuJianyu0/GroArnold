#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########################################################
# This file is a Python3 controller for GroArnold (GDDFAA) prog 
# -- a framework for compution DF based on angle-actions of 
# triaxial galaxies.
# This is similar to the workflow of ``install_and_run/run.bash''.
# Works best when run from the directory: GroArnold_framework/install_and_run/

# Structure
# There is a loop for several galaxy models in running the whole workflow.
# There are 4 modules for a certain galaxy model (user set initial parameters, although the actual parameters 
# would deviate from the setting after simulation):
# module 1. initial condition and Nbody simulation; 
# module 2. triaxiality alignment; 
# module 3. actions calculation; 
# module 4. DF of actions fitting.

# Notes
# ## path
# This file path must be in "GroArnold_framework/install_and_run/" with aim to "folder_main"; if not, the path would be wrong.
# ## resume
# #To resume at a specific point; enable --resume_point would not backup galaxy_general/ or galaxy_general_XXX/ folders.
# #resume_point 1. initial condition and simulation only (module 1; stops after simulate)
# #resume_point 2. triaxial alignment only (module 2)
# #resume_point 3. actions only (module 3)
# #resume_point 4. DF fit and plots only (module 4)
# #resume_point 5. rename current galaxy folder and continue with the next model(s)
# #resume_point 6. compare only (post-run compare step)

# #run from example resume point 2 in modelnumber 0 till run to end for all galaxy modelnumers in whole prog (recommanded): run point 2 in modelnumber 1, run point 3 in modelnumber 1, ..., run point 5 in modelnumber 1 (rename galaxy folder (galaxy_general/ as the current) into folder about modelnumber 1); make galaxy folder for modelmuber 3, run point 1 in model number 3, ... (suppose modelnumber 3 is the max number); run point 6 for all modelnumbers (compare models)
# python3 workflow_wrapper.py ./initial_conditions/settings_Ein_multicomp_spinL_axisLH/ --resume_point 2 1 --modelnumber 1

# #run from example resume point 2 in modelnumber 0 and then exit immediately (debug mode): run point 2 in modelnumber 0, exit the whole prog without any other running
# python3 workflow_wrapper.py ./initial_conditions/settings_Ein_multicomp_spinL_axisLH/ --resume_point 2 0 --modelnumber 1

# ## check whether running at same path in shell window
# ps -aux|egrep 'dice|mpirun|Gadget2|out.exe|data.exe|fit_galaxy_distribution_function.py|plot_action_figs.py'
# kill -9 "$(cat controller.pid)"  # kill the controller if detached
###########################################################



import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
# import shutil
import shlex
from datetime import datetime



#### utilities
#global switches (runtime-updated in main())
FORCE_RERUN = False

def _group_enabled(first: str, last: str, start_step: str, until_step: str) -> bool:
    """
    True iff the step-range [first,last] intersects the active window [start_step,until_step].
    Keeps module gating robust if new steps are added later.
    """
    return STEP_ORDER[first] <= STEP_ORDER[until_step] and STEP_ORDER[last] >= STEP_ORDER[start_step]

def run_cmd(cmd, cwd: Path, log: Optional[Path] = None, env: Optional[dict] = None, check=True) -> int:
    """
    Run a command (list or str) in directory `cwd`. If log is provided, stream stdout/stderr to it.
    Returns the process return code. Raises on failure if check=True.
    """
    if isinstance(cmd, str):
        shell = True
        display = cmd
    else:
        shell = False
        display = ' '.join(shlex.quote(c) for c in cmd)
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)

    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8') as fh:
            fh.write(f"\n--- CMD @ {time.strftime('%Y-%m-%d %H:%M:%S')} in {cwd} ---\n{display}\n")
            fh.flush()
            proc = subprocess.run(cmd, cwd=str(cwd), env=env, shell=shell,
                                  stdout=fh, stderr=subprocess.STDOUT, check=False)
            rc = proc.returncode
            fh.write(f"\n--- RC={rc} ---\n")
    else:
        print(f"[run] {display} (cwd={cwd})")
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, shell=shell, check=False)
        rc = proc.returncode

    if check and rc != 0:
        print("See log file: ", log)
        raise subprocess.CalledProcessError(rc, cmd)
    return rc

def run_cmd_detach(cmd, cwd: Path, log: Path, pidfile: Optional[Path] = None) -> int:
    """
    Emulate nohup: start a command detached; write logs to `log` and pid to `pidfile` if provided.
    Returns the PID.
    """
    if isinstance(cmd, str):
        popen_args = dict(args=cmd, shell=True)
        display = cmd
    else:
        popen_args = dict(args=cmd, shell=False)
        display = ' '.join(shlex.quote(c) for c in cmd)
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    lf = open(log, 'a', encoding='utf-8')
    lf.write(f"\n--- DETACH @ {time.strftime('%Y-%m-%d %H:%M:%S')} in {cwd} ---\n{display}\n")
    proc = subprocess.Popen(cwd=str(cwd), stdout=lf, stderr=subprocess.STDOUT, **popen_args)
    if pidfile:
        pidfile.write_text(str(proc.pid), encoding='utf-8')
    print(f"[detach] started PID={proc.pid} log={log}")
    return proc.pid

def copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = Path(src).read_bytes()
    Path(dst).write_bytes(data)

def ensure_dir(p: Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def _ensure_unix_line_endings(p: Path):
    """Convert CRLF -> LF if present (for *.bat carried from Windows)."""
    if not p.exists():
        return
    raw = p.read_bytes()
    if b"\r\n" in raw:
        p.write_bytes(raw.replace(b"\r\n", b"\n"))

def _make_executable(p: Path):
    """Add +x to a file if it exists."""
    if not p.exists():
        return
    st = os.stat(p)
    os.chmod(p, st.st_mode | 0o111)

def read_lines_keep(file: Path) -> List[str]:
    if not file.exists():
        return []
    return [ln.rstrip('\n') for ln in file.read_text(encoding='utf-8', errors='ignore').splitlines()]


def non_comment_lines(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        out.append(ln)
    return out

# ---- model info helper (public) ----
from pathlib import Path
import yaml
HERE = Path(__file__).resolve().parent           # install_and_run/
def _autodetect_yaml():
    # Search in initial_conditions/settings_*/unified_settings.yaml
    cands = list((HERE / "initial_conditions").glob("settings_*/unified_settings.yaml"))
    if len(cands) == 1:
        return cands[0]
    # Prefer the first under Ein_* or first alphabetically
    prio = [p for p in cands if "Ein_" in p.as_posix()]
    if prio:
        return prio[0]
    return cands[0] if cands else None

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

    # Reuse transform's own autodetection (already used before printing "using YAML: ...")
    y = _autodetect_yaml()
    if (y is None) or (not Path(y).exists()):
        print("[settings] ERROR: cannot autodetect unified_settings.yaml", file=sys.stderr)
        sys.exit(2)
    return Path(y)

def get_model_info_by_modelname(yaml_path, model_name):
    """
    Given a unified_settings.yaml and a model_name (e.g. 'Ein_...LH0'),
    return a dict with base_name, modelname, modelnumber, and any per-model fields
    such as flatz1 and potential_rotating when available.
    Exits(2) on missing YAML.
    """
    p = Path(yaml_path)
    if not p.exists():
        print(f"[settings] ERROR: YAML not found: {p}", file=sys.stderr)
        sys.exit(2)

    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        print("[settings] ERROR: YAML root must be a mapping.", file=sys.stderr)
        sys.exit(2)

    models = cfg.get("models") or (cfg.get("defaults") or {}).get("models") or []
    # base_name from first model or from folder name settings_<NAME>
    base_name = None
    if models and isinstance(models[0], dict) and "models_name" in models[0]:
        base_name = str(models[0]["models_name"])
    if not base_name:
        try:
            base_name = p.parent.name.split("settings_", 1)[1]
        except Exception:
            base_name = "Model"

    # parse trailing integer from model_name
    import re
    m = re.search(r"(\d+)$", str(model_name))
    num = int(m.group(1)) if m else None

    found = None
    for md in models:
        if not isinstance(md, dict):
            continue
        mn = md.get("modelnumber", None)
        if mn is None:
            continue
        try:
            mn = int(mn)
        except Exception:
            continue
        if f"{base_name}{mn}" == str(model_name) or (num is not None and mn == num):
            found = md
            break

    info = {
        "base_name": base_name,
        "modelname": str(model_name),
        "modelnumber": int(found.get("modelnumber", num if num is not None else -1)) if found else (num if num is not None else -1),
        "flatz1": found.get("flatz1") if isinstance(found, dict) else None,
        "potential_rotating": found.get("potential_rotating") if isinstance(found, dict) else None,
    }
    # Optional nested IC fields (if present in your YAML)
    if isinstance(found, dict) and isinstance(found.get("initial_conditions"), dict):
        ic = found["initial_conditions"]
        info["settings_file"] = ic.get("settings_file")
        info["dice_file"] = ic.get("dice_file")
    return info

# -------- resume helpers --------
STEP_ORDER = {"build":0, "prepare_ic":1, "simulate":2, "triax":3, "actions":4, "fit_plot":5, "compare":6}
def _step_enabled(name: str, start_step: str, until_step: str) -> bool:
    return STEP_ORDER[name] >= STEP_ORDER[start_step] and STEP_ORDER[name] <= STEP_ORDER[until_step]

def _marker(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

def _has_marker(path: Path) -> bool:
    return path.exists()

def _ensure_galaxy_general_for_model(paths: "Paths", model_name: str):
    """
    If resuming at a late step and galaxy_general/ is missing, but the per-model
    folder galaxy_general_<model>/ exists, temporarily restore it as galaxy_general/.
    """
    gg = paths.folder_simu_setting / "galaxy_general"
    if gg.exists():
        return
    ggm = paths.folder_simu_setting / f"galaxy_general_{model_name}"
    if ggm.exists():
        ggm.rename(gg)
        print(f"[resume] Restored {ggm.name} -> galaxy_general/ for model {model_name}")



#### Path layout (mirrors run.bash)
@dataclass
class Paths:
    home: Path
    folder_main: Path
    folder_thisFile: Path

    folder_packages: Path
    folder_dependencies: Path
    folder_make_DICE: Path

    folder_initial_condition: Path
    folder_IC_setting: Path
    folder_IC_trans: Path
    folder_DICE: Path
    folder_cold: Path

    folder_simulation: Path
    folder_simu_setting: Path
    folder_simu_source: Path
    folder_simu_si: Path
    folder_gm: Path
    folder_snapshot_txt: Path

    folder_actions: Path
    folder_SCF: Path
    folder_foci_table: Path
    folder_AA: Path

    folder_process: Path
    folder_output_fig: Path

    @staticmethod
    def from_base(home: Optional[Path] = None) -> "Paths":
        # #: [path] An example of fixed path
        # home = home or Path.home()
        # folder_main = home / "workroom/0prog/GroArnold_framework/"
        # folder_thisFile = folder_main / "install_and_run/"

        #: [path] Derive from *this file's* location so the project is relocatable.
        home = home or Path.home()
        _script_dir = Path(__file__).resolve().parent  # .../GroArnold_framework/install_and_run
        folder_thisFile = _script_dir
        folder_main = (_script_dir / "..").resolve()   # ← effectively "../" from this file

        folder_packages = folder_main / "packages/"
        folder_dependencies = folder_main / "GDDFAA/dependencies/"
        folder_make_DICE = folder_main / "GDDFAA/step1_galaxy_IC_preprocess/step2_select_generate_IC_DICE/"

        folder_initial_condition = folder_main / "GDDFAA/step1_galaxy_IC_preprocess/"
        folder_IC_setting = folder_initial_condition / "step1_set_IC_DDFA/"
        folder_IC_trans = folder_initial_condition / "step3_preprocess_IC/step1_from_ascii_to_g1_and_run/"
        folder_DICE = folder_initial_condition / "step2_select_generate_IC_DICE/"
        folder_cold = folder_initial_condition / "step2_select_generate_IC_python/"

        folder_simulation = folder_main / "GDDFAA/step2_Nbody_simulation/gadget/"
        folder_simu_setting = folder_simulation / "Gadget-2.0.7/"
        folder_simu_source = folder_simulation / "Gadget-2.0.7/Gadget2/"
        folder_simu_si = folder_simulation / "settings_and_info/"
        folder_gm = folder_simu_setting / "galaxy_general/"
        folder_snapshot_txt = folder_simu_setting / "galaxy_general/txt/"

        folder_actions = folder_main / "GDDFAA/step3_actions/"
        folder_SCF = folder_actions / "step2_Nbody_TACT/DataInterface/SCF/SCF_coeff_pot/"
        # Keep the relative mapping like run.bash (${folder_SCF}./orbitIntegSCF_adjust_a2b2/src/)
        folder_foci_table = folder_actions / "step2_Nbody_TACT/DataInterface/SCF/orbitIntegSCF_adjust_a2b2/src/"
        folder_AA = folder_actions / "step2_Nbody_TACT/aa/"

        folder_process = folder_main / "GDDFAA/step4_data_process/data_process/"
        folder_output_fig = folder_main / "GDDFAA/step4_data_process/data_process/savefig/"

        return Paths(home, folder_main, folder_thisFile, folder_packages, folder_dependencies, folder_make_DICE,
                     folder_initial_condition, folder_IC_setting, folder_IC_trans, folder_DICE, folder_cold,
                     folder_simulation, folder_simu_setting, folder_simu_source, folder_simu_si, folder_gm, folder_snapshot_txt,
                     folder_actions, folder_SCF, folder_foci_table, folder_AA, folder_process, folder_output_fig)



#### Settings readers
def parse_user_settings_multi(file: Path):
    lines = non_comment_lines(read_lines_keep(file))
    models = []
    times = []
    aa_line = None
    fit_flag = None
    for idx, ln in enumerate(lines):
        toks = ln.split()
        if not models and any(c.isalpha() for c in ln) and len(toks) >= 1:
            models = toks
            # find likely 'times' line
            for ln2 in lines[idx+1:]:
                toks2 = ln2.split()
                is_numeric = sum(t.replace('.', '', 1).replace('-', '', 1).isdigit() for t in toks2)
                if is_numeric >= max(2, len(toks2)//2):
                    times = toks2
                    # AA line
                    for ln3 in lines[idx+2:]:
                        if ln3.strip().startswith("mpirun "):
                            aa_line = ln3.strip()
                            break
                    # fit flag
                    for ln4 in lines[idx+2:]:
                        if ln4.strip().isdigit():
                            fit_flag = int(ln4.strip())
                            break
                    break
            break
    return models, times, aa_line, fit_flag

def parse_IC_setting_list(file: Path) -> List[str]:
    lines = non_comment_lines(read_lines_keep(file))
    names = []
    for ln in lines:
        toks = ln.split()
        if toks:
            names = toks
            break
    return names



#### Controller steps
# def step_install(paths: Paths, logdir: Path):
#     install_sh = paths.folder_thisFile / "install.bash"
#     if install_sh.exists():
#         run_cmd(["bash", str(install_sh.name)], cwd=paths.folder_thisFile, log=logdir/"install.log")
#     else:
#         print("[install] install.bash not found; skipping.")

def step_prepare_IC(paths: Paths, model_name: str, tag_IC: int, is_modify_IC: int, logdir: Path):
    folder_IC_setting = paths.folder_IC_setting
    filename_IC = f"IC_param_{model_name}.txt"
    src_param = folder_IC_setting / filename_IC
    dst_param = folder_IC_setting / "IC_param.txt"
    # if not src_param.exists():
    #     raise FileNotFoundError(f"[IC] Cannot find {src_param}")
    copy(src_param, dst_param)

    gm = paths.folder_gm
    copy(dst_param, gm/"IC_param.txt")

    for f in paths.folder_IC_trans.glob("*.g1"):
        try:
            f.unlink()
        except Exception:
            pass

    if tag_IC == 1:
        run_dir = paths.folder_DICE / "run"
        run_cmd(["bash", "step1_compile.bat"], cwd=paths.folder_DICE, log=logdir/"dice_compile.log") #note: compile DICE
        print("Prepare DICE exe binary, done.")
        copy(paths.folder_DICE/"build/bin/dice", run_dir/"dice")
        (run_dir/"params_files").mkdir(parents=True, exist_ok=True)
        copy(folder_IC_setting/f"IC_DICE_{model_name}.params", run_dir/"params_files/galaxy_general.params")
        copy(folder_IC_setting/"galaxy_general.config", run_dir/"galaxy_general.config")
        run_cmd(["bash", "step2_run.bat"], cwd=paths.folder_DICE, log=logdir/"dice_run.log")
        for p in (paths.folder_DICE/"run").glob("*"):
            if p.is_file():
                copy(p, paths.folder_IC_trans/p.name)
    elif tag_IC == 2:
        run_cmd(["bash", "step1_compile.bat"], cwd=paths.folder_cold, log=logdir/"cold_run.log")
        run_cmd(["bash", "step2_run.bat"], cwd=paths.folder_cold, log=logdir/"cold_run.log")
    elif tag_IC == 9:
        print("[IC] Skip IC generation (debug).")
        raise SystemExit(f"[IC] Skipped.")
    else:
        raise SystemExit(f"[IC] Unknown tag_IC={tag_IC}")

    #: do not comment this next line for snapshots reading and writing
    run_cmd(["bash", "step1_compile.bat"], cwd=paths.folder_IC_trans, log=logdir/"ic_trans_compile.log") #note: compile IC reading and writing
    print("Prepare snapshots reading and writing exe binary, done.")
    if is_modify_IC == 1:
        run_cmd(["./read_snapshot.exe", "2", "galaxy_general.g1"],
                cwd=paths.folder_IC_trans, log=logdir/"ic_convert.log")
        run_cmd(["python3", "modify_initial_condition.py", "galaxy_general.txt"],
                cwd=paths.folder_process, log=logdir/"ic_convert.log") #adjust spin to extreme
        (paths.folder_IC_trans/"galaxy_general.txt.modified.txt").rename(paths.folder_IC_trans/"galaxy_general.modified.txt")
        run_cmd(["./read_snapshot.exe", "1", "galaxy_general.modified.txt"],
                cwd=paths.folder_IC_trans, log=logdir/"ic_convert.log")
        (paths.folder_IC_trans/"galaxy_general.g1").unlink(missing_ok=True) #delete old .g1 file
        (paths.folder_IC_trans/"galaxy_general.modified.txt.g1").rename(paths.folder_IC_trans/"galaxy_general.g1")
    else:
        run_cmd(["bash", "step2_run.bat", "0"], cwd=paths.folder_IC_trans, log=logdir/"ic_convert.log")
    # sys.exit(2) #debug is_modify_IC

    #put files **into gm/** (NOT into parent Gadget-2.0.7)
    copy(paths.folder_IC_setting/f"IC_DICE_{model_name}.params", paths.folder_gm/"galaxy_general.params")
    copy(paths.folder_IC_setting/"galaxy_general.config", paths.folder_gm/"galaxy_general.config")
    copy(paths.folder_IC_trans/"galaxy_general.g1", paths.folder_gm/"galaxy_general.g1")
    # sys.exit(2) #debug whether copy

def step_build_galaxy_folder(paths: Paths, model_name: str, history_tag: str, logdir: Path,
                             is_build_gmfolder: int, is_rename_back_folder: int, reuse_existing_gm: bool = True, args_input_resume=None):
    # Before (re)building, back up any pre-existing same-name folders so "python3 ... all"
    # never overwrites prior data. Resume runs that skip the "build" step won't hit this.
    for _name in (f"galaxy_general_{model_name}", "galaxy_general"):
        target = paths.folder_simu_setting / _name
        # print(args_input_resume)
        if target.exists() and args_input_resume is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = target.with_name(f"{_name}_bak_{ts}")
            target.rename(backup)
            print(f"[build] Found existing {_name}; backed up to {backup.name}")

    if is_rename_back_folder != 0:
        return

    if is_build_gmfolder == 1:
        gg = paths.folder_simu_setting/"galaxy_general"
        # gg_bak = paths.folder_simu_setting/"galaxy_general_bak"
        # #if galaxy_general/ already exists and we want to reuse it, skip re-mk.
        # if gg.exists() and any(gg.iterdir()) and reuse_existing_gm:
        #     print("[build] galaxy_general/ exists; reuse enabled -> skipping rebuild.")
        # else:
        #     if gg_bak.exists():
        #         run_cmd(["bash", "-lc", f"rm -rf {shlex.quote(str(gg_bak))}"], cwd=paths.folder_simu_setting, log=logdir/"rm_gg_bak.log")
        #     if gg.exists():
        #         gg.rename(gg_bak)
        #     gg.mkdir(parents=True, exist_ok=True)

        gm = paths.folder_gm
        ensure_dir(gm)
        ensure_dir(gm/"snapshot")
        # sys.exit(2) #debug for galaxy_general.g1
        ensure_dir(gm/"debug")
        ensure_dir(gm/"intermediate")

        ensure_dir(gm/"init")
        # Write early per-model info (so you know modelnumber, flatz1, etc.)
        yml_path = detect_unified_yaml(None)
        _mi = get_model_info_by_modelname(yml_path, model_name)
        (gm/"init"/"model_info.txt").write_text(
            "model_info\n"
            f"  modelname: { _mi.get('modelname') }\n"
            f"  modelnumber: { _mi.get('modelnumber') }\n"
            f"  flatz1: { _mi.get('flatz1') }\n"
            f"  potential_rotating: { _mi.get('potential_rotating') }\n",
            encoding="utf-8"
        )

        ensure_dir(gm/"txt")
        ensure_dir(gm/"fit")
        ensure_dir(gm/"particle")
        ensure_dir(gm/"orbit")
        ensure_dir(gm/"aa")

        (gm/"model_infomation.txt").write_text(f"model ?, {model_name}, {history_tag}\n", encoding='utf-8')
        run_cmd(["bash", "step1_compile.bat"], cwd=paths.folder_simu_source, log=logdir/"gadget2_compile.log") #note: compile Gadget2
        copy(paths.folder_simu_source/"Gadget2", gm/"Gadget2")
        print("Prepare Gadget2 exe binary, done.")
        # copy(paths.folder_IC_setting/"IC_param.txt", gm/"IC_param.txt")

        # ini files from this install_and_run folder
        for fn in ("user_settings_multi.txt","IC_DICE_manucraft.params","IC_setting_list.txt","run.param"):
            src = paths.folder_thisFile/fn
            if src.exists():
                copy(src, gm/"init"/src.name)

        # # IC example into gm/
        # copy(paths.folder_IC_trans/"galaxy_general.g1", gm/"galaxy_general_example.g1")

        # settings_and_info/* -> gm/  (include simulate.bat, etc.)
        for p in (paths.folder_simu_si).glob("*"):
            if p.is_file():
                copy(p, gm/p.name)
        # explicitly ensure these two end up in the right places
        copy(paths.folder_simu_si/"convert_to_txt.bat", gm/"txt/convert_to_txt.bat")
        copy(paths.folder_simu_si/"example_snapshot_DF_params_fit.xv.txt", gm/"aa/example_snapshot_DF_params_fit.xv.txt")
        # Guarantee simulate.bat + run.param are in gm/, normalize line endings, set exec bits
        # (run.bash expects to run from galaxy_general/, and run.param is read from CWD)
        rp_src_this = paths.folder_thisFile/"run.param"
        if rp_src_this.exists():
            copy(rp_src_this, gm/"run.param")
        # Normalize scripts possibly copied from Windows
        _ensure_unix_line_endings(gm/"simulate.bat")
        # Gadget2 binary must be executable as well
        _make_executable(gm/"Gadget2")

def step_simulation(paths: Paths, model_name: str, n_mpi_gadget: int, is_rename_back_folder: int, logdir: Path):
    """
    Mirror run.bash section:
      cp IC_DICE_*.params -> ./galaxy_general.params
      cp galaxy_general.config -> ./galaxy_general.config
      cp .../galaxy_general.g1 -> ./
      (run from gm dir)
    """
    if is_rename_back_folder != 0:
        pass

    # #put files **into gm/** (NOT into parent Gadget-2.0.7)
    # copy(paths.folder_IC_setting/f"IC_DICE_{model_name}.params", paths.folder_gm/"galaxy_general.params")
    # copy(paths.folder_IC_setting/"galaxy_general.config", paths.folder_gm/"galaxy_general.config")
    # copy(paths.folder_IC_trans/"galaxy_general.g1", paths.folder_gm/"galaxy_general.g1")

    # skip if already completed (unless FORCE_RERUN via --resume_point)
    sim_done = paths.folder_gm/"simulate.done"
    if _has_marker(sim_done) and SKIP_EXISTING and not FORCE_RERUN:
        print("[simulate] simulate.done exists -> skipping simulation.")
    else:
        # run simulate in gm/, mimic run.bash redirection to simulate_tempinfo.txt
        cmd = f"bash simulate.bat {n_mpi_gadget} > simulate_tempinfo.txt 2>&1"
        print("Running simulation ...")
        run_cmd(cmd, cwd=paths.folder_gm, log=logdir/"simulate.log")
        _marker(sim_done)

def step_triaxial_alignment(paths: Paths, snapshot_ids: List[int], *,
                            is_run_actionerror: int, is_run_foci: int, logdir: Path):
    """
    Module 2: Triaxiality alignment & pre-actions prep:
      convert_to_txt -> fit_galaxy_distribution_function.py tag=1 -> compile SCF -> (optional) foci -> recalc tables
      Marks: intermediate/triax_<sid>.done (and foci_<sid>.done when foci run)
    """
    gm = paths.folder_gm
    txt_dir = paths.folder_snapshot_txt
    failures = []
    for i in snapshot_ids:
        sid = f"{int(i):03d}"
        triax_done = gm/"intermediate"/f"triax_{sid}.done"
        if _has_marker(triax_done) and SKIP_EXISTING and not FORCE_RERUN:
            print(f"[triax] snapshot {sid}: triax.done exists -> skipping.")
            continue
        try:
            run_cmd(["bash", "convert_to_txt.bat", f"snapshot_{sid}"],
                    cwd=txt_dir, log=logdir/f"convert_to_txt_{sid}.log")
            run_cmd(["python3", "fit_galaxy_distribution_function.py", "1", str(i), str(is_run_actionerror), "default"],
                    cwd=paths.folder_process, log=logdir/f"fit_df1_{sid}.log")
            run_cmd(["bash", "step1_2_compile_SCF.bat"], cwd=paths.folder_AA, log=logdir/f"compile_scf_{sid}.log")

            if is_run_foci == 1:
                print("Running for foci table ...")
                run_cmd(["bash", "step1_3_prepare_foci.bat"], cwd=paths.folder_AA, log=logdir/f"prepare_foci_{sid}.log")
                src = paths.folder_actions/"step2_Nbody_TACT/DataInterface/SCF/orbitIntegSCF_adjust_a2b2/src/some_lmn_foci_Pot.txt"
                dst = gm/"intermediate"/f"snapshot_{i}_lmn_foci_Pot.old.txt"
                copy(src, dst)
                orbit_src_dir = paths.folder_actions/"step2_Nbody_TACT/DataInterface/SCF/orbitIntegSCF_adjust_a2b2/src/orbit"
                dst_dir       = gm/"intermediate"/f"orbit_{i}"
                if orbit_src_dir.exists():
                    files = list(orbit_src_dir.glob("orbit_*.dat"))
                    if files:
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        if any(dst_dir.iterdir()):
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            bak = gm/"intermediate"/f"orbit_{i}_bak_{ts}"
                            bak.mkdir(parents=True, exist_ok=True)
                            for q in dst_dir.iterdir(): q.rename(bak/q.name)
                        for p in files: p.rename(dst_dir/p.name)
                run_cmd(["python3", "recalculate_foci_table.py", str(i)],
                        cwd=paths.folder_process, log=logdir/f"recalc_foci_{sid}.log")
                _marker(gm/"intermediate"/f"foci_{sid}.done")
            _marker(triax_done)
        except subprocess.CalledProcessError as e:
            failures.append((i, e.returncode))
            if not CONT_ON_ERR: raise
            print(f"[triax] snapshot {sid} FAILED (rc={e.returncode}); continuing due to --continue-on-error.")
    if failures:
        print("[triax] failures:", failures)

def step_actions_calculation(paths: Paths, snapshot_ids: List[int], *, logdir: Path):
    """
    Module 3: Actions calculation only (AA; Stäckel fudge run):
      Runs step2_run.bat for each snapshot
      Marks: aa/actions_<sid>.done
    """
    gm = paths.folder_gm
    failures = []
    for i in snapshot_ids:
        sid = f"{int(i):03d}"
        aa_done = gm/"aa"/f"actions_{sid}.done"
        if _has_marker(aa_done) and SKIP_EXISTING and not FORCE_RERUN:
            print(f"[actions] snapshot {sid}: actions.done exists -> skipping.")
            continue
        try:
            print("Running angle-actions ...")
            # run_cmd(["bash", "step1_1_compile_all.bat"], cwd=paths.folder_AA, log=logdir/f"actions_{sid}.log") #note: compile the C++ and Fortran files for angle-actions computation
            print("Prepare Nbody_TACT exe binary, done.")
            run_cmd(["bash", "step2_run.bat"], cwd=paths.folder_AA, log=logdir/f"actions_{sid}.log")
            _marker(aa_done)
        except subprocess.CalledProcessError as e:
            failures.append((i, e.returncode))
            if not CONT_ON_ERR: raise
            print(f"[actions] snapshot {sid} FAILED (rc={e.returncode}); continuing due to --continue-on-error.")
    if failures:
        print("[actions] failures:", failures)

def step_fit_and_plot(paths: Paths, snapshot_ids_fit: List[int], is_run_actionerror: int, logdir: Path):
    failures = []
    for i in snapshot_ids_fit:
        if is_run_actionerror == 1:
            run_cmd(["python3", "action_error_by_near_time.py", str(i),
                     Path(paths.folder_actions/"step1_preprocess/snapshot_interval_stde.txt").read_text().strip()],
                    cwd=paths.folder_process, log=logdir/f"action_err_{i:03d}.log")
        sid = f"{int(i):03d}"
        fit_done = paths.folder_gm/"fit"/f"fitplot_{sid}.done"
        if _has_marker(fit_done) and SKIP_EXISTING and not FORCE_RERUN:
            print(f"[fit_plot] snapshot {sid}: fitplot.done exists -> skipping.")
            continue
        try:
            print("Running fitting ...")
            run_cmd(["python3", "fit_galaxy_distribution_function.py", "2", str(i), "0", "default"],
                    cwd=paths.folder_process, log=logdir/f"fit_df2_{i:03d}.log")
            run_cmd(["python3", "plot_action_figs.py", "galaxy_general", str(i)],
                    cwd=paths.folder_process, log=logdir/f"plot_{i:03d}.log")
            _marker(fit_done)
        except subprocess.CalledProcessError as e:
            failures.append((i, e.returncode))
            if not CONT_ON_ERR:
                raise
            print(f"[fit_plot] snapshot {sid} FAILED (rc={e.returncode}); continuing due to --continue-on-error.")
    if failures:
        print("[fit_plot] failures:", failures)

def step_compare(paths: Paths, snapshot_ID_each_compared: int, is_run_actionerror: int, model0: str, logdir: Path):
    hist = paths.folder_thisFile / (f"history_runnings_{model0}/")
    ensure_dir(hist)
    Path(paths.folder_simu_setting/"params_statistics").mkdir(parents=True, exist_ok=True)
    for fn in ("user_settings_multi.txt", "run.param", "IC_setting_list.txt", "IC_DICE_manucraft.params"):
        src = paths.folder_thisFile / fn
        if src.exists():
            copy(src, hist/src.name)
    run_cmd(["python3", "fit_galaxy_distribution_function.py",
             "3", str(snapshot_ID_each_compared), str(is_run_actionerror), "read_mgs"],
            cwd=paths.folder_process, log=logdir/"compare.log")



def module_1_ic_and_sim(paths, model_name, is_modify_IC, args, *, logdir: Path):
    if _group_enabled("build", "simulate", args.start_step, args.until_step):
        print("module_1_ic_and_sim(), begin.")
        step_build_galaxy_folder(paths, model_name, "history_tag", logdir=logdir,
                                 is_build_gmfolder=args.is_build_gmfolder,
                                 is_rename_back_folder=args.is_rename_back_folder,
                                 reuse_existing_gm=True, args_input_resume=args.resume_point)
        print("step_build_galaxy_folder(), done.")
        # sys.exit(2) #debug whether build
    
        # if _step_enabled("prepare_ic", args.start_step, args.until_step):
        print("is_modify_IC: ", is_modify_IC)
        step_prepare_IC(paths, model_name, args.tag_ic, is_modify_IC, logdir=logdir)
        print("step_prepare_IC(), done.")
        # sys.exit(2) #debug whether build
    
        # if _step_enabled("simulate", args.start_step, args.until_step):
        step_simulation(paths, model_name, n_mpi_gadget=args.n_mpi_gadget,
                        is_rename_back_folder=args.is_rename_back_folder, logdir=logdir)
        print("step_simulation(), done.")
        # sys.exit(2) #debug whether build
        print("module_1_ic_and_sim(), end.")

def module_2_triax_alignment(paths: Paths, snaps_main: List[int], args, *, logdir: Path):
    if _step_enabled("triax", args.start_step, args.until_step):
        print("module_2_triax_alignment(), begin.")
        step_triaxial_alignment(paths, snaps_main,
                                is_run_actionerror=args.is_run_actionerror,
                                is_run_foci=args.is_run_foci,
                                logdir=logdir)
        print("module_2_triax_alignment(), end.")

def module_3_actions_calc(paths: Paths, snaps_main: List[int], args, *, logdir: Path):
    if _step_enabled("actions", args.start_step, args.until_step):
        print("module_3_actions_calc(), begin.")
        step_actions_calculation(paths, snaps_main, logdir=logdir)
        print("module_3_actions_calc(), end.")

def module_4_df_fit_and_plots(paths: Paths, snaps_fit: List[int], args, *, logdir: Path):
    if _step_enabled("fit_plot", args.start_step, args.until_step):
        print("module_4_df_fit_and_plots(), begin.")
        step_fit_and_plot(paths, snaps_fit, is_run_actionerror=args.is_run_actionerror, logdir=logdir)
        print("module_4_df_fit_and_plots(), end.")



def read_target_snapshots(paths: Paths) -> Tuple[List[int], List[int], int]:
    # Ensure step1_preprocess/ has the helper (run.bash copies it there)
    src = paths.folder_process/"tell_shell_read_what_argv.py"
    dst = paths.folder_actions/"step1_preprocess/tell_shell_read_what_argv.py"
    if src.exists():
        copy(src, dst)
    # Now run from that folder
    run_cmd(["python3", "tell_shell_read_what_argv.py", "0"],
            cwd=paths.folder_actions/"step1_preprocess/", log=paths.folder_thisFile/"logs/tell_shell.log")
    p1 = paths.folder_actions/"step1_preprocess/argv_target_snapshots.txt"
    p2 = paths.folder_actions/"step1_preprocess/argv_target_snapshots_DF.txt"
    p3 = paths.folder_actions/"step1_preprocess/snapshot_interval_stde.txt"
    def _read_ids(p: Path) -> List[int]:
        if not p.exists():
            return []
        txt = p.read_text(encoding='utf-8').strip().split()
        return [int(x) for x in txt if x.strip().lstrip('-').isdigit()]
    s1 = _read_ids(p1)
    s2 = _read_ids(p2)
    s3 = int((p3.read_text(encoding='utf-8').strip() if p3.exists() else "0").split()[0])
    return s1, s2, s3



def main():
    ap = argparse.ArgumentParser(prog="workflow_wrapper.py",
                                 description="Python controller for GroArnold_framework/ (translated from run.bash).")
    # Settings folder first (optional; defaults to autodetect)
    ap.add_argument("settings_dir", nargs="?", default=None,
                    help="Path to initial_conditions/settings_<NAME>/ containing unified_settings.yaml")
    # Action is optional; default to 'all' so users can just pass --resume_point
    ap.add_argument("action", nargs="?", default="all",
                    choices=["all", "install", "run", "compare"],
                    help="Which stage to execute (default: all).")

    ap.add_argument("--tag-ic", type=int, default=1, help="IC method (1=DICE, 2=cold_python, 9=skip).")
    ap.add_argument("--is-sim-little", type=int, default=0, help="Debug small-steps run (kept for parity).")
    ap.add_argument("--is-preprocess-before-sim", type=int, default=0, help="Preprocess IC before sim (0/1).")
    ap.add_argument("--is-rename-back-folder", type=int, default=0, help="Debug mode: reuse existing galaxy_general (0/1).")
    ap.add_argument("--is-rename-folder", type=int, default=1, help="After run, rename galaxy_general -> galaxy_general_<model> (0/1).")
    ap.add_argument("--is-build-gmfolder", type=int, default=1, help="Recreate galaxy_general tree (0/1).")
    ap.add_argument("--is-run-actionerror", type=int, default=0, help="Whether to compute action errors (0/1).")
    ap.add_argument("--is-run-foci", type=int, default=1, help="Whether to compute foci (0/1).")
    ap.add_argument("--n-mpi-gadget", type=int, default=4, help="MPI process count for Gadget2 & actions (as needed).")
    ap.add_argument("--detach", action="store_true", help="Detach (nohup-like) this controller run.")
    
    # --- resume / gating / convenience ---
    ap.add_argument("--model", type=str, default=None,
                    help="Run only this model name (must match a name from settings).")
    # ap.add_argument("--start-step", type=str, default="prepare_ic",
    ap.add_argument("--start-step", type=str, default="build",
                    choices=["build","prepare_ic","simulate","triax","actions","fit_plot","compare"],
                    help="Begin pipeline at this step (inclusive).")
    ap.add_argument("--until-step", type=str, default="compare",
                    choices=["build","prepare_ic","simulate","triax","actions","fit_plot","compare"],
                    help="Stop pipeline after this step (inclusive).")
    # resume_point now accepts two integers: <point> <cont_flag>
    #  - point: 1..6 as before
    #  - cont_flag: 0 => run only that module and then stop; 1 => resume from that module and run till end
    # Example:
    #   --resume_point 3 0   # run only module 3 (actions) and then exit
    #   --resume_point 3 1   # resume from module 3 and run till end
    ap.add_argument("--resume_point", nargs=2, type=int, metavar=("POINT", "CONT"),
                    help="Resume shortcut: POINT in [1..6], CONT in {0,1} (0=only module, 1=run till end)")
    ap.add_argument("--modelnumber", type=int, default=None,
                    help="Index of model in YAML models[] to target for resume (0-based). If omitted, all models are considered.")

    ap.add_argument("--continue-on-error", action="store_true",
                    help="Keep going across snapshots/models even if one fails.")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                    help="Skip steps/snapshots that already have '.done' markers (default).")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="Force redo even if '.done' markers exist.")
    ap.set_defaults(skip_existing=True)
    args = ap.parse_args()

    paths = Paths.from_base()
    logdir = paths.folder_thisFile / "logs"
    ensure_dir(logdir)

    # Globals for resume behavior — robust even if argparse flags are missing.
    global SKIP_EXISTING, CONT_ON_ERR, FORCE_RERUN
    SKIP_EXISTING   = getattr(args, "skip_existing", True)
    CONT_ON_ERR     = getattr(args, "continue_on_error", False)
    # If a resume point is requested, force re-run of gated steps regardless of '.done' markers.
    FORCE_RERUN     = bool(getattr(args, "resume_point", None))

    #IMPORTANT: refresh legacy text files from YAML *before* any parsing.
    #\ This avoids stale reads (e.g., is_modify_IC lagging one run behind).
    #\ Only required for actions that actually run the pipeline.
    if args.action in ("run", "all"):
        if args.settings_dir:
            yaml_path = (Path(args.settings_dir) / "unified_settings.yaml").resolve()
            run_cmd(["python3", "transform_yaml_settings.py", str(yaml_path)],
                    cwd=paths.folder_thisFile, log=logdir/"transform_yaml_settings.log")
            print(f"[build] Refreshed settings from {yaml_path}")
        else:
            raise SystemExit("[build] No path argument like initial_conditions/settings_*/; cannot refresh settings.")

    #Now parse the (fresh) legacy files
    models, times, aa_line, fit_flag = parse_user_settings_multi(paths.folder_thisFile/"user_settings_multi.txt")

    n_mpi_gadget = args.n_mpi_gadget
    snapshot_ID_each_compared = -1
    is_modify_IC = None
    if times:
        if len(times) >= 8:
            snapshot_ID_each_compared = int(times[5])
            n_mpi_gadget = int(times[6])
            is_modify_IC = int(times[7])
    else:
        raise SystemExit("[error] Wrong parsing from user_settings_multi.txt.")

    #Map resume_point -> (start_step, until_step, action) and compute models_to_run.
    models_orig = models[:]  # keep original full model list
    models_to_run = models_orig[:]  # default: run all
    # Per-model start policy
    first_model_start = None   # will be set if resuming
    later_models_start = "build"

    if args.resume_point:
        if len(args.resume_point) != 2:
            raise SystemExit("[error] --resume_point expects two integers: POINT CONT (e.g. --resume_point 3 1)")
        rp, cont_flag = int(args.resume_point[0]), int(args.resume_point[1])
        if cont_flag not in (0, 1):
            raise SystemExit("[error] CONT flag must be 0 or 1")

        #target model index if provided (decide run-range below based on CONT)
        model_idx = None
        if args.modelnumber is not None:
            model_idx = int(args.modelnumber)
            if model_idx < 0 or model_idx >= len(models_orig):
                raise SystemExit(f"[error] --modelnumber {model_idx} out of range (0..{len(models_orig)-1})")
            # # run only that model by default
            # models_to_run = [models_orig[model_idx]]

        # resume-point dispatch
        if rp == 1:
            # Module 1 should include build -> prepare_ic -> simulate
            start_step = "build"
        elif rp == 2:
            start_step = "triax"
        elif rp == 3:
            start_step = "actions"
        elif rp == 4:
            start_step = "fit_plot"
        elif rp == 5:
            # Special: rename galaxy_general -> galaxy_general_{model_idx_or_0}
            mi = model_idx if model_idx is not None else 0
            gg = paths.folder_simu_setting / "galaxy_general"
            target = paths.folder_simu_setting / f"galaxy_general_{models_orig[mi]}"
            if gg.exists():
                if target.exists():
                    # backup existing target if present, to avoid override
                    bak = paths.folder_simu_setting / f"{target.name}_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    target.rename(bak)
                    print(f"[resume5] Backed up existing {target.name} -> {bak.name}")
                gg.rename(target)
                print(f"[resume5] Renamed existing galaxy_general -> {target.name}")
            # After rename: if cont_flag==1, continue processing remaining models (after mi)
            if cont_flag == 1:
                models_to_run = models_orig[mi+1:]
                start_step = "build"
            else:
                # cont_flag == 0 -> only do rename then exit
                print("[resume5] Per CONT=0: rename-only requested -> exiting")
                sys.exit(0)
        elif rp == 6:
            start_step = "compare"
        else:
            raise SystemExit("[error] resume point must be in 1..6")

        # determine until_step based on cont_flag
        if cont_flag == 0:
            until_step = start_step
        else:
            until_step = "compare"

        # set the args so the rest of the script uses them
        args.start_step, args.until_step = start_step, until_step
        # args.action = "run" if start_step != "compare" else "compare"
        #if CONT=1 (run till end), we also want the compare stage after model loop.
        #\ Set action to "all" so the final compare runs (see bottom of main()).
        if start_step == "compare":
            args.action = "compare"
        else:
            args.action = "all" if cont_flag == 1 else "run"

        #if one specified a modelnumber, decide range by CONT (except rp==5 which already set it)
        if args.modelnumber is not None and rp != 5:
            if cont_flag == 1:
                # resume from this model and run through the end
                models_to_run = models_orig[model_idx:]
            else:
                # only this model
                models_to_run = [models_orig[model_idx]]

        # record per-model policy
        first_model_start = start_step
        if rp == 5:
            # rename-only already handled; for the remaining models, always rebuild from scratch
            later_models_start = "build"

    else:
        # No resume_point: default behavior run all models
        models_to_run = models_orig[:]

    if args.detach:
        pidfile = paths.folder_thisFile/"controller.pid"
        log = paths.folder_thisFile/"controller.nohup.log"
        cmd = ["python3", str(Path(__file__).name), args.action] + \
              [f"--{k.replace('_','-')}={v}" for k, v in vars(args).items()
               if k not in ("action", "detach") and v is not None and v != False]
        run_cmd_detach(cmd, cwd=paths.folder_thisFile, log=log, pidfile=pidfile)
        return

    # if args.action == "install":
    #     step_install(paths, logdir)
    #     return

    ## run
    if args.action in ("run", "all"):
        # #Always refresh legacy text files from the YAML as part of build
        # #\ Compile unified_settings.yaml -> legacy files (prefer explicit settings_dir)
        # if args.settings_dir:
        #     yaml_path = (Path(args.settings_dir) / "unified_settings.yaml").resolve()
        #     run_cmd(["python3", "transform_yaml_settings.py", str(yaml_path)],
        #         cwd=paths.folder_thisFile, log=logdir/"transform_yaml_settings.log")
        
        #     print("Set or reset initial conditions from %s, done."%(yaml_path))
        # else:
        #     raise SystemExit("[build] No path argument like initial_conditions/settings_*/, transform_yaml_settings failed.")

        if args.is_rename_folder == 1 and args.is_build_gmfolder == 1 and args.is_rename_back_folder == 0:
            if (paths.folder_process/"change_params_galaxy_init.py").exists():
                run_cmd(["python3", "change_params_galaxy_init.py"], cwd=paths.folder_process, log=logdir/"change_params.log")

        # Remember original gating to restore after loop
        orig_start, orig_until = args.start_step, args.until_step

        for idx, model_name in enumerate(models_to_run):
            print(f"[model] Begin: {model_name}")

            # Decide per-model start/until:
            if args.resume_point:
                rp, cont_flag = int(args.resume_point[0]), int(args.resume_point[1])
                if idx == 0:
                    args.start_step = first_model_start
                    args.until_step = ("compare" if cont_flag == 1 else first_model_start)
                else:
                    args.start_step = later_models_start
                    args.until_step = ("compare" if cont_flag == 1 else later_models_start)
            # If starting from a late step, ensure galaxy_general/ exists (rehydrate if needed)
            if STEP_ORDER[args.start_step] >= STEP_ORDER["triax"]:
                _ensure_galaxy_general_for_model(paths, model_name)

            ## Module 1: IC & Simulation (resume_point 1)
            module_1_ic_and_sim(paths, model_name, is_modify_IC, args, logdir=logdir/model_name)

            #snapshot targets (used by Modules 2-4)
            snaps_main, snaps_fit, snap_interval = read_target_snapshots(paths)
            print(f"[snapshots] main={snaps_main} fit={snaps_fit} (interval={snap_interval})")

            ## Module 2: Triaxial alignment (pre-actions) (resume_point 2)
            module_2_triax_alignment(paths, snaps_main, args, logdir=logdir/model_name)

            ## Module 3: Actions calculation (AA integration) (resume_point 3)
            module_3_actions_calc(paths, snaps_main, args, logdir=logdir/model_name)

            ## Module 4: DF fits & plots (resume_point 4)
            module_4_df_fit_and_plots(paths, snaps_fit, args, logdir=logdir/model_name)

            #If this is a one-shot resume (CONT=0), exit immediately after the requested module.
            #\ Do not rename the folder in debug mode.
            if args.resume_point and int(args.resume_point[1]) == 0:
                print(f"[resume] CONT=0: exiting after requested module for model {model_name}")
                sys.exit(0)

            ## rename folder and then do another loop (resume_point 5)
            _cont_flag = int(args.resume_point[1]) if args.resume_point else 1
            if args.is_rename_folder == 1 and _cont_flag == 1:
                gg = paths.folder_simu_setting/"galaxy_general"
                target = paths.folder_simu_setting/f"galaxy_general_{model_name}"
                if target.exists():
                    backup = paths.folder_simu_setting/f"{target.name}_bak"
                    run_cmd(["bash", "-lc", f"rm -rf {shlex.quote(str(backup))}"], cwd=paths.folder_simu_setting, log=logdir/"rm_target_bak.log")
                    target.rename(backup)
                if gg.exists():
                    gg.rename(target)
                print(f"[model] Renamed galaxy folder -> {target.name}")

            print(f"[model] Done: {model_name}")

        #restore original gating after the loop
        args.start_step, args.until_step = orig_start, orig_until

    ## compare after each model (resume_point 6)
    if args.action in ("compare", "all"):
        model0 = models[0]
        step_compare(paths, snapshot_ID_each_compared, args.is_run_actionerror, model0, logdir=logdir/model0)
        print("[compare] Done.")

    print("[controller] All requested stages completed.")



#### main
if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"[error] Command failed (rc={e.returncode}): {e.cmd}", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
