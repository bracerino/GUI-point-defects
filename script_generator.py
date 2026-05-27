"""Build a *standalone* Python script that reproduces a batch defect-generation run.

Used by the Streamlit app: the user can view/download the script returned by
:func:`build_batch_script` and run it locally to create every structure on disk.

The generated script is self-contained for the selected task: the relevant defect
routines (and their dependencies) are extracted from ``helpers_defects`` and embedded
directly into the script, so it does NOT import from the app. It only needs the usual
scientific stack (pymatgen, ase, numpy, scipy).
"""

import ast
import inspect

import numpy as np

import helpers_defects as _hd

_ALGORITHMS = {"original", "iterative", "genetic", "global_exact", "adaptive"}


def _clean(obj):
    """Recursively convert numpy scalar types to native Python so ``repr`` is valid."""
    if isinstance(obj, dict):
        return {_clean(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _hd_functions():
    """Map of every function defined in helpers_defects (by name)."""
    return {name: obj for name, obj in vars(_hd).items()
            if inspect.isfunction(obj) and getattr(obj, "__module__", None) == _hd.__name__}


def _closure(entry_names, defined, preseen=None):
    """Transitive set of helpers_defects functions reachable from ``entry_names``."""
    seen = set(preseen or set())
    order = []
    stack = list(entry_names)
    while stack:
        name = stack.pop()
        if name in seen or name not in defined:
            continue
        seen.add(name)
        order.append(name)
        src = inspect.getsource(defined[name])
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Name) and node.id in defined and node.id not in seen:
                stack.append(node.id)
    return order


def _strip_streamlit(src):
    """Neutralize Streamlit references in an extracted function source.

    Drops ``import streamlit`` lines and replaces standalone ``st.*`` statements
    with ``pass`` (the only such calls in the embedded routines are UI messages).
    """
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        if stripped.startswith("import streamlit") or stripped.startswith("from streamlit"):
            continue
        if stripped.startswith("st."):
            out.append(indent + "pass  # (Streamlit UI call removed)")
            continue
        out.append(line)
    return "\n".join(out)


def _custom_select_spaced_points(algorithm):
    """A minimal, st-free ``select_spaced_points`` hardwired to the chosen algorithm."""
    return (
        "def select_spaced_points(frac_coords_list, n_points, mode, target_value=0.5, random_seed=None):\n"
        "    if not frac_coords_list or n_points == 0:\n"
        "        return [], []\n"
        "    return select_spaced_points_%s(frac_coords_list, n_points, mode, target_value, random_seed)\n"
        % algorithm
    )


def _build_embedded_functions(operation_mode, algorithm):
    """Return source text defining the defect routines needed for ``operation_mode``."""
    defined = _hd_functions()
    algorithm = algorithm if algorithm in _ALGORITHMS else "original"

    preseen = set()
    if operation_mode == "Substitute Atoms":
        entries = ["substitute_atoms_in_structure", "select_spaced_points_%s" % algorithm]
        need_select = True
    elif operation_mode == "Create Vacancies":
        entries = ["remove_vacancies_from_structure", "select_spaced_points_%s" % algorithm]
        need_select = True
    elif operation_mode == "Create Substitution Cluster":
        entries = ["create_substitution_cluster", "create_substitution_cluster_rectangular"]
        need_select = False
    else:
        entries, need_select = [], False

    if need_select:
        # We emit our own select_spaced_points, so don't extract the (st-dependent) one.
        preseen.add("select_spaced_points")

    order = _closure(entries, defined, preseen=preseen)

    blocks = ["# " + "=" * 75,
              "# Defect routines embedded from the XRDlicious app (no app import needed).",
              "# Point-selection algorithm: %s" % algorithm,
              "# " + "=" * 75]
    if need_select:
        blocks.append(_custom_select_spaced_points(algorithm))
    for name in order:
        blocks.append(_strip_streamlit(inspect.getsource(defined[name])).rstrip())
    return "\n\n".join(blocks)


# Fixed body: settings consumers + generation logic. References the constants and the
# embedded functions defined above. Plain (non-f) string so all braces are literal.
_SCRIPT_BODY = r'''
# ===========================================================================
# Generation driver
# ===========================================================================
base_structure = Structure.from_str(BASE_POSCAR, fmt="poscar")


def write_structure(structure, rel_path_no_ext):
    full = os.path.join(OUTPUT_DIR, rel_path_no_ext)
    os.makedirs(os.path.dirname(full), exist_ok=True)

    if OUTPUT_FORMAT == "CIF":
        content = str(CifWriter(structure, symprec=0.1, refine_struct=False))
        with open(full + ".cif", "w") as fh:
            fh.write(content)

    elif OUTPUT_FORMAT == "VASP":
        ase_atoms = AseAtomsAdaptor.get_atoms(structure)
        sio = StringIO()
        ase_write(sio, ase_atoms, format="vasp", direct=True, sort=True)
        with open(full + ".poscar", "w") as fh:
            fh.write(sio.getvalue())

    elif OUTPUT_FORMAT == "LAMMPS":
        ase_atoms = AseAtomsAdaptor.get_atoms(structure)
        sio = StringIO()
        ase_write(sio, ase_atoms, format="lammps-data",
                  atom_style=LAMMPS_ATOM_STYLE, units=LAMMPS_UNITS, masses=True)
        with open(full + "_" + LAMMPS_ATOM_STYLE + ".lmp", "w") as fh:
            fh.write(sio.getvalue())

    elif OUTPUT_FORMAT == "XYZ":
        lattice = structure.lattice.matrix
        lines = [str(len(structure))]
        latstr = " ".join(f"{x:.6f}" for row in lattice for x in row)
        lines.append('Lattice="' + latstr + '" Properties=species:S:1:pos:R:3')
        for site in structure:
            c = structure.lattice.get_cartesian_coords(site.frac_coords)
            lines.append(f"{site.specie.symbol} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
        with open(full + ".xyz", "w") as fh:
            fh.write("\n".join(lines))

    else:
        raise ValueError("Unknown OUTPUT_FORMAT: " + str(OUTPUT_FORMAT))


def _pct_from_value(el, value, ranges):
    spec = ranges[el]
    if spec["unit"] == "count":
        cnt = spec["count"]
        return (value / cnt * 100.0) if cnt > 0 else 0.0
    return value


def generate_grid():
    elems = GRID_VARY["elements"]
    ranges = GRID_VARY["ranges"]
    value_lists = [ranges[e]["values"] for e in elems]
    combos = list(itertools.product(*value_lists))
    n_folders = len(combos)
    total = n_folders * CONFIGS_PER_CONCENTRATION
    print("Generating %d structure(s) across %d folder(s) into '%s/' ..."
          % (total, n_folders, OUTPUT_DIR))

    counter = 0
    for fi, combo in enumerate(combos, 1):
        parts = []
        for e, v in zip(elems, combo):
            if ranges[e]["unit"] == "percentage":
                parts.append("%s_%.2fperc" % (e, v))
            else:
                parts.append("%s_%datoms" % (e, int(v)))
        folder = "/".join(parts)
        print("[folder %d/%d] %s  ->  generating %d structure(s)..."
              % (fi, n_folders, folder, CONFIGS_PER_CONCENTRATION))

        for rep in range(CONFIGS_PER_CONCENTRATION):
            seed = STARTING_SEED + counter
            base = base_structure.copy()

            if GRID_VARY["op"] == "substitute":
                settings = {}
                for el, cfg in SUB_SETTINGS.items():
                    target = cfg.get("substitute", "")
                    if el in elems:
                        pct = _pct_from_value(el, combo[elems.index(el)], ranges)
                        settings[el] = {"percentage": pct, "substitute": target}
                    elif cfg.get("percentage", 0) > 0:
                        settings[el] = {"percentage": cfg["percentage"], "substitute": target}
                    else:
                        settings[el] = {"percentage": 0, "substitute": target}
                out = substitute_atoms_in_structure(base, settings, SUB_MODE, SUB_TARGET, None, seed)
            else:  # vacancy
                vac = {}
                for el, pct0 in VAC_PERCENT.items():
                    if el in elems:
                        vac[el] = _pct_from_value(el, combo[elems.index(el)], ranges)
                    elif pct0 > 0:
                        vac[el] = pct0
                    else:
                        vac[el] = 0
                out = remove_vacancies_from_structure(base, vac, VAC_MODE, VAC_TARGET, None, seed)

            write_structure(out, "%s/config_rep%02d_seed%d" % (folder, rep + 1, seed))
            counter += 1

        remaining = total - counter
        print("    finished '%s'  (%d/%d structures done, %d remaining, %d folder(s) left)"
              % (folder, counter, total, remaining, n_folders - fi))
    return counter


def generate_cluster():
    counts = CLUSTER["counts"]
    n_folders = len(counts)
    total = n_folders * CONFIGS_PER_CONCENTRATION
    print("Generating %d structure(s) across %d folder(s) into '%s/' ..."
          % (total, n_folders, OUTPUT_DIR))

    counter = 0
    for fi, num_subs in enumerate(counts, 1):
        num_subs = int(num_subs)
        folder = "%dsubs" % num_subs
        print("[folder %d/%d] %s  ->  generating %d structure(s)..."
              % (fi, n_folders, folder, CONFIGS_PER_CONCENTRATION))
        for rep in range(CONFIGS_PER_CONCENTRATION):
            seed = STARTING_SEED + counter
            base = base_structure.copy()
            if CLUSTER["shape"] == "Spherical":
                out = create_substitution_cluster(
                    base, CLUSTER["orig_el"], CLUSTER["sub_el"], num_subs,
                    CLUSTER["radius"], None, seed,
                    delete_non_clustered_original_elements=CLUSTER["delete_others"])
            else:
                out = create_substitution_cluster_rectangular(
                    base, CLUSTER["orig_el"], CLUSTER["sub_el"], num_subs,
                    tuple(CLUSTER["block"]), None, seed,
                    delete_non_clustered_original_elements=CLUSTER["delete_others"])
            write_structure(out, "%s/config_%dsubs_rep%02d_seed%d" % (folder, num_subs, rep + 1, seed))
            counter += 1

        remaining = total - counter
        print("    finished '%s'  (%d/%d structures done, %d remaining, %d folder(s) left)"
              % (folder, counter, total, remaining, n_folders - fi))
    return counter


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if OPERATION in ("Substitute Atoms", "Create Vacancies") and GRID_VARY:
        written = generate_grid()
    elif OPERATION == "Create Substitution Cluster" and CLUSTER:
        written = generate_cluster()
    else:
        print("Nothing to generate for OPERATION=%r with the given settings." % OPERATION)
        return
    print("Done. Wrote %d structures to '%s/'." % (written, OUTPUT_DIR))


if __name__ == "__main__":
    main()
'''


def build_batch_script(*, operation_mode, base_structure, total_configs,
                       configs_per_count, starting_seed, output_format="CIF",
                       lammps_atom_style="atomic", lammps_units="metal",
                       grid_vary=None, sub_settings=None, sub_mode="random", sub_target=0.5,
                       vac_percent=None, vac_mode="random", vac_target=0.5,
                       cluster_params=None, point_selection_algorithm="original"):
    """Return a standalone Python script (string) that regenerates the batch.

    The defect routines for ``operation_mode`` are embedded directly, so the script
    has no dependency on the app beyond the standard scientific stack.
    """
    base_poscar = base_structure.to(fmt="poscar")
    embedded = _build_embedded_functions(operation_mode, point_selection_algorithm)

    preamble = (
        '#!/usr/bin/env python3\n'
        '"""Auto-generated by XRDlicious - Point Defects.\n'
        '\n'
        'Standalone reproduction of a batch defect-generation run (%d structures).\n'
        'It embeds the required defect routines, so just run it anywhere with the\n'
        'scientific stack installed:\n'
        '\n'
        '    python %s\n'
        '\n'
        'Requirements: numpy, scipy, pymatgen, ase.\n'
        'Structures are written under OUTPUT_DIR (edit the constants below to change\n'
        'the output folder or file format).\n'
        '"""\n'
        'import os\n'
        'import itertools\n'
        'import random\n'
        'from io import StringIO\n'
        '\n'
        'import numpy as np\n'
        'from pymatgen.core import Structure, Element\n'
        'from pymatgen.io.ase import AseAtomsAdaptor\n'
        'from pymatgen.io.cif import CifWriter\n'
        'from ase.io import write as ase_write\n'
        'try:\n'
        '    from scipy.spatial import cKDTree\n'
        'except Exception:\n'
        '    cKDTree = None\n'
        'try:\n'
        '    from scipy.optimize import differential_evolution\n'
        'except Exception:\n'
        '    differential_evolution = None\n'
        '\n\n'
    ) % (int(total_configs), "generate_defects.py")

    settings = (
        '\n\n'
        '# ===========================================================================\n'
        '# Settings (safe to edit)\n'
        '# ===========================================================================\n'
        'OUTPUT_DIR = "generated_structures"\n'
        'OUTPUT_FORMAT = %r          # "CIF" | "VASP" | "LAMMPS" | "XYZ"\n'
        'LAMMPS_ATOM_STYLE = %r\n'
        'LAMMPS_UNITS = %r\n'
        'STARTING_SEED = %d\n'
        'CONFIGS_PER_CONCENTRATION = %d\n'
        '\n'
        '# Generation parameters captured from the app (usually no need to edit)\n'
        'OPERATION = %r\n'
        'GRID_VARY = %r\n'
        'SUB_SETTINGS = %r\n'
        'SUB_MODE = %r\n'
        'SUB_TARGET = %r\n'
        'VAC_PERCENT = %r\n'
        'VAC_MODE = %r\n'
        'VAC_TARGET = %r\n'
        'CLUSTER = %r\n'
        '\n'
        'BASE_POSCAR = r"""%s"""\n'
    ) % (
        output_format,
        lammps_atom_style,
        lammps_units,
        int(starting_seed),
        int(configs_per_count),
        operation_mode,
        _clean(grid_vary),
        _clean(sub_settings or {}),
        sub_mode,
        float(sub_target),
        _clean(vac_percent or {}),
        vac_mode,
        float(vac_target),
        _clean(cluster_params),
        base_poscar,
    )

    return preamble + embedded + settings + _SCRIPT_BODY
