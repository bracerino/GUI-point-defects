import streamlit as st
import psutil
import sys

st.set_page_config(
    page_title="XRDlicious submodule: Point Defects Creation on Uploaded Crystal Structures (CIF, LMP, POSCAR, ...)",
    layout="wide"
)

st.markdown("""
    <style>
    .block-container {
        padding-top: 0rem;
    }
    </style>
""", unsafe_allow_html=True)

from helpers_defects import *

import numpy as np
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
import streamlit.components.v1 as components
import py3Dmol
from io import StringIO
import pandas as pd
import os
from pymatgen.core import Structure as PmgStructure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import io
from pymatgen.core import Structure, Element
from PIL import Image
from pymatgen.transformations.standard_transformations import SupercellTransformation

MP_API_KEY = "UtfGa1BUI3RlWYVwfpMco2jVt8ApHOye"
ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]


def display_structure_types():
    if st.checkbox("See Crystal Structure Types"):
        with st.expander("Structure Types by Space Group", expanded=True):
            for sg, types in sorted(STRUCTURE_TYPES.items()):
                sg_symbol = SPACE_GROUP_SYMBOLS.get(sg, "Unknown")
                header = f"**Space Group {sg} ({sg_symbol})**"
                line = " | ".join([f"`{formula}` → {name}" for formula, name in types.items()])
                st.markdown(f"{header}: {line}")




def classify_interstitial_site(structure, frac_coords, dummy_element="H"):
    from pymatgen.analysis.local_env import CrystalNN
    temp_struct = structure.copy()
    temp_struct.append(dummy_element, frac_coords, coords_are_cartesian=False)
    cnn = CrystalNN()
    try:
        nn_info = cnn.get_nn_info(temp_struct, len(temp_struct) - 1)
    except Exception as e:
        st.write("CrystalNN error:", e)
        nn_info = []
    cn = len(nn_info)

    if cn == 4:
        return f"CN = {cn} **(Tetrahedral)**"
    elif cn == 6:
        return f"CN = {cn} **(Octahedral)**"
    elif cn == 3:
        return f"CN = {cn} (Trigonal Planar)"
    elif cn == 5:
        return f"CN = {cn} (Trigonal Bipyramidal)"
    else:
        return f"CN = {cn}"


def wrap_coordinates(frac_coords):
    return np.array(frac_coords) % 1


def compute_periodic_distance_matrix(frac_coords):
    n = len(frac_coords)
    dist_matrix = np.zeros((n, n))
    if n > 1:
        for i in range(n):
            for j in range(i + 1, n):
                delta = frac_coords[i] - frac_coords[j]
                delta = delta - np.round(delta)
                dist = np.linalg.norm(delta)
                dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix


def select_spaced_points(frac_coords_list, n_points, mode, target_value=0.5, random_seed=None):
    if not frac_coords_list or n_points == 0:
        return [], []

    if random_seed is not None:
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)

    frac_coords_array = np.array(frac_coords_list)
    n_available = len(frac_coords_list)

    if n_available <= n_points:
        selected_indices = list(range(n_available))
        selected_coords_out = frac_coords_list.copy()
        return selected_coords_out, selected_indices

    if mode == "farthest":
        dist_matrix = compute_periodic_distance_matrix(frac_coords_array)
        selected_indices = []

        # Randomize starting point
        if random_seed is not None:
            import random
            start_idx = random.randint(0, n_available - 1)
        else:
            start_idx = 0
        selected_indices.append(start_idx)

        for _ in range(n_points - 1):
            remaining_indices = [i for i in range(n_available) if i not in selected_indices]
            if not remaining_indices:
                break

            best_idx = remaining_indices[0]
            best_min_dist = 0

            for candidate_idx in remaining_indices:
                min_dist_to_selected = min(dist_matrix[candidate_idx][sel_idx] for sel_idx in selected_indices)
                if min_dist_to_selected > best_min_dist:
                    best_min_dist = min_dist_to_selected
                    best_idx = candidate_idx

            selected_indices.append(best_idx)

    elif mode == "nearest":
        dist_matrix = compute_periodic_distance_matrix(frac_coords_array)
        selected_indices = []

        # Randomize starting point
        if random_seed is not None:
            import random
            start_idx = random.randint(0, n_available - 1)
        else:
            start_idx = 0
        selected_indices.append(start_idx)

        for _ in range(n_points - 1):
            remaining_indices = [i for i in range(n_available) if i not in selected_indices]
            if not remaining_indices:
                break

            best_idx = remaining_indices[0]
            best_min_dist = float('inf')

            for candidate_idx in remaining_indices:
                min_dist_to_selected = min(dist_matrix[candidate_idx][sel_idx] for sel_idx in selected_indices)
                if min_dist_to_selected < best_min_dist:
                    best_min_dist = min_dist_to_selected
                    best_idx = candidate_idx

            selected_indices.append(best_idx)

    elif mode == "moderate":
        dist_matrix = compute_periodic_distance_matrix(frac_coords_array)
        selected_indices = []

        # Randomize starting point
        if random_seed is not None:
            import random
            start_idx = random.randint(0, n_available - 1)
        else:
            start_idx = 0
        selected_indices.append(start_idx)

        for _ in range(n_points - 1):
            remaining_indices = [i for i in range(n_available) if i not in selected_indices]
            if not remaining_indices:
                break

            best_idx = remaining_indices[0]
            best_score = float('inf')

            # Calculate min and max possible distances for normalization
            all_min_distances = []
            for candidate_idx in remaining_indices:
                min_dist_to_selected = min(dist_matrix[candidate_idx][sel_idx] for sel_idx in selected_indices)
                all_min_distances.append(min_dist_to_selected)

            if len(all_min_distances) > 1:
                min_possible_dist = min(all_min_distances)
                max_possible_dist = max(all_min_distances)

                # Avoid division by zero
                if max_possible_dist > min_possible_dist:
                    for candidate_idx in remaining_indices:
                        min_dist_to_selected = min(dist_matrix[candidate_idx][sel_idx] for sel_idx in selected_indices)

                        # Normalize distance to 0-1 range
                        normalized_dist = (min_dist_to_selected - min_possible_dist) / (
                                    max_possible_dist - min_possible_dist)


                        score = abs(normalized_dist - target_value)

                        if score < best_score:
                            best_score = score
                            best_idx = candidate_idx
                else:
                    best_idx = remaining_indices[0]
            else:
                best_idx = remaining_indices[0]

            selected_indices.append(best_idx)

    else:
        import random
        selected_indices = random.sample(range(n_available), n_points)

    selected_coords_out = [frac_coords_list[i] for i in selected_indices]
    return selected_coords_out, selected_indices


def insert_interstitials_ase_fast(structure_obj, interstitial_element, n_interstitials,
                                  min_distance=2.0, grid_spacing=0.5, mode="random",
                                  min_interstitial_distance=1.0, log_area=None, random_seed=None):
    from pymatgen.io.ase import AseAtomsAdaptor
    from scipy.spatial.distance import cdist
    import random

    if log_area:
        log_area.info(f"Fast insertion: {n_interstitials} {interstitial_element} atoms...")
        log_area.info(f"Using grid spacing: {grid_spacing}Å, min distance: {min_distance}Å, mode: {mode}")
        log_area.info(f"Min interstitial-interstitial distance: {min_interstitial_distance}Å")
        if random_seed is not None:
            log_area.info(f"Random seed: {random_seed}")

    new_structure = structure_obj.copy()

    try:
        ase_atoms = AseAtomsAdaptor.get_atoms(structure_obj)
        cell = ase_atoms.get_cell()
        positions = ase_atoms.get_positions()

        cell_lengths = ase_atoms.get_cell_lengths_and_angles()[:3]
        n_points = [int(length / grid_spacing) + 1 for length in cell_lengths]

        if log_area:
            log_area.write(f"Creating grid: {n_points[0]}×{n_points[1]}×{n_points[2]} = {np.prod(n_points)} points")


        grid_points = []
        for i in range(n_points[0]):
            for j in range(n_points[1]):
                for k in range(n_points[2]):
                    frac_coord = np.array([i / n_points[0], j / n_points[1], k / n_points[2]])
                    grid_points.append(frac_coord)

        grid_points = np.array(grid_points)
        grid_cart = np.dot(grid_points, cell.array)


        distances = cdist(grid_cart, positions)
        min_distances_to_atoms = np.min(distances, axis=1)

        valid_indices = np.where(min_distances_to_atoms >= min_distance)[0]
        valid_points = grid_points[valid_indices]
        valid_points_cart = grid_cart[valid_indices]

        if log_area:
            log_area.write(f"Found {len(valid_points)} valid void sites (>{min_distance}Å from atoms)")

        if len(valid_points) == 0:
            if log_area: log_area.warning("No valid interstitial sites found with current parameters")
            return new_structure


        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        selected_points = []
        selected_points_cart = []

        n_to_insert = min(n_interstitials, len(valid_points))
        if n_to_insert < n_interstitials and log_area:
            log_area.warning(f"Only {n_to_insert} sites available, requested {n_interstitials}")

        if mode == "random":
            attempts = 0
            max_attempts = len(valid_points) * 10

            while len(selected_points) < n_to_insert and attempts < max_attempts:
                attempts += 1
                candidate_idx = random.randint(0, len(valid_points) - 1)
                candidate_point = valid_points[candidate_idx]
                candidate_point_cart = valid_points_cart[candidate_idx]

                if len(selected_points_cart) == 0:
                    selected_points.append(candidate_point)
                    selected_points_cart.append(candidate_point_cart)
                else:
                    distances_to_selected = cdist([candidate_point_cart], selected_points_cart)[0]
                    min_dist_to_selected = np.min(distances_to_selected)

                    if min_dist_to_selected >= min_interstitial_distance:
                        selected_points.append(candidate_point)
                        selected_points_cart.append(candidate_point_cart)

            if len(selected_points) < n_to_insert and log_area:
                log_area.warning(
                    f"Only found {len(selected_points)} sites with min interstitial distance {min_interstitial_distance}Å")

        else:
            valid_points_list = [valid_points[i] for i in range(len(valid_points))]
            selected_points_temp, _ = select_spaced_points(valid_points_list, n_to_insert, mode, 0.5, random_seed)

            selected_points = []
            selected_points_cart = []

            for point in selected_points_temp:
                point_cart = np.dot(point, cell.array)

                if len(selected_points_cart) == 0:
                    selected_points.append(point)
                    selected_points_cart.append(point_cart)
                else:
                    distances_to_selected = cdist([point_cart], selected_points_cart)[0]
                    min_dist_to_selected = np.min(distances_to_selected)

                    if min_dist_to_selected >= min_interstitial_distance:
                        selected_points.append(point)
                        selected_points_cart.append(point_cart)

        for point in selected_points:
            new_structure.append(
                species=Element(interstitial_element),
                coords=point,
                coords_are_cartesian=False,
                validate_proximity=False
            )

        if log_area:
            log_area.success(
                f"Successfully inserted {len(selected_points)} {interstitial_element} atoms using {mode} selection")

    except Exception as e:
        if log_area: log_area.error(f"Error in fast interstitial insertion: {e}")
        return structure_obj

    return new_structure


def insert_interstitials_into_structure(structure_obj, interstitial_element, n_interstitials,
                                        which_interstitial_type_idx=0, mode="farthest",
                                        clustering_tol_val=0.75, min_dist_val=0.5, target_value=0.5, log_area=None,
                                        random_seed=None):
    from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator

    if log_area: log_area.info(f"Attempting to insert {n_interstitials} of {interstitial_element}...")
    new_structure_int = structure_obj.copy()
    try:
        generator = VoronoiInterstitialGenerator(clustering_tol=clustering_tol_val, min_dist=min_dist_val)
        unique_interstitial_types = list(generator.generate(new_structure_int, "H"))
        if not unique_interstitial_types:
            if log_area: log_area.warning("VoronoiInterstitialGenerator found no candidate sites.")
            return new_structure_int
        if log_area:
            log_area.write(f"Found {len(unique_interstitial_types)} unique interstitial site types.")
            for i_type, interstitial_type_obj in enumerate(unique_interstitial_types):
                site_label = classify_interstitial_site(new_structure_int, interstitial_type_obj.site.frac_coords)
                log_area.write(
                    f"  Type {i_type + 1}: at {np.round(interstitial_type_obj.site.frac_coords, 3)}, {site_label}, with {len(interstitial_type_obj.equivalent_sites)} equivalent sites.")

        frac_coords_to_consider = []
        if which_interstitial_type_idx == 0:
            for interstitial_type_obj in unique_interstitial_types:
                for eq_site in interstitial_type_obj.equivalent_sites:
                    frac_coords_to_consider.append(eq_site.frac_coords)
        elif 0 < which_interstitial_type_idx <= len(unique_interstitial_types):
            interstitial_type_obj = unique_interstitial_types[which_interstitial_type_idx - 1]
            for eq_site in interstitial_type_obj.equivalent_sites:
                frac_coords_to_consider.append(eq_site.frac_coords)
            if log_area: log_area.info(
                f"Focusing on interstitial type {which_interstitial_type_idx} ({len(frac_coords_to_consider)} sites).")
        else:
            if log_area: log_area.warning(
                f"Invalid interstitial type index: {which_interstitial_type_idx}. Max is {len(unique_interstitial_types)}. Using no sites.")
            return new_structure_int

        if not frac_coords_to_consider:
            if log_area: log_area.warning("No interstitial sites available to select from based on criteria.")
            return new_structure_int

        num_to_actually_insert = min(n_interstitials, len(frac_coords_to_consider))
        if num_to_actually_insert < n_interstitials and log_area:
            log_area.warning(
                f"Requested {n_interstitials} interstitials, but only {num_to_actually_insert} sites are available. Inserting {num_to_actually_insert}.")

        if num_to_actually_insert > 0:
            selected_points_coords, _ = select_spaced_points(frac_coords_to_consider, num_to_actually_insert, mode,
                                                             target_value, random_seed)

            if log_area: log_area.info(
                f"Selected {len(selected_points_coords)} sites out of {len(frac_coords_to_consider)} available sites.")

            for point_coords in selected_points_coords:
                new_structure_int.append(
                    species=Element(interstitial_element),
                    coords=point_coords,
                    coords_are_cartesian=False,
                    validate_proximity=True
                )
            if log_area: log_area.info(
                f"Successfully inserted {len(selected_points_coords)} {interstitial_element} atoms.")
        else:
            if log_area: log_area.info("No interstitials were inserted based on parameters.")
    except Exception as e_int:
        if log_area: log_area.error(f"Error during interstitial insertion: {e_int}")
    return new_structure_int


def remove_vacancies_from_structure(structure_obj, vacancy_percentages_dict, selection_mode_vac="farthest",
                                    target_value_vac=0.5, log_area=None, random_seed=None):
    if random_seed is not None:
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)

    if log_area: log_area.info(f"Attempting to create vacancies...")
    new_structure_vac = structure_obj.copy()
    indices_to_remove_overall = []
    for el_symbol, perc_to_remove in vacancy_percentages_dict.items():
        if perc_to_remove <= 0: continue
        el_indices_in_struct = [i for i, site in enumerate(new_structure_vac.sites) if
                                site.specie and site.specie.symbol == el_symbol]
        n_sites_of_el = len(el_indices_in_struct)
        n_to_remove_for_el = int(round(n_sites_of_el * perc_to_remove / 100.0))
        if log_area: log_area.write(
            f"  For element {el_symbol}: Found {n_sites_of_el} sites. Requested to remove {perc_to_remove}% ({n_to_remove_for_el} atoms).")
        if n_to_remove_for_el == 0: continue
        if n_to_remove_for_el > n_sites_of_el:
            if log_area: log_area.warning(
                f"    Cannot remove {n_to_remove_for_el} atoms of {el_symbol}, only {n_sites_of_el} exist. Removing all.")
            n_to_remove_for_el = n_sites_of_el
        el_frac_coords = [new_structure_vac.sites[i].frac_coords for i in el_indices_in_struct]
        if not el_frac_coords and n_to_remove_for_el > 0:
            if log_area: log_area.warning(f"    No coordinates found for {el_symbol} to select from.")
            continue
        _, selected_local_indices_for_removal = select_spaced_points(el_frac_coords, n_to_remove_for_el,
                                                                     selection_mode_vac, target_value_vac, random_seed)
        global_indices_for_this_el_removal = [el_indices_in_struct[i] for i in selected_local_indices_for_removal]
        indices_to_remove_overall.extend(global_indices_for_this_el_removal)
        if log_area: log_area.write(
            f"    Selected {len(global_indices_for_this_el_removal)} sites of {el_symbol} for removal.")
    if indices_to_remove_overall:
        unique_indices_to_remove = sorted(list(set(indices_to_remove_overall)), reverse=True)
        new_structure_vac.remove_sites(unique_indices_to_remove)
        if log_area: log_area.info(f"Attempted to remove {len(unique_indices_to_remove)} atoms in total.")
    else:
        if log_area: log_area.info("No atoms were selected for vacancy creation.")
    return new_structure_vac


def substitute_atoms_in_structure(structure_obj, substitution_settings_dict, selection_mode_sub="farthest",
                                  target_value_sub=0.5, log_area=None, random_seed=None):
    if random_seed is not None:
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)

    if log_area: log_area.info(f"Attempting substitutions...")
    new_species_list = [site.species for site in structure_obj.sites]
    new_coords_list = [site.frac_coords for site in structure_obj.sites]
    modified_indices_count = 0
    for orig_el_symbol, settings in substitution_settings_dict.items():
        perc_to_sub = settings.get("percentage", 0)
        sub_el_symbol = settings.get("substitute", "").strip()
        if perc_to_sub <= 0 or not sub_el_symbol: continue
        try:
            sub_element = Element(sub_el_symbol)
        except Exception:
            if log_area: log_area.warning(
                f"  Invalid substitute element symbol: '{sub_el_symbol}'. Skipping for {orig_el_symbol}.")
            continue
        orig_el_indices_in_struct = [i for i, site in enumerate(structure_obj.sites) if
                                     site.specie and site.specie.symbol == orig_el_symbol]
        n_sites_of_orig_el = len(orig_el_indices_in_struct)
        n_to_sub_for_el = int(round(n_sites_of_orig_el * perc_to_sub / 100.0))
        if log_area: log_area.write(
            f"  For {orig_el_symbol} -> {sub_el_symbol}: Found {n_sites_of_orig_el} sites of {orig_el_symbol}. Requested to substitute {perc_to_sub}% ({n_to_sub_for_el} atoms).")
        if n_to_sub_for_el == 0: continue
        if n_to_sub_for_el > n_sites_of_orig_el:
            if log_area: log_area.warning(
                f"    Cannot substitute {n_to_sub_for_el} atoms of {orig_el_symbol}, only {n_sites_of_orig_el} exist. Substituting all.")
            n_to_sub_for_el = n_sites_of_orig_el
        orig_el_frac_coords = [structure_obj.sites[i].frac_coords for i in orig_el_indices_in_struct]
        if not orig_el_frac_coords and n_to_sub_for_el > 0:
            if log_area: log_area.warning(f"    No coordinates found for {orig_el_symbol} to select from.")
            continue
        _, selected_local_indices_for_substitution = select_spaced_points(orig_el_frac_coords, n_to_sub_for_el,
                                                                          selection_mode_sub, target_value_sub,
                                                                          random_seed)
        global_indices_for_this_el_substitution = [orig_el_indices_in_struct[i] for i in
                                                   selected_local_indices_for_substitution]
        for global_idx_to_sub in global_indices_for_this_el_substitution:
            new_species_list[global_idx_to_sub] = sub_element
            modified_indices_count += 1
        if log_area: log_area.write(
            f"    Selected {len(global_indices_for_this_el_substitution)} sites of {orig_el_symbol} for substitution with {sub_el_symbol}.")
    if modified_indices_count > 0:
        final_substituted_structure = Structure(lattice=structure_obj.lattice, species=new_species_list,
                                                coords=new_coords_list, coords_are_cartesian=False)
        if log_area: log_area.info(f"Attempted to substitute {modified_indices_count} atoms in total.")
        return final_substituted_structure
    else:
        if log_area: log_area.info("No atoms were selected for substitution.")
        return structure_obj.copy()


def get_orthogonal_cell(structure, max_atoms=200):
    from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation

    from pymatgen.io.ase import AseAtomsAdaptor
    ase_atoms = AseAtomsAdaptor.get_atoms(structure)
    angles = ase_atoms.get_cell_lengths_and_angles()[3:]
    if all(abs(angle - 90.0) < 1e-6 for angle in angles):
        return structure.copy()

    try:
        transformer = CubicSupercellTransformation(
            max_atoms=max_atoms,
            min_atoms=len(structure),
            force_90_degrees=True,
            allow_orthorhombic=True,
            angle_tolerance=0.1,
            min_length=5.0
        )

        orthogonal_structure = transformer.apply_transformation(structure)
        return orthogonal_structure

    except Exception as e:
        try:
            from pymatgen.transformations.standard_transformations import SupercellTransformation

            supercell_matrices = [
                [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                [[1, -1, 0], [1, 1, 0], [0, 0, 1]],
                [[2, -1, 0], [1, 1, 0], [0, 0, 1]]
            ]

            for matrix in supercell_matrices:
                try:
                    sc_transformer = SupercellTransformation(matrix)
                    test_structure = sc_transformer.apply_transformation(structure)

                    ase_test = AseAtomsAdaptor.get_atoms(test_structure)
                    test_angles = ase_test.get_cell_lengths_and_angles()[3:]

                    if all(abs(angle - 90.0) < 5.0 for angle in test_angles):
                        return test_structure

                except Exception:
                    continue

        except Exception:
            pass

        return structure.copy()


def get_structure_info(structure):
    if structure is None:
        return "No structure", ""

    atom_count = len(structure)
    ase_atoms = AseAtomsAdaptor.get_atoms(structure)
    cell_params = ase_atoms.get_cell_lengths_and_angles()
    volume = ase_atoms.get_volume()

    info_text = f"**{atom_count} atoms**\n"

    element_counts = {}
    for site in structure:
        element = site.specie.symbol
        element_counts[element] = element_counts.get(element, 0) + 1

    element_info = ", ".join([f"{elem}: {count}" for elem, count in sorted(element_counts.items())])
    info_text += f"**Elements:** {element_info}\n"

    info_text += f"a={cell_params[0]:.3f}Å, b={cell_params[1]:.3f}Å, c={cell_params[2]:.3f}Å\n"
    info_text += f"α={cell_params[3]:.1f}°, β={cell_params[4]:.1f}°, γ={cell_params[5]:.1f}°\n"
    info_text += f"Vol={volume:.2f}Å³"

    return info_text


st.markdown("""
    <style>
    div.stButton > button {
        background-color: #0099ff; color: white; font-size: 16px; font-weight: bold;
        padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
    }
    div.stButton > button:active, div.stButton > button:focus {
        background-color: #007acc !important; color: white !important; box-shadow: none !important;
    }
    div[data-testid="stDataFrameContainer"] table td { font-size: 16px !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

components.html("""
    <head><meta name="description" content="XRDlicious submodule: Point Defects Creation on Uploaded Crystal Structures (CIF, LMP, POSCAR, ...)"></head>
    """, height=0)

st.markdown("#### XRDlicious submodule:  Point Defects Creation on Uploaded Crystal Structures (CIF, LMP, POSCAR, ...)")
col1_header, col2_header = st.columns([1.25, 1])
with col2_header:
    st.info(
        "🌀 Developed by [IMPLANT team](https://implant.fs.cvut.cz/). 📺 [Quick tutorial HERE.](https://www.youtube.com/watch?v=7ZgQ0fnR8dQ&ab_channel=Implantgroup)")
with col1_header:
    st.info("Visit the main [XRDlicious](http://xrdlicious.com) page")


show_database_search = st.checkbox("Enable database search",
                                   value=False,
                                   help="Enable to search in Materials Project, AFLOW, and COD databases")


def get_space_group_info(number):
    symbol = SPACE_GROUP_SYMBOLS.get(number, f"SG#{number}")
    return symbol


if show_database_search:
    with st.expander("Search for Structures Online in Databases", icon="🔍", expanded=True):
        cols, cols2, cols3 = st.columns([1.5, 1.5, 3.5])
        with cols:
            db_choices = st.multiselect(
                "Select Database(s)",
                options=["Materials Project", "AFLOW", "COD"],
                default=["Materials Project", "AFLOW", "COD"],
                help="Choose which databases to search for structures. You can select multiple databases."
            )

            if not db_choices:
                st.warning("Please select at least one database to search.")

            st.markdown("**Maximum number of structures to be found in each database (for improving performance):**")
            col_limits = st.columns(3)

            search_limits = {}
            if "Materials Project" in db_choices:
                with col_limits[0]:
                    search_limits["Materials Project"] = st.number_input(
                        "MP Limit:", min_value=1, max_value=2000, value=300, step=10,
                        help="Maximum results from Materials Project"
                    )
            if "AFLOW" in db_choices:
                with col_limits[1]:
                    search_limits["AFLOW"] = st.number_input(
                        "AFLOW Limit:", min_value=1, max_value=2000, value=300, step=10,
                        help="Maximum results from AFLOW"
                    )
            if "COD" in db_choices:
                with col_limits[2]:
                    search_limits["COD"] = st.number_input(
                        "COD Limit:", min_value=1, max_value=2000, value=300, step=10,
                        help="Maximum results from COD"
                    )

        with cols2:
            search_mode = st.radio(
                "Search by:",
                options=["Elements", "Structure ID", "Space Group + Elements", "Formula", "Search Mineral"],
                help="Choose your search strategy"
            )

            if search_mode == "Elements":
                selected_elements = st.multiselect(
                    "Select elements for search:",
                    options=ELEMENTS,
                    default=["Sr", "Ti", "O"],
                    help="Choose one or more chemical elements"
                )
                search_query = " ".join(selected_elements) if selected_elements else ""

            elif search_mode == "Structure ID":
                structure_ids = st.text_area(
                    "Enter Structure IDs (one per line):",
                    value="mp-5229\ncod_1512124\naflow:010158cb2b41a1a5",
                    help="Enter structure IDs. Examples:\n- Materials Project: mp-5229\n- COD: cod_1512124 (with cod_ prefix)\n- AFLOW: aflow:010158cb2b41a1a5 (AUID format)"
                )

            elif search_mode == "Space Group + Elements":
                col_sg1, col_sg2 = st.columns(2)
                with col_sg1:
                    all_space_groups_help = "Enter space group number (1-230)\n\nAll space groups:\n\n"
                    for num in sorted(SPACE_GROUP_SYMBOLS.keys()):
                        all_space_groups_help += f"• {num}: {SPACE_GROUP_SYMBOLS[num]}\n\n"

                    space_group_number = st.number_input(
                        "Space Group Number:",
                        min_value=1,
                        max_value=230,
                        value=221,
                        help=all_space_groups_help
                    )
                    sg_symbol = get_space_group_info(space_group_number)
                    st.info(f"#:**{sg_symbol}**")

                selected_elements = st.multiselect(
                    "Select elements for search:",
                    options=ELEMENTS,
                    default=["Sr", "Ti", "O"],
                    help="Choose one or more chemical elements"
                )

            elif search_mode == "Formula":
                formula_input = st.text_input(
                    "Enter Chemical Formula:",
                    value="Sr Ti O3",
                    help="Enter chemical formula with spaces between elements. Examples:\n- Sr Ti O3 (strontium titanate)\n- Ca C O3 (calcium carbonate)\n- Al2 O3 (alumina)"
                )

            elif search_mode == "Search Mineral":
                mineral_options = []
                mineral_mapping = {}

                for space_group, minerals in MINERALS.items():
                    for mineral_name, formula in minerals.items():
                        option_text = f"{mineral_name} - SG #{space_group}"
                        mineral_options.append(option_text)
                        mineral_mapping[option_text] = {
                            'space_group': space_group,
                            'formula': formula,
                            'mineral_name': mineral_name
                        }
                mineral_options.sort()

                selected_mineral = st.selectbox(
                    "Select Mineral Structure:",
                    options=mineral_options,
                    help="Choose a mineral structure type. The exact formula and space group will be automatically set.",
                    index=2
                )

                if selected_mineral:
                    mineral_info = mineral_mapping[selected_mineral]

                    sg_symbol = get_space_group_info(mineral_info['space_group'])
                    st.info(f"**Structure:** {mineral_info['mineral_name']}, **Space Group:** {mineral_info['space_group']} ({sg_symbol}), "
                            f"**Formula:** {mineral_info['formula']}")


                    space_group_number = mineral_info['space_group']
                    formula_input = mineral_info['formula']

                    st.success(f"**Search will use:** Formula = {formula_input}, Space Group = {space_group_number}")

            show_element_info = st.checkbox("ℹ️ Show information about element groups")
            if show_element_info:
                st.markdown("""
                **Element groups note:**
                **Common Elements (14):** H, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca  
                **Transition Metals (10):** Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn  
                **Alkali Metals (6):** Li, Na, K, Rb, Cs, Fr  
                **Alkaline Earth (6):** Be, Mg, Ca, Sr, Ba, Ra  
                **Noble Gases (6):** He, Ne, Ar, Kr, Xe, Rn  
                **Halogens (5):** F, Cl, Br, I, At  
                **Lanthanides (15):** La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu  
                **Actinides (15):** Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr  
                **Other Elements (51):** All remaining elements
                """)

        if st.button("Search Selected Databases"):
            if not db_choices:
                st.error("Please select at least one database to search.")
            else:
                for db_choice in db_choices:
                    if db_choice == "Materials Project":
                        mp_limit = search_limits.get("Materials Project", 50)
                        with st.spinner(f"Searching **the MP database** (limit: {mp_limit}), please wait. 😊"):
                            try:
                                with MPRester(MP_API_KEY) as mpr:
                                    docs = None

                                    if search_mode == "Elements":
                                        elements_list = [el.strip() for el in search_query.split() if el.strip()]
                                        if not elements_list:
                                            st.error("Please enter at least one element for the search.")
                                            continue
                                        elements_list_sorted = sorted(set(elements_list))
                                        docs = mpr.materials.summary.search(
                                            elements=elements_list_sorted,
                                            num_elements=len(elements_list_sorted),
                                            fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                        )

                                    elif search_mode == "Structure ID":
                                        mp_ids = [id.strip() for id in structure_ids.split('\n')
                                                  if id.strip() and id.strip().startswith('mp-')]
                                        if not mp_ids:
                                            st.warning("No valid Materials Project IDs found (should start with 'mp-')")
                                            continue
                                        docs = mpr.materials.summary.search(
                                            material_ids=mp_ids,
                                            fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                        )

                                    elif search_mode == "Space Group + Elements":
                                        elements_list = sorted(set(selected_elements))
                                        if not elements_list:
                                            st.warning(
                                                "Please select elements for Materials Project space group search.")
                                            continue

                                        search_params = {
                                            "elements": elements_list,
                                            "num_elements": len(elements_list),
                                            "fields": ["material_id", "formula_pretty", "symmetry", "nsites", "volume"],
                                            "spacegroup_number": space_group_number
                                        }

                                        docs = mpr.materials.summary.search(**search_params)

                                    elif search_mode == "Formula":
                                        if not formula_input.strip():
                                            st.warning("Please enter a chemical formula for Materials Project search.")
                                            continue

                                        # Convert space-separated format to compact format (Sr Ti O3 -> SrTiO3)
                                        clean_formula = formula_input.strip()
                                        if ' ' in clean_formula:
                                            parts = clean_formula.split()
                                            compact_formula = ''.join(parts)
                                        else:
                                            compact_formula = clean_formula

                                        docs = mpr.materials.summary.search(
                                            formula=compact_formula,
                                            fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                        )

                                    elif search_mode == "Search Mineral":
                                        if not selected_mineral:
                                            st.warning(
                                                "Please select a mineral structure for Materials Project search.")
                                            continue
                                        clean_formula = formula_input.strip()
                                        if ' ' in clean_formula:
                                            parts = clean_formula.split()
                                            compact_formula = ''.join(parts)
                                        else:
                                            compact_formula = clean_formula

                                        # Search by formula and space group
                                        docs = mpr.materials.summary.search(
                                            formula=compact_formula,
                                            spacegroup_number=space_group_number,
                                            fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                        )

                                    if docs:
                                        status_placeholder = st.empty()
                                        st.session_state.mp_options = []
                                        st.session_state.full_structures_see = {}
                                        limited_docs = docs[:mp_limit]

                                        for doc in limited_docs:
                                            full_structure = mpr.get_structure_by_material_id(doc.material_id,
                                                                                              conventional_unit_cell=True)
                                            st.session_state.full_structures_see[doc.material_id] = full_structure
                                            lattice = full_structure.lattice
                                            leng = len(full_structure)
                                            lattice_str = (f"{lattice.a:.3f} {lattice.b:.3f} {lattice.c:.3f} Å, "
                                                           f"{lattice.alpha:.1f}, {lattice.beta:.1f}, {lattice.gamma:.1f} °")
                                            st.session_state.mp_options.append(
                                                f"{doc.material_id}: {doc.formula_pretty} ({doc.symmetry.symbol} #{doc.symmetry.number}) [{lattice_str}], {float(doc.volume):.1f} Å³, {leng} atoms"
                                            )
                                            status_placeholder.markdown(
                                                f"- **Structure loaded:** `{full_structure.composition.reduced_formula}` ({doc.material_id})"
                                            )
                                        if len(limited_docs) < len(docs):
                                            st.info(
                                                f"Showing first {mp_limit} of {len(docs)} total Materials Project results. Increase limit to see more.")
                                        st.success(
                                            f"Found {len(st.session_state.mp_options)} structures in Materials Project.")
                                    else:
                                        st.session_state.mp_options = []
                                        st.warning("No matching structures found in Materials Project.")
                            except Exception as e:
                                st.error(f"An error occurred with Materials Project: {e}")

                    elif db_choice == "AFLOW":
                        aflow_limit = search_limits.get("AFLOW", 50)
                        with st.spinner(f"Searching **the AFLOW database** (limit: {aflow_limit}), please wait. 😊"):
                            try:
                                results = []

                                if search_mode == "Elements":
                                    elements_list = [el.strip() for el in search_query.split() if el.strip()]
                                    if not elements_list:
                                        st.warning("Please enter elements for AFLOW search.")
                                        continue
                                    ordered_elements = sorted(elements_list)
                                    ordered_str = ",".join(ordered_elements)
                                    aflow_nspecies = len(ordered_elements)

                                    results = list(
                                        search(catalog="icsd")
                                        .filter((AFLOW_K.species % ordered_str) & (AFLOW_K.nspecies == aflow_nspecies))
                                        .select(
                                            AFLOW_K.auid,
                                            AFLOW_K.compound,
                                            AFLOW_K.geometry,
                                            AFLOW_K.spacegroup_relax,
                                            AFLOW_K.aurl,
                                            AFLOW_K.files,
                                        )
                                    )

                                elif search_mode == "Structure ID":
                                    aflow_auids = []
                                    for id_line in structure_ids.split('\n'):
                                        id_line = id_line.strip()
                                        if id_line.startswith('aflow:'):
                                            auid = id_line.replace('aflow:', '').strip()
                                            aflow_auids.append(auid)

                                    if not aflow_auids:
                                        st.warning("No valid AFLOW AUIDs found (should start with 'aflow:')")
                                        continue

                                    results = []
                                    for auid in aflow_auids:
                                        try:
                                            result = list(search(catalog="icsd")
                                                          .filter(AFLOW_K.auid == f"aflow:{auid}")
                                                          .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                  AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                  AFLOW_K.files))
                                            results.extend(result)
                                        except Exception as e:
                                            st.warning(f"AFLOW search failed for AUID '{auid}': {e}")
                                            continue

                                elif search_mode == "Space Group + Elements":
                                    if not selected_elements:
                                        st.warning("Please select elements for AFLOW space group search.")
                                        continue
                                    ordered_elements = sorted(selected_elements)
                                    ordered_str = ",".join(ordered_elements)
                                    aflow_nspecies = len(ordered_elements)

                                    try:
                                        results = list(search(catalog="icsd")
                                                       .filter((AFLOW_K.species % ordered_str) &
                                                               (AFLOW_K.nspecies == aflow_nspecies) &
                                                               (AFLOW_K.spacegroup_relax == space_group_number))
                                                       .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                               AFLOW_K.spacegroup_relax, AFLOW_K.aurl, AFLOW_K.files))
                                    except Exception as e:
                                        st.warning(f"AFLOW space group search failed: {e}")
                                        results = []


                                elif search_mode == "Formula":

                                    if not formula_input.strip():
                                        st.warning("Please enter a chemical formula for AFLOW search.")

                                        continue


                                    def convert_to_aflow_formula(formula_input):

                                        import re

                                        formula_parts = formula_input.strip().split()

                                        elements_dict = {}

                                        for part in formula_parts:

                                            match = re.match(r'([A-Z][a-z]?)(\d*)', part)

                                            if match:
                                                element = match.group(1)

                                                count = match.group(2) if match.group(
                                                    2) else "1"  # Add "1" if no number

                                                elements_dict[element] = count

                                        aflow_parts = []

                                        for element in sorted(elements_dict.keys()):
                                            aflow_parts.append(f"{element}{elements_dict[element]}")

                                        return "".join(aflow_parts)


                                    # Generate 2x multiplied formula
                                    def multiply_formula_by_2(formula_input):

                                        import re

                                        formula_parts = formula_input.strip().split()

                                        elements_dict = {}

                                        for part in formula_parts:

                                            match = re.match(r'([A-Z][a-z]?)(\d*)', part)

                                            if match:
                                                element = match.group(1)

                                                count = int(match.group(2)) if match.group(2) else 1

                                                elements_dict[element] = str(count * 2)  # Multiply by 2

                                        aflow_parts = []

                                        for element in sorted(elements_dict.keys()):
                                            aflow_parts.append(f"{element}{elements_dict[element]}")

                                        return "".join(aflow_parts)


                                    aflow_formula = convert_to_aflow_formula(formula_input)

                                    aflow_formula_2x = multiply_formula_by_2(formula_input)

                                    if aflow_formula_2x != aflow_formula:

                                        results = list(search(catalog="icsd")

                                                       .filter((AFLOW_K.compound == aflow_formula) |

                                                               (AFLOW_K.compound == aflow_formula_2x))

                                                       .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,

                                                               AFLOW_K.spacegroup_relax, AFLOW_K.aurl, AFLOW_K.files))

                                        st.info(
                                            f"Searching for both {aflow_formula} and {aflow_formula_2x} formulas simultaneously")

                                    else:
                                        results = list(search(catalog="icsd")
                                                       .filter(AFLOW_K.compound == aflow_formula)
                                                       .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                               AFLOW_K.spacegroup_relax, AFLOW_K.aurl, AFLOW_K.files))

                                        st.info(f"Searching for formula {aflow_formula}")


                                elif search_mode == "Search Mineral":
                                    if not selected_mineral:
                                        st.warning("Please select a mineral structure for AFLOW search.")
                                        continue


                                    def convert_to_aflow_formula_mineral(formula_input):
                                        import re
                                        formula_parts = formula_input.strip().split()
                                        elements_dict = {}
                                        for part in formula_parts:

                                            match = re.match(r'([A-Z][a-z]?)(\d*)', part)
                                            if match:
                                                element = match.group(1)

                                                count = match.group(2) if match.group(
                                                    2) else "1"  # Always add "1" for single atoms

                                                elements_dict[element] = count

                                        aflow_parts = []

                                        for element in sorted(elements_dict.keys()):
                                            aflow_parts.append(f"{element}{elements_dict[element]}")

                                        return "".join(aflow_parts)


                                    def multiply_mineral_formula_by_2(formula_input):

                                        import re

                                        formula_parts = formula_input.strip().split()

                                        elements_dict = {}

                                        for part in formula_parts:
                                            match = re.match(r'([A-Z][a-z]?)(\d*)', part)
                                            if match:
                                                element = match.group(1)
                                                count = int(match.group(2)) if match.group(2) else 1
                                                elements_dict[element] = str(count * 2)  # Multiply by 2
                                        aflow_parts = []
                                        for element in sorted(elements_dict.keys()):
                                            aflow_parts.append(f"{element}{elements_dict[element]}")
                                        return "".join(aflow_parts)


                                    aflow_formula = convert_to_aflow_formula_mineral(formula_input)

                                    aflow_formula_2x = multiply_mineral_formula_by_2(formula_input)

                                    # Search for both formulas with space group constraint in a single query

                                    if aflow_formula_2x != aflow_formula:
                                        results = list(search(catalog="icsd")
                                                       .filter(((AFLOW_K.compound == aflow_formula) |
                                                                (AFLOW_K.compound == aflow_formula_2x)) &
                                                               (AFLOW_K.spacegroup_relax == space_group_number))
                                                       .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                               AFLOW_K.spacegroup_relax, AFLOW_K.aurl, AFLOW_K.files))

                                        st.info(
                                            f"Searching {mineral_info['mineral_name']} for both {aflow_formula} and {aflow_formula_2x} with space group {space_group_number}")

                                    else:
                                        results = list(search(catalog="icsd")
                                                       .filter((AFLOW_K.compound == aflow_formula) &
                                                               (AFLOW_K.spacegroup_relax == space_group_number))
                                                       .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                               AFLOW_K.spacegroup_relax, AFLOW_K.aurl, AFLOW_K.files))

                                        st.info(
                                            f"Searching {mineral_info['mineral_name']} for formula {aflow_formula} with space group {space_group_number}")

                                if results:
                                    status_placeholder = st.empty()
                                    st.session_state.aflow_options = []
                                    st.session_state.entrys = {}

                                    limited_results = results[:aflow_limit]

                                    for entry in limited_results:
                                        st.session_state.entrys[entry.auid] = entry
                                        st.session_state.aflow_options.append(
                                            f"{entry.auid}: {entry.compound} ({entry.spacegroup_relax}) {entry.geometry}"
                                        )
                                        status_placeholder.markdown(
                                            f"- **Structure loaded:** `{entry.compound}` (aflow_{entry.auid})"
                                        )
                                    if len(limited_results) < len(results):
                                        st.info(
                                            f"Showing first {aflow_limit} of {len(results)} total AFLOW results. Increase limit to see more.")
                                    st.success(f"Found {len(st.session_state.aflow_options)} structures in AFLOW.")
                                else:
                                    st.session_state.aflow_options = []
                                    st.warning("No matching structures found in AFLOW.")
                            except Exception as e:
                                st.warning(f"No matching structures found in AFLOW.")
                                st.session_state.aflow_options = []

                    elif db_choice == "COD":
                        cod_limit = search_limits.get("COD", 50)
                        with st.spinner(f"Searching **the COD database** (limit: {cod_limit}), please wait. 😊"):
                            try:
                                cod_entries = []

                                if search_mode == "Elements":
                                    elements = [el.strip() for el in search_query.split() if el.strip()]
                                    if elements:
                                        params = {'format': 'json', 'detail': '1'}
                                        for i, el in enumerate(elements, start=1):
                                            params[f'el{i}'] = el
                                        params['strictmin'] = str(len(elements))
                                        params['strictmax'] = str(len(elements))
                                        cod_entries = get_cod_entries(params)
                                    else:
                                        st.warning("Please enter elements for COD search.")
                                        continue

                                elif search_mode == "Structure ID":
                                    cod_ids = []
                                    for id_line in structure_ids.split('\n'):
                                        id_line = id_line.strip()
                                        if id_line.startswith('cod_'):
                                            # Extract numeric ID from cod_XXXXX format
                                            numeric_id = id_line.replace('cod_', '').strip()
                                            if numeric_id.isdigit():
                                                cod_ids.append(numeric_id)

                                    if not cod_ids:
                                        st.warning(
                                            "No valid COD IDs found (should start with 'cod_' followed by numbers)")
                                        continue

                                    cod_entries = []
                                    for cod_id in cod_ids:
                                        try:
                                            params = {'format': 'json', 'detail': '1', 'id': cod_id}
                                            entry = get_cod_entries(params)
                                            if entry:
                                                if isinstance(entry, list):
                                                    cod_entries.extend(entry)
                                                else:
                                                    cod_entries.append(entry)
                                        except Exception as e:
                                            st.warning(f"COD search failed for ID {cod_id}: {e}")
                                            continue

                                elif search_mode == "Space Group + Elements":
                                    elements = selected_elements
                                    if elements:
                                        params = {'format': 'json', 'detail': '1'}
                                        for i, el in enumerate(elements, start=1):
                                            params[f'el{i}'] = el
                                        params['strictmin'] = str(len(elements))
                                        params['strictmax'] = str(len(elements))
                                        params['space_group_number'] = str(space_group_number)

                                        cod_entries = get_cod_entries(params)
                                    else:
                                        st.warning("Please select elements for COD space group search.")
                                        continue

                                elif search_mode == "Formula":
                                    if not formula_input.strip():
                                        st.warning("Please enter a chemical formula for COD search.")
                                        continue

                                    # alphabet sorting
                                    alphabet_form = sort_formula_alphabetically(formula_input)
                                    print(alphabet_form)
                                    params = {'format': 'json', 'detail': '1', 'formula': alphabet_form}
                                    cod_entries = get_cod_entries(params)

                                elif search_mode == "Search Mineral":
                                    if not selected_mineral:
                                        st.warning("Please select a mineral structure for COD search.")
                                        continue

                                    # Use both formula and space group for COD search
                                    alphabet_form = sort_formula_alphabetically(formula_input)
                                    params = {
                                        'format': 'json',
                                        'detail': '1',
                                        'formula': alphabet_form,
                                        'space_group_number': str(space_group_number)
                                    }
                                    cod_entries = get_cod_entries(params)

                                if cod_entries and isinstance(cod_entries, list):
                                    status_placeholder = st.empty()
                                    st.session_state.cod_options = []
                                    st.session_state.full_structures_see_cod = {}

                                    limited_entries = cod_entries[:cod_limit]

                                    for entry in limited_entries:
                                        try:
                                            cif_content = get_cif_from_cod(entry)
                                            if cif_content:
                                                structure = get_cod_str(cif_content)
                                                cod_id = f"cod_{entry.get('file')}"
                                                st.session_state.full_structures_see_cod[cod_id] = structure
                                                spcs = entry.get("sg", "Unknown")
                                                spcs_number = entry.get("sgNumber", "Unknown")

                                                cell_volume = structure.lattice.volume
                                                st.session_state.cod_options.append(
                                                    f"{cod_id}: {structure.composition.reduced_formula} ({spcs} #{spcs_number}) [{structure.lattice.a:.3f} {structure.lattice.b:.3f} {structure.lattice.c:.3f} Å, {structure.lattice.alpha:.2f} "
                                                    f"{structure.lattice.beta:.2f} {structure.lattice.gamma:.2f}] °, {cell_volume:.1f} Å³, {len(structure)} atoms "
                                                )
                                                status_placeholder.markdown(
                                                    f"- **Structure loaded:** `{structure.composition.reduced_formula}` (cod_{entry.get('file')})")
                                        except Exception as e:
                                            st.warning(
                                                f"Error processing COD entry {entry.get('file', 'unknown')}: {e}")
                                            continue

                                    if st.session_state.cod_options:
                                        if len(limited_entries) < len(cod_entries):
                                            st.info(
                                                f"Showing first {cod_limit} of {len(cod_entries)} total COD results. Increase limit to see more.")
                                        st.success(f"Found {len(st.session_state.cod_options)} structures in COD.")
                                    else:
                                        st.warning("COD: No valid structures could be processed.")
                                else:
                                    st.session_state.cod_options = []
                                    st.warning("COD: No matching structures found.")
                            except Exception as e:
                                st.warning(f"COD search error: {e}")
                                st.session_state.cod_options = []

           # with cols2:
           #     image = Image.open("images/Rabbit2.png")
           #     st.image(image, use_container_width=True)

        with cols3:
            if any(x in st.session_state for x in ['mp_options', 'aflow_options', 'cod_options']):
                tabs = []
                if 'mp_options' in st.session_state and st.session_state.mp_options:
                    tabs.append("Materials Project")
                if 'aflow_options' in st.session_state and st.session_state.aflow_options:
                    tabs.append("AFLOW")
                if 'cod_options' in st.session_state and st.session_state.cod_options:
                    tabs.append("COD")

                if tabs:
                    selected_tab = st.tabs(tabs)

                    tab_index = 0
                    if 'mp_options' in st.session_state and st.session_state.mp_options:
                        with selected_tab[tab_index]:
                            st.subheader("🧬 Structures Found in Materials Project")
                            selected_structure = st.selectbox("Select a structure from MP:",
                                                              st.session_state.mp_options)
                            selected_id = selected_structure.split(":")[0].strip()
                            composition = selected_structure.split(":", 1)[1].split("(")[0].strip()
                            file_name = f"{selected_id}_{composition}.cif"
                            file_name = re.sub(r'[\\/:"*?<>|]+', '_', file_name)

                            if selected_id in st.session_state.full_structures_see:
                                selected_entry = st.session_state.full_structures_see[selected_id]

                                conv_lattice = selected_entry.lattice
                                cell_volume = selected_entry.lattice.volume
                                density = str(selected_entry.density).split()[0]
                                n_atoms = len(selected_entry)
                                atomic_den = n_atoms / cell_volume

                                structure_type = identify_structure_type(selected_entry)
                                st.write(f"**Structure type:** {structure_type}")
                                analyzer = SpacegroupAnalyzer(selected_entry)
                                st.write(
                                    f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")

                                st.write(
                                    f"**Material ID:** {selected_id}, **Formula:** {composition}, N. of Atoms {n_atoms}")

                                st.write(
                                    f"**Conventional Lattice:** a = {conv_lattice.a:.4f} Å, b = {conv_lattice.b:.4f} Å, c = {conv_lattice.c:.4f} Å, α = {conv_lattice.alpha:.1f}°, β = {conv_lattice.beta:.1f}°, γ = {conv_lattice.gamma:.1f}° (Volume {cell_volume:.1f} Å³)")
                                st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                                mp_url = f"https://materialsproject.org/materials/{selected_id}"
                                st.write(f"**Link:** {mp_url}")

                                col_mpd, col_mpb = st.columns([2, 1])
                                with col_mpd:
                                    if st.button("Add Selected Structure (MP)", key="add_btn_mp"):
                                        pmg_structure = st.session_state.full_structures_see[selected_id]
                                        check_structure_size_and_warn(pmg_structure, f"MP structure {selected_id}")
                                        st.session_state.full_structures[file_name] = pmg_structure
                                        cif_writer = CifWriter(pmg_structure)
                                        cif_content = cif_writer.__str__()
                                        cif_file = io.BytesIO(cif_content.encode('utf-8'))
                                        cif_file.name = file_name
                                        if 'uploaded_files' not in st.session_state:
                                            st.session_state.uploaded_files = []
                                        if all(f.name != file_name for f in st.session_state.uploaded_files):
                                            st.session_state.uploaded_files.append(cif_file)
                                        st.success("Structure added from Materials Project!")
                                with col_mpb:
                                    st.download_button(
                                        label="Download MP CIF",
                                        data=str(
                                            CifWriter(st.session_state.full_structures_see[selected_id], symprec=0.01)),
                                        file_name=file_name,
                                        type="primary",
                                        mime="chemical/x-cif"
                                    )
                                    st.info(
                                        f"**Note**: If H element is missing in CIF file, it is not shown in the formula either.")
                        tab_index += 1

                    if 'aflow_options' in st.session_state and st.session_state.aflow_options:
                        with selected_tab[tab_index]:
                            st.subheader("🧬 Structures Found in AFLOW")
                            st.warning(
                                "The AFLOW does not provide atomic occupancies and includes only information about primitive cell in API. For better performance, volume and n. of atoms are purposely omitted from the expander.")
                            selected_structure = st.selectbox("Select a structure from AFLOW:",
                                                              st.session_state.aflow_options)
                            selected_auid = selected_structure.split(": ")[0].strip()
                            selected_entry = next(
                                (entry for entry in st.session_state.entrys.values() if entry.auid == selected_auid),
                                None)
                            if selected_entry:

                                cif_files = [f for f in selected_entry.files if
                                             f.endswith("_sprim.cif") or f.endswith(".cif")]

                                if cif_files:

                                    cif_filename = cif_files[0]

                                    # Correct the AURL: replace the first ':' with '/'

                                    host_part, path_part = selected_entry.aurl.split(":", 1)

                                    corrected_aurl = f"{host_part}/{path_part}"

                                    file_url = f"http://{corrected_aurl}/{cif_filename}"
                                    response = requests.get(file_url)
                                    cif_content = response.content

                                    structure_from_aflow = Structure.from_str(cif_content.decode('utf-8'), fmt="cif")
                                    converted_structure = get_full_conventional_structure(structure_from_aflow,
                                                                                          symprec=0.1)

                                    conv_lattice = converted_structure.lattice
                                    cell_volume = converted_structure.lattice.volume
                                    density = str(converted_structure.density).split()[0]
                                    n_atoms = len(converted_structure)
                                    atomic_den = n_atoms / cell_volume

                                    structure_type = identify_structure_type(converted_structure)
                                    st.write(f"**Structure type:** {structure_type}")
                                    analyzer = SpacegroupAnalyzer(structure_from_aflow)
                                    st.write(
                                        f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")
                                    st.write(
                                        f"**AUID:** {selected_entry.auid}, **Formula:** {selected_entry.compound}, **N. of Atoms:** {n_atoms}")
                                    st.write(
                                        f"**Conventional Lattice:** a = {conv_lattice.a:.4f} Å, b = {conv_lattice.b:.4f} Å, c = {conv_lattice.c:.4f} Å, α = {conv_lattice.alpha:.1f}°, β = {conv_lattice.beta:.1f}°, "
                                        f"γ = {conv_lattice.gamma:.1f}° (Volume {cell_volume:.1f} Å³)")
                                    st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                                    linnk = f"https://aflowlib.duke.edu/search/ui/material/?id=" + selected_entry.auid
                                    st.write("**Link:**", linnk)

                                    if st.button("Add Selected Structure (AFLOW)", key="add_btn_aflow"):
                                        if 'uploaded_files' not in st.session_state:
                                            st.session_state.uploaded_files = []
                                        cif_file = io.BytesIO(cif_content)
                                        cif_file.name = f"{selected_entry.compound}_{selected_entry.auid}.cif"

                                        st.session_state.full_structures[cif_file.name] = structure_from_aflow

                                        check_structure_size_and_warn(structure_from_aflow, cif_file.name)
                                        if all(f.name != cif_file.name for f in st.session_state.uploaded_files):
                                            st.session_state.uploaded_files.append(cif_file)
                                        st.success("Structure added from AFLOW!")

                                    st.download_button(
                                        label="Download AFLOW CIF",
                                        data=cif_content,
                                        file_name=f"{selected_entry.compound}_{selected_entry.auid}.cif",
                                        type="primary",
                                        mime="chemical/x-cif"
                                    )
                                    st.info(
                                        f"**Note**: If H element is missing in CIF file, it is not shown in the formula either.")
                                else:
                                    st.warning("No CIF file found for this AFLOW entry.")
                        tab_index += 1

                    # COD tab
                    if 'cod_options' in st.session_state and st.session_state.cod_options:
                        with selected_tab[tab_index]:
                            st.subheader("🧬 Structures Found in COD")
                            selected_cod_structure = st.selectbox(
                                "Select a structure from COD:",
                                st.session_state.cod_options,
                                key='sidebar_select_cod'
                            )
                            cod_id = selected_cod_structure.split(":")[0].strip()
                            if cod_id in st.session_state.full_structures_see_cod:
                                selected_entry = st.session_state.full_structures_see_cod[cod_id]
                                lattice = selected_entry.lattice
                                cell_volume = selected_entry.lattice.volume
                                density = str(selected_entry.density).split()[0]
                                n_atoms = len(selected_entry)
                                atomic_den = n_atoms / cell_volume

                                idcodd = cod_id.removeprefix("cod_")

                                structure_type = identify_structure_type(selected_entry)
                                st.write(f"**Structure type:** {structure_type}")
                                analyzer = SpacegroupAnalyzer(selected_entry)
                                st.write(
                                    f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")

                                st.write(
                                    f"**COD ID:** {idcodd}, **Formula:** {selected_entry.composition.reduced_formula}, **N. of Atoms:** {n_atoms}")
                                st.write(
                                    f"**Conventional Lattice:** a = {lattice.a:.3f} Å, b = {lattice.b:.3f} Å, c = {lattice.c:.3f} Å, α = {lattice.alpha:.2f}°, β = {lattice.beta:.2f}°, γ = {lattice.gamma:.2f}° (Volume {cell_volume:.1f} Å³)")
                                st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                                cod_url = f"https://www.crystallography.net/cod/{cod_id.split('_')[1]}.html"
                                st.write(f"**Link:** {cod_url}")

                                file_name = f"{selected_entry.composition.reduced_formula}_COD_{cod_id.split('_')[1]}.cif"

                                if st.button("Add Selected Structure (COD)", key="sid_add_btn_cod"):
                                    cif_writer = CifWriter(selected_entry, symprec=0.01)
                                    cif_data = str(cif_writer)
                                    st.session_state.full_structures[file_name] = selected_entry
                                    cif_file = io.BytesIO(cif_data.encode('utf-8'))
                                    cif_file.name = file_name
                                    if 'uploaded_files' not in st.session_state:
                                        st.session_state.uploaded_files = []
                                    if all(f.name != file_name for f in st.session_state.uploaded_files):
                                        st.session_state.uploaded_files.append(cif_file)

                                    check_structure_size_and_warn(selected_entry, file_name)
                                    st.success("Structure added from COD!")

                                st.download_button(
                                    label="Download COD CIF",
                                    data=str(CifWriter(selected_entry, symprec=0.01)),
                                    file_name=file_name,
                                    mime="chemical/x-cif", type="primary",
                                )
                                st.info(
                                    f"**Note**: If H element is missing in CIF file, it is not shown in the formula either.")



if "first_run_note" not in st.session_state: st.session_state.first_run_note = True
if st.session_state.first_run_note:
    colh1, colh2 = st.columns([1, 3])
    with colh1:
        try:
            st.image(Image.open("images/Rb.png"))
        except FileNotFoundError:
            st.warning("Note: Rb.png image not found in images/ folder.")
    with colh2:
        st.info("""
        Upload your crystal structure file. It will default to its **conventional cell**.
        You can optionally customize the cell representation. Then, apply supercell dimensions and create point defects.
        """)
    st.session_state.first_run_note = False

default_session_states = {
    'uploaded_files': [], 'full_structures': {}, 'active_original_structure': None,
    'represented_structure': None, 'current_cell_representation_type': "Conventional Cell",
    'change_cell_representation_checkbox': False, 'selected_file': None,
    'current_structure': None, 'current_structure_before_defects': None,
    'supercell_settings_applied': False, 'supercell_n_a': 1, 'supercell_n_b': 1, 'supercell_n_c': 1,
    'applied_supercell_n_a': 1, 'applied_supercell_n_b': 1, 'applied_supercell_n_c': 1,
    'applied_cell_type_name': "Conventional_Cell", 'helpful': False, 'preview_structure': None,
    'show_3d_visualization': True, 'show_atomic_labels': False, 'generated_structures': {},
    'enable_batch_generation': False
}
for key, value in default_session_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

CELL_REPRESENTATION_OPTIONS = ["Conventional Cell", "Primitive Cell (Niggli)", "Primitive Cell (LLL)",
                               "Primitive Cell (no reduction)", "Orthogonal Cell"]

st.markdown("<hr style='border: none; height: 6px; background-color: #3399ff; border-radius: 8px; margin: 20px 0;'>",
            unsafe_allow_html=True)


def reset_supercell_and_defect_states():
    st.session_state.supercell_settings_applied = False
    st.session_state.current_structure_before_defects = None
    st.session_state.supercell_n_a = 1
    st.session_state.supercell_n_b = 1
    st.session_state.supercell_n_c = 1
    st.session_state.applied_supercell_n_a = 1
    st.session_state.applied_supercell_n_b = 1
    st.session_state.applied_supercell_n_c = 1
    st.session_state.helpful = False


def set_default_conventional_representation(show_error=True):
    if st.session_state.active_original_structure:
        try:
            analyzer = SpacegroupAnalyzer(st.session_state.active_original_structure, symprec=0.1)
            default_conventional = analyzer.get_conventional_standard_structure()
            st.session_state.represented_structure = default_conventional.copy()
            st.session_state.current_structure = default_conventional.copy()
            st.session_state.current_cell_representation_type = "Conventional Cell"
            st.session_state.applied_cell_type_name = "Conventional_Cell"
            st.session_state.preview_structure = None
            reset_supercell_and_defect_states()
            return True
        except Exception as e_conv:
            if show_error: st.error(
                f"Could not generate conventional cell for {st.session_state.selected_file}: {e_conv}")
            st.session_state.represented_structure = st.session_state.active_original_structure.copy()
            st.session_state.current_structure = st.session_state.active_original_structure.copy()
            st.session_state.current_cell_representation_type = "Original Uploaded"
            st.session_state.applied_cell_type_name = "Original_Uploaded"
            st.session_state.preview_structure = None
            reset_supercell_and_defect_states()
            return False
    return False


st.sidebar.markdown("## 🍕 XRDlicious")
st.sidebar.subheader("📁📤 Upload Your Structure Files")
uploaded_files_user_sidebar = st.sidebar.file_uploader(
    "Upload Structure Files (CIF, POSCAR, XSF, PW, CFG, ...):",
    type=None, accept_multiple_files=True, key="sidebar_uploader"
)

if uploaded_files_user_sidebar:
    for file_data in uploaded_files_user_sidebar:
        if file_data.name not in [f.name for f in st.session_state.uploaded_files]:
            try:
                bytes_data = file_data.getvalue()
                temp_file = io.BytesIO(bytes_data)
                temp_file.name = file_data.name
                structure_pmg_raw = load_structure(temp_file)
                st.session_state.full_structures[file_data.name] = structure_pmg_raw
                st.session_state.uploaded_files.append(file_data)
            except Exception as e:
                st.sidebar.error(f"Failed to parse {file_data.name}: {e}")
                if file_data.name in st.session_state.full_structures: del st.session_state.full_structures[
                    file_data.name]

st.sidebar.markdown("### Final List of Structure Files:")
st.sidebar.write(
    [f.name for f in st.session_state.uploaded_files] if st.session_state.uploaded_files else "No files uploaded yet.")

st.sidebar.markdown("### 🗑️ Remove uploaded structure(s)")
files_to_remove_names = []
for i, file_obj in enumerate(list(st.session_state.uploaded_files)):
    col1_rem, col2_rem = st.sidebar.columns([4, 1])
    col1_rem.write(file_obj.name)
    if col2_rem.button("❌", key=f"remove_{file_obj.name}_{i}"):
        files_to_remove_names.append(file_obj.name)

if files_to_remove_names:
    st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if
                                       f.name not in files_to_remove_names]
    for name in files_to_remove_names:
        if name in st.session_state.full_structures: del st.session_state.full_structures[name]
        if st.session_state.selected_file == name:
            for key in default_session_states: st.session_state[key] = default_session_states[key]
            st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if
                                               f.name != name]
    st.rerun()

if st.session_state.uploaded_files:
    file_options = [f.name for f in st.session_state.uploaded_files]
    current_selected_file_index = 0
    if st.session_state.selected_file and st.session_state.selected_file in file_options:
        current_selected_file_index = file_options.index(st.session_state.selected_file)
    elif file_options:
        st.session_state.selected_file = file_options[0]

    st.subheader("Select Structure for Operations:")
    selector_key = "file_selector_selectbox" if len(file_options) > 5 else "file_selector_radio"
    newly_selected_file = st.selectbox("Available files:", file_options, index=current_selected_file_index,
                                       key=selector_key) if len(file_options) > 5 else st.radio("Available files:",
                                                                                                file_options,
                                                                                                index=current_selected_file_index,
                                                                                                key=selector_key)

    if newly_selected_file and newly_selected_file != st.session_state.selected_file:
        st.session_state.selected_file = newly_selected_file
        if st.session_state.selected_file in st.session_state.full_structures:
            st.session_state.active_original_structure = st.session_state.full_structures[
                st.session_state.selected_file].copy()
            set_default_conventional_representation()
            st.session_state.change_cell_representation_checkbox = False
        else:
            st.session_state.active_original_structure = None
            st.session_state.represented_structure = None
            st.session_state.current_structure = None
        st.rerun()
    elif not st.session_state.active_original_structure and st.session_state.selected_file and st.session_state.selected_file in st.session_state.full_structures:
        st.session_state.active_original_structure = st.session_state.full_structures[
            st.session_state.selected_file].copy()
        set_default_conventional_representation()
        st.session_state.change_cell_representation_checkbox = False
        st.rerun()

    if st.session_state.current_structure and st.session_state.represented_structure:
        st.markdown("### 1. Cell Representation")

        col_checkbox, col_info = st.columns([1, 1])

        with col_checkbox:
            customize_cell = st.checkbox(
                "Customize Cell Representation (default is Conventional)",
                value=st.session_state.change_cell_representation_checkbox,
                key="cell_rep_checkbox",
                help="Check this box to choose a cell representation other than the default conventional cell."
            )

        if customize_cell != st.session_state.change_cell_representation_checkbox:
            st.session_state.change_cell_representation_checkbox = customize_cell
            if not customize_cell:
                if st.session_state.current_cell_representation_type != "Conventional Cell":
                    set_default_conventional_representation()
                    st.rerun()

        if st.session_state.change_cell_representation_checkbox:
            current_rep_type_idx = CELL_REPRESENTATION_OPTIONS.index(st.session_state.current_cell_representation_type) \
                if st.session_state.current_cell_representation_type in CELL_REPRESENTATION_OPTIONS else 0

            selected_representation_type_ui = st.radio(
                "Choose cell type:", options=CELL_REPRESENTATION_OPTIONS, index=current_rep_type_idx,
                key="cell_rep_radio", horizontal=True
            )

            if selected_representation_type_ui != st.session_state.current_cell_representation_type:
                if st.session_state.active_original_structure:
                    base_for_rep = st.session_state.active_original_structure.copy()
                    try:
                        _analyzer = SpacegroupAnalyzer(base_for_rep, symprec=0.1)
                        if selected_representation_type_ui == "Conventional Cell":
                            preview_structure = _analyzer.get_conventional_standard_structure()
                        elif selected_representation_type_ui == "Primitive Cell (Niggli)":
                            preview_structure = _analyzer.get_primitive_standard_structure().get_reduced_structure(
                                reduction_algo="niggli")
                        elif selected_representation_type_ui == "Primitive Cell (LLL)":
                            preview_structure = _analyzer.get_primitive_standard_structure().get_reduced_structure(
                                reduction_algo="LLL")
                        elif selected_representation_type_ui == "Primitive Cell (no reduction)":
                            preview_structure = _analyzer.get_primitive_standard_structure()
                        elif selected_representation_type_ui == "Orthogonal Cell":
                            preview_structure = get_orthogonal_cell(base_for_rep)

                        st.session_state.preview_structure = preview_structure
                    except Exception as e_preview:
                        st.error(
                            f"Error previewing cell representation '{selected_representation_type_ui}': {e_preview}")
                        st.session_state.preview_structure = None

            with col_info:
                if st.session_state.preview_structure:
                    st.markdown("**Preview:**")
                    preview_info = get_structure_info(st.session_state.preview_structure)
                    st.markdown(preview_info)

            if st.button("Apply Selected Cell Representation", key="apply_cell_rep_btn"):
                if st.session_state.preview_structure:
                    st.session_state.represented_structure = st.session_state.preview_structure.copy()
                    st.session_state.current_structure = st.session_state.preview_structure.copy()
                    st.session_state.current_cell_representation_type = selected_representation_type_ui
                    st.session_state.applied_cell_type_name = selected_representation_type_ui.replace(" ",
                                                                                                      "_").replace(
                        "(", "").replace(")", "").replace("__", "_")
                    st.session_state.preview_structure = None
                    reset_supercell_and_defect_states()
                    st.rerun()

        else:
            with col_info:
                current_info = get_structure_info(st.session_state.represented_structure)
                st.markdown(f"**Current: {st.session_state.current_cell_representation_type}**")
                st.markdown(current_info)

        st.info(
            f"Supercell will be built from: **{st.session_state.current_cell_representation_type}** ({len(st.session_state.represented_structure)} atoms)")

        st.markdown("### 2. Define Supercell")
        supercell_applied_locally = st.session_state.supercell_settings_applied

        sc_col1, sc_col2, sc_col3 = st.columns(3)
        with sc_col1:
            n_a_ui = st.number_input("Repeat a", 1, 10, st.session_state.supercell_n_a, 1, key="sc_a",
                                     disabled=supercell_applied_locally)
        with sc_col2:
            n_b_ui = st.number_input("Repeat b", 1, 10, st.session_state.supercell_n_b, 1, key="sc_b",
                                     disabled=supercell_applied_locally)
        with sc_col3:
            n_c_ui = st.number_input("Repeat c", 1, 10, st.session_state.supercell_n_c, 1, key="sc_c",
                                     disabled=supercell_applied_locally)

        base_atoms_sc_info = len(st.session_state.represented_structure)
        st.info(
            f"Preview: Applying dimensions to '{st.session_state.current_cell_representation_type}' ({base_atoms_sc_info} atoms) -> **{base_atoms_sc_info * n_a_ui * n_b_ui * n_c_ui} atoms**.")

        apply_sc_col, reset_sc_col = st.columns(2)
        with apply_sc_col:
            if st.button("Apply Supercell", disabled=supercell_applied_locally, key="apply_supercell_btn"):
                if st.session_state.represented_structure:
                    st.session_state.supercell_n_a, st.session_state.supercell_n_b, st.session_state.supercell_n_c = n_a_ui, n_b_ui, n_c_ui
                    try:
                        transformer = SupercellTransformation([[n_a_ui, 0, 0], [0, n_b_ui, 0], [0, 0, n_c_ui]])
                        final_sc = transformer.apply_transformation(st.session_state.represented_structure.copy())
                        st.session_state.current_structure_before_defects = final_sc.copy()
                        st.session_state.current_structure = final_sc.copy()
                        st.session_state.supercell_settings_applied = True
                        st.session_state.applied_supercell_n_a, st.session_state.applied_supercell_n_b, st.session_state.applied_supercell_n_c = n_a_ui, n_b_ui, n_c_ui
                        st.session_state.helpful = False
                        st.rerun()
                    except Exception as e_sc:
                        st.error(f"Error applying supercell: {e_sc}")

        with reset_sc_col:
            if st.button("Reset Transformations & Supercell", key="reset_all_trans_btn"):
                set_default_conventional_representation()
                st.session_state.change_cell_representation_checkbox = False
                st.rerun()

        if st.session_state.supercell_settings_applied and st.session_state.current_structure_before_defects:
            st.markdown("### 3. Create Point Defects on the Applied Supercell")
            if st.session_state.current_structure != st.session_state.current_structure_before_defects and not st.session_state.helpful:
                st.session_state.current_structure = st.session_state.current_structure_before_defects.copy()

            active_pmg_for_defects = st.session_state.current_structure
            col_defect_ops, col_defect_log = st.columns([2, 1])
            with col_defect_ops:
                atom_count_defects = len(active_pmg_for_defects)
                defect_op_limit = 32
                defect_ops = ["Insert Interstitials (Voronoi method)", "Insert Interstitials (Fast Grid method)",
                              "Create Vacancies", "Substitute Atoms"]
                current_defect_op_options = defect_ops
                if atom_count_defects > defect_op_limit:
                    st.warning(
                        f"⚠️ Interstitials (Voronoi) disabled: Structure has {atom_count_defects} atoms (limit: {defect_op_limit}).")
                    current_defect_op_options = ["Insert Interstitials (Fast Grid method)", "Create Vacancies",
                                                 "Substitute Atoms"]

                op_mode_key = "defect_op_mode_limited" if atom_count_defects > defect_op_limit else "defect_op_mode_full"
                operation_mode = st.selectbox("Choose Defect Operation", current_defect_op_options, key=op_mode_key)

                if operation_mode == "Insert Interstitials (Voronoi method)":
                    st.markdown("**Insert Interstitials Settings (Voronoi - Accurate but Slow)**")
                    int_c1, int_c2, int_c3 = st.columns(3)
                    int_el = int_c1.text_input("Element", "N", key="int_el_voronoi")
                    int_n = int_c2.number_input("# Insert", 1, key="int_n_voronoi")
                    int_type_idx = int_c3.number_input("Site Type (0=all)", 0, key="int_type_idx")
                    int_c4, int_c5, int_c6 = st.columns(3)
                    int_mode = int_c4.selectbox("Mode", ["farthest", "nearest", "moderate"], 0, key="int_mode")
                    int_clust = int_c5.number_input("Clust Tol.", 0.75, step=0.05, format="%.2f", key="int_clust")
                    int_min_dist = int_c6.number_input("Min Dist.", 0.5, step=0.05, format="%.2f", key="int_min_dist")
                    int_target = 0.5
                    if int_mode == "moderate":
                        int_target = st.number_input("Target (0=nearest, 1=farthest)", 0.0, 1.0, 0.5, 0.1,
                                                     format="%.1f",
                                                     key="int_target_voronoi")
                elif operation_mode == "Insert Interstitials (Fast Grid method)":
                    st.markdown("**Insert Interstitials Settings (Fast Grid - Quick for Large Structures)**")
                    int_c1, int_c2, int_c3 = st.columns(3)
                    int_el_fast = int_c1.text_input("Element", "N", key="int_el_fast")
                    int_n_fast = int_c2.number_input("# Insert", 1, key="int_n_fast")
                    int_mode_fast = int_c3.selectbox("Mode", ["moderate"], 0,
                                                     key="int_mode_fast")
                    int_c4, int_c5, int_c6 = st.columns(3)
                    int_min_dist_fast = int_c4.number_input("Min Distance from atoms (Å)", 1.5, step=0.1, format="%.1f",
                                                            key="int_min_dist_fast",
                                                            help="Minimum distance from existing atoms")
                    int_grid_spacing = int_c5.number_input("Grid Spacing (Å)", 0.5, step=0.1, format="%.1f",
                                                           key="int_grid_spacing",
                                                           help="Smaller = more precise but slower")
                    int_min_int_dist = int_c6.number_input("Min Interstitial-Interstitial Distance (Å)", 1.0, step=0.1,
                                                           format="%.1f",
                                                           key="int_min_int_dist",
                                                           help="Minimum distance between interstitials")
                    int_target_fast = 0.5
                    if int_mode_fast == "moderate":
                        int_target_fast = st.number_input("Target (0=nearest, 1=farthest)", 0.0, 1.0, 0.5, 0.1,
                                                          format="%.1f",
                                                          key="int_target_fast")
                elif operation_mode == "Create Vacancies":
                    st.markdown("**Create Vacancies Settings**")
                    vac_c1, vac_c2 = st.columns(2)
                    vac_mode = vac_c1.selectbox("Selection Mode", ["farthest", "nearest", "moderate"], 0,
                                                key="vac_mode")
                    vac_target = 0.5
                    if vac_mode == "moderate":
                        vac_target = vac_c2.number_input("Target (0=nearest, 1=farthest)", 0.0, 1.0, 0.5, 0.1,
                                                         format="%.1f",
                                                         key="vac_target")
                    vac_els = sorted(list(set(s.specie.symbol for s in active_pmg_for_defects.sites if s.specie)))
                    vac_percent = {}
                    if vac_els:
                        st.markdown("**Set vacancy percentages:**")
                        cols_pr = 3
                        n_rows = (len(vac_els) + cols_pr - 1) // cols_pr
                        for r in range(n_rows):
                            vac_perc_cols = st.columns(cols_pr)
                            for c_idx in range(cols_pr):
                                el_idx = r * cols_pr + c_idx
                                if el_idx < len(vac_els):
                                    el_v = vac_els[el_idx]
                                    el_count = sum(
                                        1 for site in active_pmg_for_defects.sites if site.specie.symbol == el_v)
                                    with vac_perc_cols[c_idx]:
                                        vac_percent[el_v] = st.number_input(
                                            f"% {el_v} (total: {el_count})", 0.0, 100.0, 0.0, 1.0, "%.1f",
                                            key=f"vac_perc_{el_v}",
                                            help=f"Remove percentage of {el_count} {el_v} atoms"
                                        )

                        st.markdown("**Preview of vacancies to be created:**")
                        preview_text = []
                        total_to_remove = 0
                        for el_symbol, perc_to_remove in vac_percent.items():
                            if perc_to_remove > 0:
                                el_count = sum(
                                    1 for site in active_pmg_for_defects.sites if site.specie.symbol == el_symbol)
                                n_to_remove = int(round(el_count * perc_to_remove / 100.0))
                                total_to_remove += n_to_remove
                                preview_text.append(
                                    f"**{el_symbol}**: {n_to_remove} atoms ({perc_to_remove}% of {el_count})")

                        if preview_text:
                            for text in preview_text:
                                st.write(f"• {text}")
                            st.info(f"**Total atoms to remove: {total_to_remove}**")
                        else:
                            st.info("No vacancies will be created with current settings.")
                    else:
                        st.warning("No elements for vacancies.")
                elif operation_mode == "Substitute Atoms":
                    st.markdown("**Substitute Atoms Settings**")
                    sub_c1, sub_c2 = st.columns(2)
                    sub_mode = sub_c1.selectbox("Selection Mode", ["farthest", "nearest", "moderate"], 0,
                                                key="sub_mode")
                    sub_target = 0.5
                    if sub_mode == "moderate":
                        sub_target = sub_c2.number_input("Target (0=nearest, 1=farthest)", 0.0, 1.0, 0.5, 0.1,
                                                         format="%.1f",
                                                         key="sub_target")
                    sub_els = sorted(list(set(s.specie.symbol for s in active_pmg_for_defects.sites if s.specie)))
                    sub_settings = {}
                    if sub_els:
                        st.markdown("**Set substitution parameters:**")
                        cols_pr_s = 2
                        n_rows_s = (len(sub_els) + cols_pr_s - 1) // cols_pr_s
                        for r_s in range(n_rows_s):
                            sub_perc_cols = st.columns(cols_pr_s * 2)
                            for c_idx_s in range(cols_pr_s):
                                el_idx_s = r_s * cols_pr_s + c_idx_s
                                if el_idx_s < len(sub_els):
                                    el_s = sub_els[el_idx_s]
                                    el_count_s = sum(
                                        1 for site in active_pmg_for_defects.sites if site.specie.symbol == el_s)
                                    wc1, wc2 = c_idx_s * 2, c_idx_s * 2 + 1
                                    if wc2 < len(sub_perc_cols):
                                        with sub_perc_cols[wc1]:
                                            sub_p = st.number_input(
                                                f"% {el_s} (total: {el_count_s})", 0.0, 100.0, 0.0, 1.0, "%.1f",
                                                key=f"sub_p_{el_s}",
                                                help=f"Substitute percentage of {el_count_s} {el_s} atoms"
                                            )
                                        with sub_perc_cols[wc2]:
                                            sub_t_el = st.text_input(f"Replace {el_s} with", key=f"sub_t_{el_s}")
                                        sub_settings[el_s] = {"percentage": sub_p, "substitute": sub_t_el.strip()}
                                    elif wc1 < len(sub_perc_cols):
                                        with sub_perc_cols[wc1]:
                                            sub_p = st.number_input(
                                                f"% {el_s} (total: {el_count_s})", 0.0, 100.0, 0.0, 1.0, "%.1f",
                                                key=f"sub_p_{el_s}",
                                                help=f"Substitute percentage of {el_count_s} {el_s} atoms"
                                            )
                                            sub_t_el = st.text_input(f"Replace {el_s} with", key=f"sub_t_{el_s}",
                                                                     help="Substitute element")
                                            sub_settings[el_s] = {"percentage": sub_p, "substitute": sub_t_el.strip()}

                        st.markdown("**Preview of substitutions to be made:**")
                        preview_text_sub = []
                        total_to_substitute = 0
                        for orig_el_symbol, settings in sub_settings.items():
                            perc_to_sub = settings.get("percentage", 0)
                            sub_el_symbol = settings.get("substitute", "").strip()
                            if perc_to_sub > 0 and sub_el_symbol:
                                el_count_sub = sum(
                                    1 for site in active_pmg_for_defects.sites if site.specie.symbol == orig_el_symbol)
                                n_to_substitute = int(round(el_count_sub * perc_to_sub / 100.0))
                                total_to_substitute += n_to_substitute
                                preview_text_sub.append(
                                    f"**{orig_el_symbol} → {sub_el_symbol}**: {n_to_substitute} atoms ({perc_to_sub}% of {el_count_sub})")

                        if preview_text_sub:
                            for text in preview_text_sub:
                                st.write(f"• {text}")
                            st.info(f"**Total atoms to substitute: {total_to_substitute}**")
                        else:
                            st.info("No substitutions will be made with current settings.")
                    else:
                        st.warning("No elements for substitution.")

                enable_batch = st.checkbox(
                    "🎲 Enable Batch Generation (Generate multiple configurations with different random seeds)",
                    value=st.session_state.enable_batch_generation,
                    key="enable_batch_checkbox")
                if enable_batch != st.session_state.enable_batch_generation:
                    st.session_state.enable_batch_generation = enable_batch

                if st.session_state.enable_batch_generation:
                    st.markdown("### 🎲 Batch Generation & Download")
                    st.info(
                        "Generate multiple configurations using the parameters configured above with different random seeds")

                    batch_col1, batch_col2 = st.columns(2)

                    with batch_col1:
                        st.markdown("##### Generate Multiple Configurations")
                        n_configurations = st.number_input("Number of configurations", 1, 50, 10, 1, key="n_configs")
                        starting_seed = st.number_input("Starting random seed", 0, 9999, 42, 1, key="start_seed")

                        if st.button("🎲 Generate Multiple Defect Configurations", key="generate_batch_btn"):
                            if st.session_state.current_structure_before_defects:
                                with st.spinner(f"Generating {n_configurations} configurations..."):
                                    st.session_state.generated_structures = {}

                                    for i in range(n_configurations):
                                        seed = starting_seed + i
                                        config_name = f"config_{i + 1:02d}_seed{seed}"

                                        base_struct = st.session_state.current_structure_before_defects.copy()

                                        if operation_mode == "Insert Interstitials (Fast Grid method)":
                                            modified_struct = insert_interstitials_ase_fast(
                                                base_struct, int_el_fast, int_n_fast, int_min_dist_fast,
                                                int_grid_spacing, int_mode_fast, int_min_int_dist, None, seed
                                            )
                                        elif operation_mode == "Insert Interstitials (Voronoi method)":
                                            modified_struct = insert_interstitials_into_structure(
                                                base_struct, int_el, int_n, int_type_idx, int_mode,
                                                int_clust, int_min_dist, int_target, None, seed
                                            )
                                        elif operation_mode == "Create Vacancies":
                                            modified_struct = remove_vacancies_from_structure(
                                                base_struct, vac_percent, vac_mode, vac_target, None, seed
                                            )
                                        elif operation_mode == "Substitute Atoms":
                                            modified_struct = substitute_atoms_in_structure(
                                                base_struct, sub_settings, sub_mode, sub_target, None, seed
                                            )
                                        else:
                                            modified_struct = base_struct

                                        st.session_state.generated_structures[config_name] = modified_struct

                                    st.success(
                                        f"Generated {len(st.session_state.generated_structures)} configurations!")
                            else:
                                st.error("No supercell structure available. Apply supercell first.")

                    with batch_col2:
                        st.markdown("##### Batch Download")
                        if st.session_state.generated_structures:
                            st.info(f"📦 {len(st.session_state.generated_structures)} configurations ready")

                            download_format = st.selectbox("Download format", ["CIF", "VASP", "LAMMPS", "XYZ"],
                                                           key="batch_format")

                            batch_fractional = True
                            batch_selective = False
                            batch_atom_style = "atomic"
                            batch_units = "metal"
                            batch_masses = True
                            batch_skew = False

                            if download_format == "VASP":
                                batch_fractional = st.checkbox("Fractional coordinates", True, key="batch_frac")
                                batch_selective = st.checkbox("Selective dynamics", False, key="batch_sel")
                            elif download_format == "LAMMPS":
                                batch_atom_style = st.selectbox("atom_style", ["atomic", "charge", "full"],
                                                                key="batch_atom_style")
                                batch_units = st.selectbox("units", ["metal", "real", "si"], key="batch_units")
                                batch_masses = st.checkbox("Include masses", True, key="batch_masses")
                                batch_skew = st.checkbox("Force skew", False, key="batch_skew")

                            if st.button("📦 Download All Configurations", key="download_batch_btn"):
                                import zipfile
                                from io import BytesIO

                                zip_buffer = BytesIO()

                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                    for config_name, structure in st.session_state.generated_structures.items():
                                        try:
                                            if download_format == "CIF":
                                                content = str(CifWriter(structure, symprec=0.1, refine_struct=False))
                                                filename = f"{config_name}.cif"

                                            elif download_format == "VASP":
                                                ase_struct = AseAtomsAdaptor.get_atoms(structure)
                                                if batch_selective:
                                                    from ase.constraints import FixAtoms

                                                    ase_struct.set_constraint(FixAtoms(indices=[]))

                                                sio = StringIO()
                                                write(sio, ase_struct, format="vasp", direct=batch_fractional,
                                                      sort=True)
                                                content = sio.getvalue()
                                                filename = f"{config_name}.poscar"

                                            elif download_format == "LAMMPS":
                                                ase_struct = AseAtomsAdaptor.get_atoms(structure)
                                                sio = StringIO()
                                                write(sio, ase_struct, format="lammps-data",
                                                      atom_style=batch_atom_style, units=batch_units,
                                                      masses=batch_masses, force_skew=batch_skew)
                                                content = sio.getvalue()
                                                filename = f"{config_name}_{batch_atom_style}.lmp"

                                            elif download_format == "XYZ":
                                                lattice_vectors = structure.lattice.matrix
                                                cart_coords = []
                                                elements = []
                                                for site in structure:
                                                    cart_coords.append(
                                                        structure.lattice.get_cartesian_coords(site.frac_coords))
                                                    elements.append(site.specie.symbol)

                                                xyz_lines = [str(len(structure))]
                                                lattice_string = " ".join(
                                                    [f"{x:.6f}" for row in lattice_vectors for x in row])
                                                properties = "Properties=species:S:1:pos:R:3"
                                                xyz_lines.append(f'Lattice="{lattice_string}" {properties}')

                                                for element, coord in zip(elements, cart_coords):
                                                    xyz_lines.append(
                                                        f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")

                                                content = "\n".join(xyz_lines)
                                                filename = f"{config_name}.xyz"

                                            zip_file.writestr(filename, content)

                                        except Exception as e:
                                            st.error(f"Error processing {config_name}: {e}")

                                zip_buffer.seek(0)

                                base_name = st.session_state.selected_file.split('.')[
                                    0] if st.session_state.selected_file else "defects"
                                zip_filename = f"{base_name}_{operation_mode.replace(' ', '_')}_{n_configurations}configs.zip"

                                st.download_button(
                                    label="💾 Download ZIP Archive",
                                    data=zip_buffer.getvalue(),
                                    file_name=zip_filename,
                                    mime="application/zip",
                                    type="primary"
                                )
                        else:
                            st.info("Generate configurations first to enable batch download")

                if operation_mode == "Insert Interstitials (Voronoi method)":
                    col_preview, col_apply = st.columns(2)
                    with col_preview:
                        if st.button("Preview Interstitial Sites", key="preview_interstitials_btn"):
                            with col_defect_log:
                                st.markdown("###### Interstitial Sites Preview")
                                with st.spinner("Calculating available interstitials sites..."):
                                    try:
                                        from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator

                                        generator = VoronoiInterstitialGenerator(clustering_tol=int_clust,
                                                                                 min_dist=int_min_dist)
                                        unique_interstitial_types = list(generator.generate(active_pmg_for_defects, "H"))

                                        if not unique_interstitial_types:
                                            st.warning("No interstitial sites found.")
                                        else:
                                            st.success(
                                                f"Found {len(unique_interstitial_types)} unique interstitial site types:")

                                            total_sites = 0
                                            for i_type, interstitial_type_obj in enumerate(unique_interstitial_types):
                                                site_coords = interstitial_type_obj.site.frac_coords
                                                site_classification = classify_interstitial_site(active_pmg_for_defects,
                                                                                                 site_coords)
                                                equiv_sites_count = len(interstitial_type_obj.equivalent_sites)
                                                total_sites += equiv_sites_count

                                                st.write(f"**Type {i_type + 1}:**")
                                                st.write(f"  Position: {np.round(site_coords, 4)}")
                                                st.write(f"  Classification: {site_classification}")
                                                st.write(f"  Equivalent sites: {equiv_sites_count}")
                                                st.write("---")

                                            st.info(f"**Total available interstitial sites: {total_sites}**")

                                            if int_type_idx == 0:
                                                st.write(f"Selected: All types ({total_sites} sites)")
                                            elif 0 < int_type_idx <= len(unique_interstitial_types):
                                                selected_type = unique_interstitial_types[int_type_idx - 1]
                                                selected_count = len(selected_type.equivalent_sites)
                                                st.write(f"Selected: Type {int_type_idx} ({selected_count} sites)")

                                    except Exception as e:
                                        st.error(f"Error analyzing interstitial sites: {e}")

                    with col_apply:
                        if st.button("Apply Interstitials to Structure", key="apply_interstitials_btn"):
                            init_n = len(active_pmg_for_defects)
                            mod_struct = insert_interstitials_into_structure(active_pmg_for_defects, int_el, int_n,
                                                                             int_type_idx,
                                                                             int_mode, int_clust, int_min_dist,
                                                                             int_target, col_defect_log)
                            if len(mod_struct) > init_n:
                                st.session_state.current_structure = mod_struct
                                st.session_state.helpful = True
                                with col_defect_log:
                                    st.success("Interstitials applied to structure.")
                                st.rerun()
                            else:
                                with col_defect_log:
                                    st.warning("No interstitials were added to the structure.")

                elif st.button("Apply Selected Defect Operation", key="apply_defect_op_btn"):
                    mod_struct = active_pmg_for_defects.copy()
                    changed = False
                    with col_defect_log:
                        st.markdown(f"###### Log: {operation_mode}")

                    if operation_mode == "Insert Interstitials (Fast Grid method)":
                        init_n = len(mod_struct)
                        mod_struct = insert_interstitials_ase_fast(mod_struct, int_el_fast, int_n_fast,
                                                                   int_min_dist_fast, int_grid_spacing, int_mode_fast,
                                                                   int_min_int_dist, col_defect_log)
                        if len(mod_struct) > init_n: changed = True
                    elif operation_mode == "Create Vacancies":
                        init_n = len(mod_struct)
                        mod_struct = remove_vacancies_from_structure(mod_struct, vac_percent, vac_mode, vac_target,
                                                                     col_defect_log)
                        if len(mod_struct) < init_n: changed = True
                    elif operation_mode == "Substitute Atoms":
                        init_comp = mod_struct.composition
                        mod_struct = substitute_atoms_in_structure(mod_struct, sub_settings, sub_mode, sub_target,
                                                                   col_defect_log)
                        if mod_struct.composition != init_comp: changed = True

                    st.session_state.current_structure = mod_struct
                    st.session_state.helpful = changed
                    with col_defect_log:
                        col_defect_log.success(
                            "Defect op applied." if changed else "Defect op finished (structure may be unchanged).")
                    st.rerun()

                if st.button("🔄 Reset Defects (to applied supercell)", key="reset_defects_btn"):
                    if st.session_state.current_structure_before_defects:
                        st.session_state.current_structure = st.session_state.current_structure_before_defects.copy()
                    st.session_state.helpful = False
                    with col_defect_log:
                        st.info("Defects reset to post-supercell state.")
                    st.rerun()
            with col_defect_log:
                if 'apply_defect_op_btn' not in st.session_state or not st.session_state.get('helpful',
                                                                                             True):
                    st.caption("Logs from defect operations will appear here.")

        elif not st.session_state.supercell_settings_applied and st.session_state.represented_structure:
            st.warning(
                "Supercell not yet applied. Please apply supercell dimensions in Step 2 before creating defects.")

        st.markdown("---")
        st.markdown("### 🔬 Structure Visualization & Download")
        pmg_to_visualize = st.session_state.current_structure
        if pmg_to_visualize:
            ase_to_visualize = AseAtomsAdaptor.get_atoms(pmg_to_visualize)
            col_viz, col_dl = st.columns([2, 1])
            with col_viz:
                show_3d = st.checkbox("Show 3D Visualization",
                                      value=st.session_state.show_3d_visualization,
                                      key="show_3d_cb_main")
                if show_3d != st.session_state.show_3d_visualization:
                    st.session_state.show_3d_visualization = show_3d

                show_labels = st.checkbox("Show atomic labels",
                                          value=st.session_state.show_atomic_labels,
                                          key="show_labels_cb_main")
                if show_labels != st.session_state.show_atomic_labels:
                    st.session_state.show_atomic_labels = show_labels

                if st.session_state.show_3d_visualization:
                    xyz_io = StringIO()
                    write(xyz_io, ase_to_visualize, format="xyz")
                    xyz_str = xyz_io.getvalue()
                    view = py3Dmol.view(width=600, height=500)
                    view.addModel(xyz_str, "xyz")
                    view.setStyle({'model': 0}, {"sphere": {"radius": 0.3, "colorscheme": "Jmol"}})
                    cell_viz = ase_to_visualize.get_cell()
                    if np.linalg.det(cell_viz) > 1e-6:
                        add_box(view, cell_viz, color='black', linewidth=1.5)
                    if st.session_state.show_atomic_labels:
                        for i, atom in enumerate(ase_to_visualize):
                            view.addLabel(f"{atom.symbol}{i}", {
                                "position": {"x": atom.position[0], "y": atom.position[1], "z": atom.position[2]},
                                "backgroundColor": "white", "fontColor": "black", "fontSize": 10,
                                "borderThickness": 0.5, "borderColor": "grey"})
                    view.zoomTo()
                    view.zoom(1.1)
                    html_viz = view._make_html()
                    st.components.v1.html(
                        f"<div style='display:flex;justify-content:center;border:1px solid #e0e0e0;border-radius:5px;overflow:hidden;min-height:510px;'>{html_viz}</div>",
                        height=520)

                    elems_legend = sorted(list(set(ase_to_visualize.get_chemical_symbols())))
                    legend_items = [
                        f"<div style='margin-right:10px;display:flex;align-items:center;'><div style='width:15px;height:15px;background-color:{jmol_colors.get(e, '#CCCCCC')};margin-right:5px;border:1px solid black;'></div><span>{e}</span></div>"
                        for e in elems_legend]
                    st.markdown(
                        f"<div style='display:flex;flex-wrap:wrap;align-items:center;justify-content:center;margin-top:10px;'>{''.join(legend_items)}</div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div style='display:flex;justify-content:center;align-items:center;border:1px solid #e0e0e0;border-radius:5px;height:520px;background-color:#f8f9fa;'>"
                        "<p style='color:#666;font-style:italic;'>3D visualization is disabled. Check the box above to enable it.</p>"
                        "</div>",
                        unsafe_allow_html=True
                    )

            with col_dl:
                st.markdown("##### Structure Info")
                cp = ase_to_visualize.get_cell_lengths_and_angles()
                vol = ase_to_visualize.get_volume()

                total_atoms = len(ase_to_visualize)
                element_counts = {}
                for atom in ase_to_visualize:
                    element = atom.symbol
                    element_counts[element] = element_counts.get(element, 0) + 1

                st.markdown(f"<b>Total Atoms:</b> {total_atoms}", unsafe_allow_html=True)

                element_info_lines = []
                for element, count in sorted(element_counts.items()):
                    percentage = (count / total_atoms) * 100
                    element_info_lines.append(f"{element}: {count} ({percentage:.1f}%)")

                element_info_text = "<br>".join(element_info_lines)
                st.markdown(f"<b>Composition:</b><br>{element_info_text}", unsafe_allow_html=True)

                st.markdown(
                    f"a={cp[0]:.4f} Å b={cp[1]:.4f} Å c={cp[2]:.4f} Å<br>α={cp[3]:.2f}° β={cp[4]:.2f}° γ={cp[5]:.2f}°<br>Vol={vol:.2f} Å³",
                    unsafe_allow_html=True)

                try:
                    sga = SpacegroupAnalyzer(pmg_to_visualize, symprec=0.1)
                    st.markdown(
                        f"<b>Space Group:</b> {sga.get_space_group_symbol()} ({sga.get_space_group_number()})",
                        unsafe_allow_html=True)
                except Exception:
                    st.markdown("<b>Space Group:</b> N/A or low symmetry", unsafe_allow_html=True)

                st.markdown("##### Download Options")
                fmt_dl = st.radio("Format", ("CIF", "VASP", "LAMMPS", "XYZ"), horizontal=True, key="dl_fmt_radio")
                base_fn_dl = st.session_state.selected_file.split('.')[
                    0] if st.session_state.selected_file else "custom"
                applied_cell_fn = st.session_state.applied_cell_type_name
                sc_factors_fn = f"SC{st.session_state.applied_supercell_n_a}x{st.session_state.applied_supercell_n_b}x{st.session_state.applied_supercell_n_c}"
                suffix_fn = f"{applied_cell_fn}_{sc_factors_fn}" if st.session_state.supercell_settings_applied else applied_cell_fn

                dl_content, dl_name, dl_mime = None, "structure", "text/plain"
                try:
                    if fmt_dl == "CIF":
                        dl_content = str(CifWriter(pmg_to_visualize, symprec=0.1, refine_struct=False))
                        dl_name = f"{base_fn_dl}_{suffix_fn}.cif"
                        dl_mime = "chemical/x-cif"
                    elif fmt_dl == "VASP":
                        sio = StringIO()
                        colsss, colyyy = st.columns([1, 1])
                        with colsss:
                            use_fractional = st.checkbox("Output POSCAR with fractional coordinates",
                                                         value=True,
                                                         key="poscar_fractional")
                        with colyyy:
                            from ase.constraints import FixAtoms

                            use_selective_dynamics = st.checkbox("Include Selective dynamics (all atoms free)",
                                                                 value=False, key="poscar_sd")
                            if use_selective_dynamics:
                                constraint = FixAtoms(indices=[])
                                ase_to_visualize.set_constraint(constraint)

                        write(sio, ase_to_visualize, format="vasp",
                              direct=use_fractional, sort=True)
                        dl_content = sio.getvalue()
                        dl_name = f"{base_fn_dl}_{suffix_fn}.poscar"
                    elif fmt_dl == "LAMMPS":
                        st.markdown("**LAMMPS Export Options**")
                        lmp_col1, lmp_col2 = st.columns(2)
                        with lmp_col1:
                            atom_style = st.selectbox("Select atom_style", ["atomic", "charge", "full"], index=0,
                                                      key="lmp_style_dl")
                            units = st.selectbox("Select units", ["metal", "real", "si"], index=0, key="lmp_units_dl")
                        with lmp_col2:
                            include_masses = st.checkbox("Include atomic masses", value=True, key="lmp_masses_dl")
                            force_skew = st.checkbox("Force triclinic cell (skew)", value=False, key="lmp_skew_dl")

                        sio = StringIO()
                        write(sio, ase_to_visualize, format="lammps-data",
                              atom_style=atom_style, units=units, masses=include_masses, force_skew=force_skew)
                        dl_content = sio.getvalue()
                        dl_name = f"{base_fn_dl}_{suffix_fn}_{atom_style}.lmp"
                    elif fmt_dl == "XYZ":
                        lattice_vectors = pmg_to_visualize.lattice.matrix
                        cart_coords = []
                        elements = []
                        for site in pmg_to_visualize:
                            cart_coords.append(pmg_to_visualize.lattice.get_cartesian_coords(site.frac_coords))
                            elements.append(site.specie.symbol)

                        xyz_lines = []
                        xyz_lines.append(str(len(pmg_to_visualize)))

                        lattice_string = " ".join([f"{x:.6f}" for row in lattice_vectors for x in row])
                        properties = "Properties=species:S:1:pos:R:3"
                        comment_line = f'Lattice="{lattice_string}" {properties}'
                        xyz_lines.append(comment_line)

                        for element, coord in zip(elements, cart_coords):
                            line = f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}"
                            xyz_lines.append(line)

                        dl_content = "\n".join(xyz_lines)
                        dl_name = f"{base_fn_dl}_{suffix_fn}.xyz"
                        dl_mime = "chemical/x-xyz"

                    if dl_content:
                        st.download_button(f"Download {fmt_dl}", dl_content, dl_name, dl_mime,
                                           type="primary", key=f"dl_btn_{fmt_dl}")
                except Exception as e_dl:
                    st.error(f"Error generating {fmt_dl}: {e_dl}")

            if st.session_state.show_atomic_labels and st.session_state.show_3d_visualization:
                atom_info_data = []
                inv_cell_tbl = np.eye(3)
                det_cell = np.linalg.det(cell_viz)
                if det_cell > 1e-9: inv_cell_tbl = np.linalg.inv(cell_viz)
                for i, atom in enumerate(ase_to_visualize):
                    pos, frac = atom.position, np.dot(atom.position, inv_cell_tbl)
                    atom_info_data.append(
                        {"Atom": f"{atom.symbol}{i}", "X": f"{pos[0]:.3f}", "Y": f"{pos[1]:.3f}", "Z": f"{pos[2]:.3f}",
                         "Frac X": f"{frac[0]:.3f}", "Frac Y": f"{frac[1]:.3f}", "Frac Z": f"{frac[2]:.3f}"})
                if atom_info_data:
                    st.markdown("##### Atomic Positions")
                    st.dataframe(pd.DataFrame(atom_info_data), height=200)
        else:
            st.warning("No structure available for visualization. Upload/select a file.")
    else:
        if st.session_state.uploaded_files: st.warning(
            "Structure selected but not fully processed. Try re-selecting or re-uploading.")
else:
    st.info("👈 Upload crystal structure files from the sidebar to begin!")

st.markdown("<br>" * 4, unsafe_allow_html=True)



def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # in MB


memory_usage = get_memory_usage()
st.write(
    f"🔍 Current memory usage: **{memory_usage:.2f} MB**. We are now using free hosting by Streamlit Community Cloud servis, which has a limit for RAM memory of 2.6 GBs. For more extensive computations, please compile the application locally from the [GitHub](https://github.com/bracerino/xrdlicious).")

st.markdown("""

### Acknowledgments

This project uses several open-source tools and datasets. We gratefully acknowledge their authors: **[Matminer](https://github.com/hackingmaterials/matminer)** Licensed under the [Modified BSD License](https://github.com/hackingmaterials/matminer/blob/main/LICENSE). **[Pymatgen](https://github.com/materialsproject/pymatgen)** Licensed under the [MIT License](https://github.com/materialsproject/pymatgen/blob/master/LICENSE).
 **[ASE (Atomic Simulation Environment)](https://gitlab.com/ase/ase)** Licensed under the [GNU Lesser General Public License (LGPL)](https://gitlab.com/ase/ase/-/blob/master/COPYING.LESSER). **[Py3DMol](https://github.com/avirshup/py3dmol/tree/master)** Licensed under the [BSD-style License](https://github.com/avirshup/py3dmol/blob/master/LICENSE.txt). **[Materials Project](https://next-gen.materialsproject.org/)** Data from the Materials Project is made available under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). **[AFLOW](http://aflow.org)** Licensed under the [GNU General Public License (GPL)](https://www.gnu.org/licenses/gpl-3.0.html)
 **[Crystallographic Open Database (COD)](https://www.crystallography.net/cod/)** under the CC0 license.
""")
