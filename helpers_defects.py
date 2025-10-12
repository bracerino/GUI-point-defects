# import pkg_resources
# installed_packages = sorted([(d.project_name, d.version) for d in pkg_resources.working_set])
# st.subheader("Installed Python Modules")
# for package, version in installed_packages:
#    st.write(f"{package}=={version}")

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from matminer.featurizers.structure import PartialRadialDistributionFunction
import numpy as np
from ase.io import write
from scipy.spatial import cKDTree
from pymatgen.io.ase import AseAtomsAdaptor
import streamlit.components.v1 as components
import py3Dmol
from io import StringIO
import pandas as pd
import os
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import io
from pymatgen.core import Structure, Element
from PIL import Image
from pymatgen.transformations.standard_transformations import SupercellTransformation

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.diffraction.neutron import NDCalculator
from collections import defaultdict
from itertools import combinations
import streamlit.components.v1 as components
import py3Dmol
from io import StringIO
import pandas as pd
import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events
from pymatgen.core import Structure as PmgStructure
import matplotlib.colors as mcolors
import streamlit as st
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from math import cos, radians, sqrt
import io
import re
import spglib
from pymatgen.core import Structure
from aflow import search, K
from aflow import search  # ensure your file is not named aflow.py!
import aflow.keywords as AFLOW_K
import requests
from PIL import Image

# import aflow.keywords as K
from pymatgen.io.cif import CifWriter



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



import numpy as np
import itertools
import random
import time
from scipy.optimize import differential_evolution
import warnings


def swap_nearest_elements(structure_obj, el1_symbol, el2_symbol, n_swaps, max_distance, log_area=None,
                          random_seed=None):
    """
    Swaps a specified number of nearest pairs between two elements.

    Args:
        structure_obj (pymatgen.core.Structure): The input crystal structure.
        el1_symbol (str): Symbol of the first element (e.g., the interstitial).
        el2_symbol (str): Symbol of the second element (e.g., the host atom).
        n_swaps (int): The number of element pairs to swap.
        max_distance (float): Maximum distance (in Angstroms) to consider for "nearest" pairs.
        log_area (streamlit.delta_generator.DeltaGenerator, optional): Streamlit area for logging.
        random_seed (int, optional): Seed for random operations.

    Returns:
        pymatgen.core.Structure: The modified structure with elements swapped.
    """
    if log_area:
        log_area.info(
            f"Attempting to swap {n_swaps} nearest '{el1_symbol}' with '{el2_symbol}' within {max_distance:.2f} Å.")

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    new_structure = structure_obj.copy()

    el1_indices = [i for i, site in enumerate(new_structure.sites) if site.specie.symbol == el1_symbol]
    el2_indices = [i for i, site in enumerate(new_structure.sites) if site.specie.symbol == el2_symbol]

    if not el1_indices or not el2_indices:
        if log_area:
            log_area.warning(
                f"Could not find both '{el1_symbol}' and '{el2_symbol}' in the structure. No swaps performed.")
        return new_structure

    # Get Cartesian coordinates for both element types
    el1_coords_cart = np.array([new_structure.sites[i].coords for i in el1_indices])
    el2_coords_cart = np.array([new_structure.sites[i].coords for i in el2_indices])

    # Build k-d tree for faster nearest-neighbor search of el2 atoms
    tree_el2 = cKDTree(el2_coords_cart)

    potential_swaps = []  # List of (distance, el1_global_idx, el2_global_idx)

    # Find nearest el2 for each el1 within max_distance
    for i, el1_cart_coord in enumerate(el1_coords_cart):
        dist, idx = tree_el2.query(el1_cart_coord, k=1, distance_upper_bound=max_distance)

        # Check if a neighbor within max_distance was found
        if dist < float('inf') and dist <= max_distance:
            el1_global_idx = el1_indices[i]
            el2_global_idx = el2_indices[idx]
            potential_swaps.append((dist, el1_global_idx, el2_global_idx))

    if not potential_swaps:
        if log_area:
            log_area.warning(f"No '{el1_symbol}' and '{el2_symbol}' pairs found within {max_distance:.2f} Å to swap.")
        return new_structure

    # Sort by distance (nearest first) and take unique pairs
    potential_swaps.sort(key=lambda x: x[0])

    performed_swaps_count = 0
    used_el1_indices = set()
    used_el2_indices = set()

    # Iterate through potential swaps to select non-overlapping pairs
    for _, el1_global_idx, el2_global_idx in potential_swaps:
        if performed_swaps_count >= n_swaps:
            break

        if el1_global_idx not in used_el1_indices and el2_global_idx not in used_el2_indices:
            # Perform the swap on the new_structure
            el1_site = new_structure.sites[el1_global_idx]
            el2_site = new_structure.sites[el2_global_idx]

            new_structure.replace(el1_global_idx, el2_site.specie, el1_site.frac_coords)
            new_structure.replace(el2_global_idx, el1_site.specie, el2_site.frac_coords)

            used_el1_indices.add(el1_global_idx)
            used_el2_indices.add(el2_global_idx)
            performed_swaps_count += 1
            if log_area:
                log_area.write(
                    f"  Swapped atom {el1_global_idx} ({el1_symbol}) with atom {el2_global_idx} ({el2_symbol}).")

    if log_area:
        log_area.success(f"Successfully performed {performed_swaps_count} element swaps.")

    return new_structure

def select_cluster_points(structure_obj, target_element_symbol, n_points, cluster_radius, random_seed=None,
                          log_area=None):

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    if log_area:
        log_area.info(
            f"Selecting {n_points} '{target_element_symbol}' atoms for clustering with radius {cluster_radius} Å.")

    element_sites_indices = [i for i, site in enumerate(structure_obj.sites) if
                             site.specie.symbol == target_element_symbol]

    if not element_sites_indices or n_points == 0:
        if log_area: log_area.warning(f"No '{target_element_symbol}' atoms found or zero points requested.")
        return [], []

    if n_points > len(element_sites_indices):
        if log_area: log_area.warning(
            f"Requested {n_points} points but only {len(element_sites_indices)} are available. Selecting all.")
        n_points = len(element_sites_indices)
        return [structure_obj.sites[i].frac_coords for i in element_sites_indices], element_sites_indices

    all_cart_coords = np.array([site.coords for site in structure_obj.sites])

    periodic_coords = []
    periodic_indices = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                offset = np.dot([i, j, k], structure_obj.lattice.matrix)
                periodic_coords.append(all_cart_coords + offset)
                periodic_indices.extend(range(len(structure_obj)))

    periodic_coords = np.vstack(periodic_coords)

    tree = cKDTree(periodic_coords)
    seed_global_idx = random.choice(element_sites_indices)

    selected_indices = {seed_global_idx}

    if log_area:
        log_area.write(
            f"  Starting cluster with seed atom #{seed_global_idx} ({structure_obj.sites[seed_global_idx].specie.symbol}).")

    while len(selected_indices) < n_points:
        frontier_indices_for_query = list(selected_indices)

        all_neighbors_found = set()

        for frontier_idx in frontier_indices_for_query:
            frontier_cart_coord = structure_obj.sites[frontier_idx].coords


            neighbor_periodic_indices = tree.query_ball_point(frontier_cart_coord, r=cluster_radius)

            original_neighbor_indices = {periodic_indices[i] for i in neighbor_periodic_indices}
            all_neighbors_found.update(original_neighbor_indices)


        valid_candidates = [
            idx for idx in all_neighbors_found
            if idx in element_sites_indices and idx not in selected_indices
        ]

        if not valid_candidates:
            if log_area: log_area.warning(
                f"  Could not expand cluster further. Only {len(selected_indices)} atoms of the required {n_points} were found within the radius.")
            break

        min_dist_to_cluster = float('inf')
        next_best_idx = -1

        selected_cart_coords = np.array([structure_obj.sites[idx].coords for idx in selected_indices])

        for candidate_global_idx in valid_candidates:
            candidate_cart_coord = structure_obj.sites[candidate_global_idx].coords


            temp_delta_cart = selected_cart_coords - candidate_cart_coord
            min_dist_val = float('inf')
            for d_cart in temp_delta_cart:
                d_frac = structure_obj.lattice.get_fractional_coords(d_cart)
                d_frac -= np.round(d_frac)
                d_cart_pbc = structure_obj.lattice.get_cartesian_coords(d_frac)
                min_dist_val = min(min_dist_val, np.linalg.norm(d_cart_pbc))

            if min_dist_val < min_dist_to_cluster:
                min_dist_to_cluster = min_dist_val
                next_best_idx = candidate_global_idx

        if next_best_idx != -1:
            selected_indices.add(next_best_idx)
            if log_area:
                log_area.write(f"  Added atom #{next_best_idx} to cluster. Total: {len(selected_indices)}/{n_points}")
        else:
            break

    final_selected_indices = list(selected_indices)
    final_selected_frac_coords = [structure_obj.sites[i].frac_coords for i in final_selected_indices]

    if log_area: log_area.success(f"Finished selecting {len(final_selected_indices)} atoms for clustering.")

    return final_selected_frac_coords, final_selected_indices

def create_substitution_cluster(structure_obj, orig_el_symbol, sub_el_symbol, n_to_substitute,
                                cluster_radius, log_area=None, random_seed=None,
                                delete_non_clustered_original_elements: bool = False): #

    if log_area:
        log_area.info(
            f"Attempting to create a cluster of {n_to_substitute} '{sub_el_symbol}' atoms replacing '{orig_el_symbol}'.")
        if delete_non_clustered_original_elements:
            log_area.info(f"Enabled: All other '{orig_el_symbol}' atoms will be deleted.")

    if not sub_el_symbol:
        if log_area: log_area.error("Substitute element cannot be empty.")
        return structure_obj.copy()

    try:
        sub_element = Element(sub_el_symbol)
    except Exception:
        if log_area: log_area.error(f"Invalid substitute element: '{sub_el_symbol}'.")
        return structure_obj.copy()

    # Use the cluster selection logic to find the indices
    _, selected_global_indices = select_cluster_points(
        structure_obj, orig_el_symbol, n_to_substitute, cluster_radius, random_seed, log_area
    )

    if not selected_global_indices:
        if log_area: log_area.warning("No atoms were selected for clustering. Returning original structure.")
        return structure_obj.copy()

    # Create lists to build the new structure
    final_species = []
    final_frac_coords = []
    original_site_indices_to_keep = set()

    for i, site in enumerate(structure_obj.sites):
        if i in selected_global_indices:
            final_species.append(sub_element)
            final_frac_coords.append(site.frac_coords)
            original_site_indices_to_keep.add(i)
        elif delete_non_clustered_original_elements and site.specie.symbol == orig_el_symbol:
            if log_area: log_area.write(f"  Skipping (deleting) non-clustered {orig_el_symbol} at index {i}.")
            continue
        else:
            # Keep all other atoms as they are
            final_species.append(site.species)
            final_frac_coords.append(site.frac_coords)
            original_site_indices_to_keep.add(i)

    final_clustered_structure = Structure(
        lattice=structure_obj.lattice,
        species=final_species,
        coords=final_frac_coords,
        coords_are_cartesian=False
    )

    if log_area:
        if delete_non_clustered_original_elements:
            log_area.success(
                f"Successfully created a cluster of {len(selected_global_indices)} '{sub_el_symbol}' atoms "
                f"and deleted {len(structure_obj.sites) - len(final_clustered_structure)} non-clustered '{orig_el_symbol}' atoms. "
                f"Total atoms: {len(structure_obj.sites)} -> {len(final_clustered_structure)}.")
        else:
            log_area.success(
                f"Successfully created a cluster of {len(selected_global_indices)} '{sub_el_symbol}' atoms.")

    return final_clustered_structure


def apply_atomic_displacements(structure, displacement_mode="uniform", max_displacement=0.1,
                               std_displacement=0.05, coordinate_system="cartesian",
                               selected_elements=None, log_column=None, random_seed=None):
    import numpy as np
    import streamlit as st
    from pymatgen.core import Structure, Element

    # Convert random_seed to a plain Python int or None
    if random_seed is not None:
        try:
            random_seed = int(random_seed)
        except (TypeError, ValueError):
            random_seed = None

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    displaced_structure = structure.copy()

    atoms_to_displace = []
    for i, site in enumerate(displaced_structure.sites):
        if selected_elements is None or site.specie.symbol in selected_elements:
            atoms_to_displace.append(i)

    if log_column:
        with log_column:
            st.write(f"Displacing {len(atoms_to_displace)} of {len(displaced_structure)} atoms")
            st.write(f"Mode: {displacement_mode}, Coordinate system: {coordinate_system}")

    displaced_count = 0
    for atom_idx in atoms_to_displace:
        site = displaced_structure.sites[atom_idx]

        if displacement_mode == "uniform":
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            magnitude = np.random.uniform(0, max_displacement)
            displacement_vector = direction * magnitude
        else:  # gaussian
            displacement_vector = np.random.normal(0, std_displacement, 3)

        if coordinate_system == "cartesian":
            cart_coords = displaced_structure.lattice.get_cartesian_coords(site.frac_coords)
            new_cart_coords = cart_coords + displacement_vector
            new_frac_coords = displaced_structure.lattice.get_fractional_coords(new_cart_coords)
        else:  # fractional
            new_frac_coords = site.frac_coords + displacement_vector

        new_frac_coords = new_frac_coords % 1.0

        displaced_structure.replace(atom_idx, site.specie, new_frac_coords)
        displaced_count += 1

    if log_column:
        with log_column:
            st.success(f"Successfully displaced {displaced_count} atoms")
            if displacement_mode == "uniform":
                st.write(f"Maximum displacement: {max_displacement:.3f} Å")
            else:
                st.write(f"Std deviation: {std_displacement:.3f} Å")

    return displaced_structure

def find_octahedral_sites(structure, min_distance=1.5):
    from pymatgen.core import Element

    frac_coords = structure.frac_coords
    potential_sites = []

    for i in [0, 0.5]:
        for j in [0, 0.5]:
            for k in [0, 0.5]:
                if (i, j, k) != (0, 0, 0) and (i, j, k) != (0.5, 0.5, 0.5):
                    site = np.array([i, j, k])

                    min_dist_to_atoms = float('inf')
                    for atom_coord in frac_coords:
                        delta = site - atom_coord
                        delta = delta - np.round(delta)
                        dist = np.linalg.norm(np.dot(delta, structure.lattice.matrix))
                        min_dist_to_atoms = min(min_dist_to_atoms, dist)

                    if min_dist_to_atoms >= min_distance:
                        potential_sites.append(site)

    return potential_sites


class ManualInterstitialType:
    def __init__(self, sites, site_type="Octahedral"):
        self.site_type = site_type
        self.equivalent_sites = [type('obj', (object,), {'frac_coords': site})() for site in sites]
        if sites:
            self.site = type('obj', (object,), {'frac_coords': sites[0]})()
        else:
            self.site = None
def periodic_distance(coord1: np.ndarray, coord2: np.ndarray) -> float:
    delta = coord1 - coord2
    delta = delta - np.round(delta)
    return np.linalg.norm(delta)


def select_spaced_points_global_exact(frac_coords_list, n_points, mode="farthest", target_value=0.5, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    n_available = len(frac_coords_list)
    if n_available <= n_points:
        return frac_coords_list.copy(), list(range(n_available))

    from math import comb
    total_combinations = comb(n_available, n_points)
    st.info(f"Total combinations: {total_combinations}")
    if total_combinations > 1000000:
        raise ValueError(f"Too many combinations for exact algorithm. Use iterative or genetic instead.")

    best_selection = None
    best_min_distance = 0

    for combination in itertools.combinations(range(n_available), n_points):
        selected_coords = [frac_coords_list[i] for i in combination]

        min_pairwise_dist = float('inf')
        for i in range(len(selected_coords)):
            for j in range(i + 1, len(selected_coords)):
                dist = periodic_distance(selected_coords[i], selected_coords[j])
                min_pairwise_dist = min(min_pairwise_dist, dist)

        if min_pairwise_dist > best_min_distance:
            best_min_distance = min_pairwise_dist
            best_selection = combination

    if best_selection is None:
        best_selection = tuple(range(n_points))

    selected_coords = [frac_coords_list[i] for i in best_selection]
    return selected_coords, list(best_selection)


def select_spaced_points_genetic(frac_coords_list, n_points, mode="farthest", target_value=0.5, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    n_available = len(frac_coords_list)
    if n_available <= n_points:
        return frac_coords_list.copy(), list(range(n_available))

    coords_array = np.array(frac_coords_list)

    def objective_function(selection_weights):
        selected_indices = np.argsort(selection_weights)[-n_points:]
        min_dist = float('inf')
        for i in range(len(selected_indices)):
            for j in range(i + 1, len(selected_indices)):
                idx1, idx2 = selected_indices[i], selected_indices[j]
                dist = periodic_distance(coords_array[idx1], coords_array[idx2])
                min_dist = min(min_dist, dist)
        return -min_dist

    bounds = [(0, 1) for _ in range(n_available)]

    max_iter = min(1000, max(100, n_available * 10))
    pop_size = min(50, max(15, n_available // 2))

    result = differential_evolution(
        objective_function, bounds, maxiter=max_iter,
        popsize=pop_size, seed=random_seed, disp=False
    )

    best_weights = result.x
    selected_indices = np.argsort(best_weights)[-n_points:]
    selected_indices = sorted(selected_indices)

    selected_coords = [frac_coords_list[i] for i in selected_indices]
    return selected_coords, list(selected_indices)


def select_spaced_points_iterative(frac_coords_list, n_points, mode="farthest", target_value=0.5, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    n_available = len(frac_coords_list)
    if n_available <= n_points:
        return frac_coords_list.copy(), list(range(n_available))

    selected_coords, selected_indices = select_spaced_points_original(
        frac_coords_list, n_points, mode, target_value, random_seed
    )

    def calculate_min_pairwise_distance(indices):
        min_dist = float('inf')
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                dist = periodic_distance(frac_coords_list[indices[i]], frac_coords_list[indices[j]])
                min_dist = min(min_dist, dist)
        return min_dist

    current_min_dist = calculate_min_pairwise_distance(selected_indices)

    # Iterative improvement
    max_iterations = min(1000, n_available * 5)
    for iteration in range(max_iterations):
        improved = False
        unselected_indices = [i for i in range(n_available) if i not in selected_indices]

        for sel_idx in range(len(selected_indices)):
            for unsel_idx in unselected_indices:
                new_selection = selected_indices.copy()
                new_selection[sel_idx] = unsel_idx
                new_min_dist = calculate_min_pairwise_distance(new_selection)

                if new_min_dist > current_min_dist:
                    selected_indices = new_selection
                    current_min_dist = new_min_dist
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    selected_coords = [frac_coords_list[i] for i in selected_indices]
    return selected_coords, selected_indices


def select_spaced_points_adaptive(frac_coords_list, n_points, mode="farthest", target_value=0.5, random_seed=None):
    n_available = len(frac_coords_list)

    if n_available <= 12:
        return select_spaced_points_global_exact(frac_coords_list, n_points, mode, target_value, random_seed)
    elif n_available <= 50:
        return select_spaced_points_iterative(frac_coords_list, n_points, mode, target_value, random_seed)
    elif n_available <= 200:
        return select_spaced_points_genetic(frac_coords_list, n_points, mode, target_value, random_seed)
    else:
        return select_spaced_points_original(frac_coords_list, n_points, mode, target_value, random_seed)


def select_spaced_points(frac_coords_list, n_points, mode, target_value=0.5, random_seed=None):
    if not frac_coords_list or n_points == 0:
        return [], []

    try:
        algorithm = st.session_state.get('point_selection_algorithm', 'original')
    except:
        algorithm = 'original'
    if algorithm == "original":
        return select_spaced_points_original(frac_coords_list, n_points, mode, target_value, random_seed)

    elif algorithm == "iterative":
        return select_spaced_points_iterative(frac_coords_list, n_points, mode, target_value, random_seed)

    elif algorithm == "genetic":
        return select_spaced_points_genetic(frac_coords_list, n_points, mode, target_value, random_seed)

    elif algorithm == "global_exact":
        return select_spaced_points_global_exact(frac_coords_list, n_points, mode, target_value, random_seed)

    elif algorithm == "adaptive":
        return select_spaced_points_adaptive(frac_coords_list, n_points, mode, target_value, random_seed)

    else:
        return select_spaced_points_original(frac_coords_list, n_points, mode, target_value, random_seed)

def get_formula_type(formula):
    elements = []
    counts = []

    import re
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)

    for element, count in matches:
        elements.append(element)
        counts.append(int(count) if count else 1)

    if len(elements) == 1:
        return "A"

    elif len(elements) == 2:
        # Binary compounds
        if counts[0] == 1 and counts[1] == 1:
            return "AB"
        elif counts[0] == 1 and counts[1] == 2:
            return "AB2"
        elif counts[0] == 2 and counts[1] == 1:
            return "A2B"
        elif counts[0] == 1 and counts[1] == 3:
            return "AB3"
        elif counts[0] == 3 and counts[1] == 1:
            return "A3B"
        elif counts[0] == 1 and counts[1] == 4:
            return "AB4"
        elif counts[0] == 4 and counts[1] == 1:
            return "A4B"
        elif counts[0] == 1 and counts[1] == 5:
            return "AB5"
        elif counts[0] == 5 and counts[1] == 1:
            return "A5B"
        elif counts[0] == 1 and counts[1] == 6:
            return "AB6"
        elif counts[0] == 6 and counts[1] == 1:
            return "A6B"
        elif counts[0] == 2 and counts[1] == 3:
            return "A2B3"
        elif counts[0] == 3 and counts[1] == 2:
            return "A3B2"
        elif counts[0] == 2 and counts[1] == 5:
            return "A2B5"
        elif counts[0] == 5 and counts[1] == 2:
            return "A5B2"
        elif counts[0] == 1 and counts[1] == 12:
            return "AB12"
        elif counts[0] == 12 and counts[1] == 1:
            return "A12B"
        elif counts[0] == 2 and counts[1] == 17:
            return "A2B17"
        elif counts[0] == 17 and counts[1] == 2:
            return "A17B2"
        elif counts[0] == 3 and counts[1] == 4:
            return "A3B4"
        else:
            return f"A{counts[0]}B{counts[1]}"

    elif len(elements) == 3:
        # Ternary compounds
        if counts[0] == 1 and counts[1] == 1 and counts[2] == 1:
            return "ABC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 3:
            return "ABC3"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 1:
            return "AB3C"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 1:
            return "A3BC"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 4:
            return "AB2C4"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 4:
            return "A2BC4"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 2:
            return "AB4C2"
        elif counts[0] == 2 and counts[1] == 4 and counts[2] == 1:
            return "A2B4C"
        elif counts[0] == 4 and counts[1] == 1 and counts[2] == 2:
            return "A4BC2"
        elif counts[0] == 4 and counts[1] == 2 and counts[2] == 1:
            return "A4B2C"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 1:
            return "AB2C"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 1:
            return "A2BC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 2:
            return "ABC2"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 4:
            return "ABC4"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 1:
            return "AB4C"
        elif counts[0] == 4 and counts[1] == 1 and counts[2] == 1:
            return "A4BC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 5:
            return "ABC5"
        elif counts[0] == 1 and counts[1] == 5 and counts[2] == 1:
            return "AB5C"
        elif counts[0] == 5 and counts[1] == 1 and counts[2] == 1:
            return "A5BC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 6:
            return "ABC6"
        elif counts[0] == 1 and counts[1] == 6 and counts[2] == 1:
            return "AB6C"
        elif counts[0] == 6 and counts[1] == 1 and counts[2] == 1:
            return "A6BC"
        elif counts[0] == 2 and counts[1] == 2 and counts[2] == 1:
            return "A2B2C"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 2:
            return "A2BC2"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 2:
            return "AB2C2"
        elif counts[0] == 3 and counts[1] == 2 and counts[2] == 1:
            return "A3B2C"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 2:
            return "A3BC2"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 2:
            return "AB3C2"
        elif counts[0] == 2 and counts[1] == 3 and counts[2] == 1:
            return "A2B3C"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 3:
            return "A2BC3"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 3:
            return "AB2C3"
        elif counts[0] == 3 and counts[1] == 3 and counts[2] == 1:
            return "A3B3C"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 3:
            return "A3BC3"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 3:
            return "AB3C3"
        elif counts[0] == 4 and counts[1] == 3 and counts[2] == 1:
            return "A4B3C"
        elif counts[0] == 4 and counts[1] == 1 and counts[2] == 3:
            return "A4BC3"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 3:
            return "AB4C3"
        elif counts[0] == 3 and counts[1] == 4 and counts[2] == 1:
            return "A3B4C"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 4:
            return "A3BC4"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 4:
            return "AB3C4"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 4:
            return "ABC6"
        elif counts[0] == 2 and counts[1] == 2 and counts[2] == 7:
            return "A2B2C7"
        else:
            return f"A{counts[0]}B{counts[1]}C{counts[2]}"

    elif len(elements) == 4:
        # Quaternary compounds
        if counts[0] == 1 and counts[1] == 1 and counts[2] == 1 and counts[3] == 1:
            return "ABCD"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 1 and counts[3] == 3:
            return "ABCD3"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 3 and counts[3] == 1:
            return "ABC3D"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 1 and counts[3] == 1:
            return "AB3CD"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 1 and counts[3] == 1:
            return "A3BCD"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 1 and counts[3] == 4:
            return "ABCD4"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 4 and counts[3] == 1:
            return "ABC4D"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 1 and counts[3] == 1:
            return "AB4CD"
        elif counts[0] == 4 and counts[1] == 1 and counts[2] == 1 and counts[3] == 1:
            return "A4BCD"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 1 and counts[3] == 4:
            return "AB2CD4"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 1 and counts[3] == 4:
            return "A2BCD4"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 2 and counts[3] == 4:
            return "ABC2D4"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 4 and counts[3] == 1:
            return "AB2C4D"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 4 and counts[3] == 1:
            return "A2BC4D"
        elif counts[0] == 2 and counts[1] == 4 and counts[2] == 1 and counts[3] == 1:
            return "A2B4CD"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 1 and counts[3] == 1:
            return "A2BCD"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 1 and counts[3] == 1:
            return "AB2CD"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 2 and counts[3] == 1:
            return "ABC2D"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 1 and counts[3] == 2:
            return "ABCD2"
        elif counts[0] == 3 and counts[1] == 2 and counts[2] == 1 and counts[3] == 1:
            return "A3B2CD"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 2 and counts[3] == 1:
            return "A3BC2D"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 1 and counts[3] == 2:
            return "A3BCD2"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 2 and counts[3] == 1:
            return "AB3C2D"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 1 and counts[3] == 2:
            return "AB3CD2"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 3 and counts[3] == 2:
            return "ABC3D2"
        elif counts[0] == 2 and counts[1] == 3 and counts[2] == 1 and counts[3] == 1:
            return "A2B3CD"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 3 and counts[3] == 1:
            return "A2BC3D"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 1 and counts[3] == 3:
            return "A2BCD3"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 3 and counts[3] == 1:
            return "AB2C3D"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 1 and counts[3] == 3:
            return "AB2CD3"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 2 and counts[3] == 3:
            return "ABC2D3"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 1 and counts[3] == 6:
            return "A1B4C1D6"
        elif counts[0] == 5 and counts[1] == 3 and counts[2] == 1 and counts[3] == 13:
            return "A5B3C1D13"
        elif counts[0] == 2 and counts[1] == 2 and counts[2] == 4 and counts[3] == 9:
            return "A2B2C4D9"

        elif counts == [3, 2, 1, 4]:  # Garnet-like: Ca3Al2Si3O12
            return "A3B2C1D4"
        else:
            return f"A{counts[0]}B{counts[1]}C{counts[2]}D{counts[3]}"

    elif len(elements) == 5:
        # Five-element compounds (complex minerals like apatite)
        if counts == [1, 1, 1, 1, 1]:
            return "ABCDE"
        elif counts == [10, 6, 2, 31, 1]:  # Apatite-like: Ca10(PO4)6(OH)2
            return "A10B6C2D31E"
        elif counts == [5, 3, 13, 1, 1]:  # Simplified apatite: Ca5(PO4)3OH
            return "A5B3C13DE"
        elif counts == [5, 3, 13, 1, 1]:  # Simplified apatite: Ca5(PO4)3OH
            return "A5B3C13"
        elif counts == [3, 2, 3, 12, 1]:  # Garnet-like: Ca3Al2Si3O12
            return "A3B2C3D12E"

        else:
            return f"A{counts[0]}B{counts[1]}C{counts[2]}D{counts[3]}E{counts[4]}"

    elif len(elements) == 6:
        # Six-element compounds (very complex minerals)
        if counts == [1, 1, 1, 1, 1, 1]:
            return "ABCDEF"
        elif counts == [1, 1, 2, 6, 1, 1]:  # Complex silicate-like
            return "ABC2D6EF"
        else:
            # For 6+ elements, use a more compact notation
            element_count_pairs = []
            for i, count in enumerate(counts):
                element_letter = chr(65 + i)  # A, B, C, D, E, F, ...
                if count == 1:
                    element_count_pairs.append(element_letter)
                else:
                    element_count_pairs.append(f"{element_letter}{count}")
            return "".join(element_count_pairs)

    else:
        if len(elements) <= 10:
            element_count_pairs = []
            for i, count in enumerate(counts):
                element_letter = chr(65 + i)  # A, B, C, D, E, F, G, H, I, J
                if count == 1:
                    element_count_pairs.append(element_letter)
                else:
                    element_count_pairs.append(f"{element_letter}{count}")
            return "".join(element_count_pairs)
        else:
            return "Complex"

def check_structure_size_and_warn(structure, structure_name="structure"):
    n_atoms = len(structure)

    if n_atoms > 75:
        st.info(f"ℹ️ **Structure Notice**: {structure_name} contains a large number of **{n_atoms} atoms**. "
                f"Calculations may take longer depending on selected parameters. Please be careful to "
                f"not consume much memory, we are hosted on a free server. 😊")
        return "moderate"
    else:
        return "small"


SPACE_GROUP_SYMBOLS = {
    1: "P1", 2: "P-1", 3: "P2", 4: "P21", 5: "C2", 6: "Pm", 7: "Pc", 8: "Cm", 9: "Cc", 10: "P2/m",
    11: "P21/m", 12: "C2/m", 13: "P2/c", 14: "P21/c", 15: "C2/c", 16: "P222", 17: "P2221", 18: "P21212", 19: "P212121", 20: "C2221",
    21: "C222", 22: "F222", 23: "I222", 24: "I212121", 25: "Pmm2", 26: "Pmc21", 27: "Pcc2", 28: "Pma2", 29: "Pca21", 30: "Pnc2",
    31: "Pmn21", 32: "Pba2", 33: "Pna21", 34: "Pnn2", 35: "Cmm2", 36: "Cmc21", 37: "Ccc2", 38: "Amm2", 39: "Aem2", 40: "Ama2",
    41: "Aea2", 42: "Fmm2", 43: "Fdd2", 44: "Imm2", 45: "Iba2", 46: "Ima2", 47: "Pmmm", 48: "Pnnn", 49: "Pccm", 50: "Pban",
    51: "Pmma", 52: "Pnna", 53: "Pmna", 54: "Pcca", 55: "Pbam", 56: "Pccn", 57: "Pbcm", 58: "Pnnm", 59: "Pmmn", 60: "Pbcn",
    61: "Pbca", 62: "Pnma", 63: "Cmcm", 64: "Cmca", 65: "Cmmm", 66: "Cccm", 67: "Cmma", 68: "Ccca", 69: "Fmmm", 70: "Fddd",
    71: "Immm", 72: "Ibam", 73: "Ibca", 74: "Imma", 75: "P4", 76: "P41", 77: "P42", 78: "P43", 79: "I4", 80: "I41",
    81: "P-4", 82: "I-4", 83: "P4/m", 84: "P42/m", 85: "P4/n", 86: "P42/n", 87: "I4/m", 88: "I41/a", 89: "P422", 90: "P4212",
    91: "P4122", 92: "P41212", 93: "P4222", 94: "P42212", 95: "P4322", 96: "P43212", 97: "I422", 98: "I4122", 99: "P4mm", 100: "P4bm",
    101: "P42cm", 102: "P42nm", 103: "P4cc", 104: "P4nc", 105: "P42mc", 106: "P42bc", 107: "P42mm", 108: "P42cm", 109: "I4mm", 110: "I4cm",
    111: "I41md", 112: "I41cd", 113: "P-42m", 114: "P-42c", 115: "P-421m", 116: "P-421c", 117: "P-4m2", 118: "P-4c2", 119: "P-4b2", 120: "P-4n2",
    121: "I-4m2", 122: "I-4c2", 123: "I-42m", 124: "I-42d", 125: "P4/mmm", 126: "P4/mcc", 127: "P4/nbm", 128: "P4/nnc", 129: "P4/mbm", 130: "P4/mnc",
    131: "P4/nmm", 132: "P4/ncc", 133: "P42/mmc", 134: "P42/mcm", 135: "P42/nbc", 136: "P42/mnm", 137: "P42/mbc", 138: "P42/mnm", 139: "I4/mmm", 140: "I4/mcm",
    141: "I41/amd", 142: "I41/acd", 143: "P3", 144: "P31", 145: "P32", 146: "R3", 147: "P-3", 148: "R-3", 149: "P312", 150: "P321",
    151: "P3112", 152: "P3121", 153: "P3212", 154: "P3221", 155: "R32", 156: "P3m1", 157: "P31m", 158: "P3c1", 159: "P31c", 160: "R3m",
    161: "R3c", 162: "P-31m", 163: "P-31c", 164: "P-3m1", 165: "P-3c1", 166: "R-3m", 167: "R-3c", 168: "P6", 169: "P61", 170: "P65",
    171: "P62", 172: "P64", 173: "P63", 174: "P-6", 175: "P6/m", 176: "P63/m", 177: "P622", 178: "P6122", 179: "P6522", 180: "P6222",
    181: "P6422", 182: "P6322", 183: "P6mm", 184: "P6cc", 185: "P63cm", 186: "P63mc", 187: "P-6m2", 188: "P-6c2", 189: "P-62m", 190: "P-62c",
    191: "P6/mmm", 192: "P6/mcc", 193: "P63/mcm", 194: "P63/mmc", 195: "P23", 196: "F23", 197: "I23", 198: "P213", 199: "I213", 200: "Pm-3",
    201: "Pn-3", 202: "Fm-3", 203: "Fd-3", 204: "Im-3", 205: "Pa-3", 206: "Ia-3", 207: "P432", 208: "P4232", 209: "F432", 210: "F4132",
    211: "I432", 212: "P4332", 213: "P4132", 214: "I4132", 215: "P-43m", 216: "F-43m", 217: "I-43m", 218: "P-43n", 219: "F-43c", 220: "I-43d",
    221: "Pm-3m", 222: "Pn-3n", 223: "Pm-3n", 224: "Pn-3m", 225: "Fm-3m", 226: "Fm-3c", 227: "Fd-3m", 228: "Fd-3c", 229: "Im-3m", 230: "Ia-3d"
}


def identify_structure_type(structure):
    try:
        analyzer = SpacegroupAnalyzer(structure)
        spg_symbol = analyzer.get_space_group_symbol()
        spg_number = analyzer.get_space_group_number()
        crystal_system = analyzer.get_crystal_system()

        formula = structure.composition.reduced_formula
        formula_type = get_formula_type(formula)
       # print("------")
        print(formula)
       # print(formula_type)
        #print(spg_number)
        if spg_number in STRUCTURE_TYPES and spg_number == 62 and formula_type in STRUCTURE_TYPES[spg_number] and formula == "CaCO3":
           # print("YES")
           # print(spg_number)
           # print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**Aragonite (CaCO3)**"
        elif spg_number in STRUCTURE_TYPES and spg_number ==167 and formula_type in STRUCTURE_TYPES[spg_number] and formula == "CaCO3":
          #  print("YES")
          # print(spg_number)
           # print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**Calcite (CaCO3)**"
        elif spg_number in STRUCTURE_TYPES and spg_number ==227 and formula_type in STRUCTURE_TYPES[spg_number] and formula == "SiO2":
           # print("YES")
           # print(spg_number)
           # print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**β - Cristobalite (SiO2)**"
        elif formula == "C" and spg_number in STRUCTURE_TYPES and spg_number ==194 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**Graphite**"
        elif formula == "MoS2" and spg_number in STRUCTURE_TYPES and spg_number ==194 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**MoS2 Type**"
        elif formula == "NiAs" and spg_number in STRUCTURE_TYPES and spg_number ==194 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**Nickeline (NiAs)**"
        elif formula == "ReO3" and spg_number in STRUCTURE_TYPES and spg_number ==221 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**ReO3 type**"
        elif formula == "TlI" and spg_number in STRUCTURE_TYPES and spg_number ==63 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**TlI structure**"
        elif spg_number in STRUCTURE_TYPES and formula_type in STRUCTURE_TYPES[
            spg_number]:
           # print("YES")
            structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**{structure_type}**"

        pearson = f"{crystal_system[0]}{structure.num_sites}"
        return f"**{crystal_system.capitalize()}** (Formula: {formula_type}, Pearson: {pearson})"

    except Exception as e:
        return f"Error identifying structure: {str(e)}"
STRUCTURE_TYPES = {
    # Cubic Structures
    225: {  # Fm-3m
        "A": "FCC (Face-centered cubic)",
        "AB": "Rock Salt (NaCl)",
        "AB2": "Fluorite (CaF2)",
        "A2B": "Anti-Fluorite",
        "AB3": "Cu3Au (L1₂)",
        "A3B": "AuCu3 type",
        "ABC": "Half-Heusler (C1b)",
        "AB6": "K2PtCl6 (cubic antifluorite)",
    },
    92: {
        "AB2": "α-Cristobalite (SiO2)"
    },
    229: {  # Im-3m
        "A": "BCC (Body-centered cubic)",
        "AB12": "NaZn13 type",
        "AB": "Tungsten carbide (WC)"
    },
    221: {  # Pm-3m
        "A": "Simple cubic (SC)",
        "AB": "Cesium Chloride (CsCl)",
        "ABC3": "Perovskite (Cubic, ABO3)",
        "AB3": "Cu3Au type",
        "A3B": "Cr3Si (A15)",
        #"AB6": "ReO3 type"
    },
    227: {  # Fd-3m
        "A": "Diamond cubic",

        "AB2": "Fluorite-like",
        "AB2C4": "Normal spinel",
        "A3B4": "Inverse spinel",
        "AB2C4": "Spinel",
        "A8B": "Gamma-brass",
        "AB2": "β - Cristobalite (SiO2)",
        "A2B2C7": "Pyrochlore"
    },
    55: {  # Pbca
        "AB2": "Brookite (TiO₂ polymorph)"
    },
    216: {  # F-43m
        "AB": "Zinc Blende (Sphalerite)",
        "A2B": "Antifluorite"
    },
    215: {  # P-43m
        "ABC3": "Inverse-perovskite",
        "AB4": "Half-anti-fluorite"
    },
    223: {  # Pm-3n
        "AB": "α-Mn structure",
        "A3B": "Cr3Si-type"
    },
    230: {  # Ia-3d
        "A3B2C1D4": "Garnet structure ((Ca,Mg,Fe)3(Al,Fe)2(SiO4)3)",
        "AB2": "Pyrochlore"
    },
    217: {  # I-43m
        "A12B": "α-Mn structure"
    },
    219: {  # F-43c
        "AB": "Sodium thallide"
    },
    205: {  # Pa-3
        "A2B": "Cuprite (Cu2O)",
        "AB6": "ReO3 structure",
        "AB2": "Pyrite (FeS2)",
    },
    156: {
        "AB2": "CdI2 type",
    },
    # Hexagonal Structures
    194: {  # P6_3/mmc
        "AB": "Wurtzite (high-T)",
        "AB2": "AlB2 type (hexagonal)",
        "A3B": "Ni3Sn type",
        "A3B": "DO19 structure (Ni3Sn-type)",
        "A": "Graphite (hexagonal)",
        "A": "HCP (Hexagonal close-packed)",
        #"AB2": "MoS2 type",
    },
    186: {  # P6_3mc
        "AB": "Wurtzite (ZnS)",
    },
    191: {  # P6/mmm


        "AB2": "AlB2 type",
        "AB5": "CaCu5 type",
        "A2B17": "Th2Ni17 type"
    },
    193: {  # P6_3/mcm
        "A3B": "Na3As structure",
        "ABC": "ZrBeSi structure"
    },
   # 187: {  # P-6m2
#
 #   },
    164: {  # P-3m1
        "AB2": "CdI2 type",
        "A": "Graphene layers"
    },
    166: {  # R-3m
        "A": "Rhombohedral",
        "A2B3": "α-Al2O3 type",
        "ABC2": "Delafossite (CuAlO2)"
    },
    160: {  # R3m
        "A2B3": "Binary tetradymite",
        "AB2": "Delafossite"
    },

    # Tetragonal Structures
    139: {  # I4/mmm
        "A": "Body-centered tetragonal",
        "AB": "β-Tin",
        "A2B": "MoSi2 type",
        "A3B": "Ni3Ti structure"
    },
    136: {  # P4_2/mnm
        "AB2": "Rutile (TiO2)"
    },
    123: {  # P4/mmm
        "AB": "γ-CuTi",
        "AB": "CuAu (L10)"
    },
    140: {  # I4/mcm
        "AB2": "Anatase (TiO2)",
        "A": "β-W structure"
    },
    141: {  # I41/amd
        "AB2": "Anatase (TiO₂)",
        "A": "α-Sn structure",
        "ABC4": "Zircon (ZrSiO₄)"
    },
    122: {  # P-4m2
        "ABC2": "Chalcopyrite (CuFeS2)"
    },
    129: {  # P4/nmm
        "AB": "PbO structure"
    },

    # Orthorhombic Structures
    62: {  # Pnma
        "ABC3": "Aragonite (CaCO₃)",
        "AB2": "Cotunnite (PbCl2)",
        "ABC3": "Perovskite (orthorhombic)",
        "A2B": "Fe2P type",
        "ABC3": "GdFeO3-type distorted perovskite",
        "A2BC4": "Olivine ((Mg,Fe)2SiO4)",
        "ABC4": "Barite (BaSO₄)"
    },
    63: {  # Cmcm
        "A": "α-U structure",
        "AB": "CrB structure",
        "AB2": "HgBr2 type"
    },
    74: {  # Imma
        "AB": "TlI structure",
    },
    64: {  # Cmca
        "A": "α-Ga structure"
    },
    65: {  # Cmmm
        "A2B": "η-Fe2C structure"
    },
    70: {  # Fddd
        "A": "Orthorhombic unit cell"
    },

    # Monoclinic Structures
    14: {  # P21/c
        "AB": "Monoclinic structure",
        "AB2": "Baddeleyite (ZrO2)",
        "ABC4": "Monazite (CePO4)"
    },
    12: {  # C2/m
        "A2B2C7": "Thortveitite (Sc2Si2O7)"
    },
    15: {  # C2/c
        "A1B4C1D6": "Gypsum (CaH4O6S)",
        "ABC6": "Gypsum (CaH4O6S)",
        "ABC4": "Scheelite (CaWO₄)",
        "ABC5": "Sphene (CaTiSiO₅)"
    },
    1: {
        "A2B2C4D9": "Kaolinite"
    },
    # Triclinic Structures
    2: {  # P-1
        "AB": "Triclinic structure",
        "ABC3": "Wollastonite (CaSiO3)",
    },

    # Other important structures
    99: {  # P4mm
        "ABCD3": "Tetragonal perovskite"
    },
    167: {  # R-3c
        "ABC3": "Calcite (CaCO3)",
        "A2B3": "Corundum (Al2O3)"
    },
    176: {  # P6_3/m
        "A10B6C2D31E": "Apatite (Ca10(PO4)6(OH)2)",
        "A5B3C1D13": "Apatite (Ca5(PO4)3OH",
        "A5B3C13": "Apatite (Ca5(PO4)3OH"
    },
    58: {  # Pnnm
        "AB2": "Marcasite (FeS2)"
    },
    11: {  # P21/m
        "A2B": "ThSi2 type"
    },
    72: {  # Ibam
        "AB2": "MoSi2 type"
    },
    198: {  # P213
        "AB": "FeSi structure",
        "A12": "β-Mn structure"
    },
    88: {  # I41/a
        "ABC4": "Scheelite (CaWO4)"
    },
    33: {  # Pna21
        "AB": "FeAs structure"
    },
    130: {  # P4/ncc
        "AB2": "Cristobalite (SiO2)"
    },
    152: {  # P3121
        "AB2": "Quartz (SiO2)"
    },
    200: {  # Pm-3
        "A3B3C": "Fe3W3C"
    },
    224: {  # Pn-3m
        "AB": "Pyrochlore-related",
        "A2B": "Cuprite (Cu2O)"
    },
    127: {  # P4/mbm
        "AB": "σ-phase structure",
        "AB5": "CaCu5 type"
    },
    148: {  # R-3
        "ABC3": "Calcite (CaCO₃)",
        "ABC3": "Ilmenite (FeTiO₃)",
        "ABCD3": "Dolomite",
    },
    69: {  # Fmmm
        "A": "β-W structure"
    },
    128: {  # P4/mnc
        "A3B": "Cr3Si (A15)"
    },
    206: {  # Ia-3
        "AB2": "Pyrite derivative",
        "AB2": "Pyrochlore (defective)",
        "A2B3": "Bixbyite"
    },
    212: {  # P4_3 32

        "A4B3": "Mn4Si3 type"
    },
    180: {
        "AB2": "β-quartz (SiO2)",
    },
    226: {  # Fm-3c
        "AB2": "BiF3 type"
    },
    196: {  # F23
        "AB2": "FeS2 type"
    },
    96: {
        "AB2": "α-Cristobalite (SiO2)"
    }

}


def get_full_conventional_structure_diffra(structure, symprec=1e-3):
    lattice = structure.lattice.matrix
    positions = structure.frac_coords

    species_list = [site.species for site in structure]
    species_to_type = {}
    type_to_species = {}
    type_index = 1

    types = []
    for sp in species_list:
        sp_tuple = tuple(sorted(sp.items()))  # make it hashable
        if sp_tuple not in species_to_type:
            species_to_type[sp_tuple] = type_index
            type_to_species[type_index] = sp
            type_index += 1
        types.append(species_to_type[sp_tuple])

    cell = (lattice, positions, types)

    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)

    std_lattice = dataset.std_lattice
    std_positions = dataset.std_positions
    std_types = dataset.std_types

    new_species_list = [type_to_species[t] for t in std_types]

    conv_structure = Structure(
        lattice=std_lattice,
        species=new_species_list,
        coords=std_positions,
        coords_are_cartesian=False
    )

    return conv_structure


def get_full_conventional_structure(structure, symprec=1e-3):
    # Create the spglib cell tuple: (lattice, fractional coords, atomic numbers)
    cell = (structure.lattice.matrix, structure.frac_coords,
            [max(site.species, key=site.species.get).number for site in structure])

    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    std_lattice = dataset['std_lattice']
    std_positions = dataset['std_positions']
    std_types = dataset['std_types']

    conv_structure = Structure(std_lattice, std_types, std_positions)
    return conv_structure


def rgb_color(color_tuple, opacity=0.8):
    r, g, b = [int(255 * x) for x in color_tuple]
    return f"rgba({r},{g},{b},{opacity})"


def load_structure(file_or_name):
    if isinstance(file_or_name, str):
        filename = file_or_name
    else:
        filename = file_or_name.name
        with open(filename, "wb") as f:
            f.write(file_or_name.getbuffer())
    if filename.lower().endswith(".cif"):
        mg_structure = PmgStructure.from_file(filename)
    elif filename.lower().endswith(".data"):
        filename = filename.replace(".data", ".lmp")
        from pymatgen.io.lammps.data import LammpsData
        mg_structure = LammpsData.from_file(filename, atom_style="atomic").structure
    elif filename.lower().endswith(".lmp"):
        from pymatgen.io.lammps.data import LammpsData
        mg_structure = LammpsData.from_file(filename, atom_style="atomic").structure
    else:
        atoms = read(filename)
        mg_structure = AseAtomsAdaptor.get_structure(atoms)
    return mg_structure


def lattice_same_conventional_vs_primitive(structure):
    try:
        analyzer = SpacegroupAnalyzer(structure)
        primitive = analyzer.get_primitive_standard_structure()
        conventional = analyzer.get_conventional_standard_structure()

        lattice_diff = np.abs(primitive.lattice.matrix - conventional.lattice.matrix)
        volume_diff = abs(primitive.lattice.volume - conventional.lattice.volume)

        if np.all(lattice_diff < 1e-3) and volume_diff < 1e-2:
            return True
        else:
            return False
    except Exception as e:
        return None  # Could not determine


def get_cod_entries(params):
    try:
        response = requests.get('https://www.crystallography.net/cod/result', params=params)
        if response.status_code == 200:
            results = response.json()
            return results  # Returns a list of entries
        else:
            st.error(f"COD search error: {response.status_code}")
            return []
    except Exception as e:
        st.write(
            "Error during connection to COD database. Probably reason is that the COD database server is currently down.")


def get_cif_from_cod(entry):
    file_url = entry.get('file')
    if file_url:
        response = requests.get(f"https://www.crystallography.net/cod/{file_url}.cif")
        if response.status_code == 200:
            return response.text
    return None


def get_structure_from_mp(mp_id):
    with MPRester(MP_API_KEY) as mpr:
        structure = mpr.get_structure_by_material_id(mp_id)
        return structure


from pymatgen.io.cif import CifParser


def get_structure_from_cif_url(cif_url):
    response = requests.get(f"https://www.crystallography.net/cod/{cif_url}.cif")
    if response.status_code == 200:
        #  writer = CifWriter(response.text, symprec=0.01)
        #  parser = CifParser.from_string(writer)
        #  structure = parser.get_structures(primitive=False)[0]
        return response.text
    else:
        raise ValueError(f"Failed to fetch CIF from URL: {cif_url}")


def get_cod_str(cif_content):
    parser = CifParser.from_str(cif_content)
    structure = parser.get_structures(primitive=False)[0]
    return structure


def add_box(view, cell, color='black', linewidth=2):
    a, b, c = np.array(cell[0]), np.array(cell[1]), np.array(cell[2])
    corners = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                corner = i * a + j * b + k * c
                corners.append(corner)
    edges = []
    for idx in range(8):
        i = idx & 1
        j = (idx >> 1) & 1
        k = (idx >> 2) & 1
        if i == 0:
            edges.append((corners[idx], corners[idx + 1]))
        if j == 0:
            edges.append((corners[idx], corners[idx + 2]))
        if k == 0:
            edges.append((corners[idx], corners[idx + 4]))
    for start, end in edges:
        view.addLine({
            'start': {'x': float(start[0]), 'y': float(start[1]), 'z': float(start[2])},
            'end': {'x': float(end[0]), 'y': float(end[1]), 'z': float(end[2])},
            'color': color,
            'linewidth': linewidth
        })
    arrow_radius = 0.04
    arrow_color = '#000000'
    for vec in [a, b, c]:
        view.addArrow({
            'start': {'x': 0, 'y': 0, 'z': 0},
            'end': {'x': float(vec[0]), 'y': float(vec[1]), 'z': float(vec[2])},
            'color': arrow_color,
            'radius': arrow_radius
        })
    offset = 0.3

    def add_axis_label(vec, label_val):
        norm = np.linalg.norm(vec)
        end = vec + offset * vec / (norm + 1e-6)
        view.addLabel(label_val, {
            'position': {'x': float(end[0]), 'y': float(end[1]), 'z': float(end[2])},
            'fontSize': 14,
            'fontColor': color,
            'showBackground': False
        })

    a_len = np.linalg.norm(a)
    b_len = np.linalg.norm(b)
    c_len = np.linalg.norm(c)
    add_axis_label(a, f"a = {a_len:.3f} Å")
    add_axis_label(b, f"b = {b_len:.3f} Å")
    add_axis_label(c, f"c = {c_len:.3f} Å")


# --- Structure Visualization ---
jmol_colors = {
    "H": "#FFFFFF",
    "He": "#D9FFFF",
    "Li": "#CC80FF",
    "Be": "#C2FF00",
    "B": "#FFB5B5",
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "Ne": "#B3E3F5",
    "Na": "#AB5CF2",
    "Mg": "#8AFF00",
    "Al": "#BFA6A6",
    "Si": "#F0C8A0",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "Ar": "#80D1E3",
    "K": "#8F40D4",
    "Ca": "#3DFF00",
    "Sc": "#E6E6E6",
    "Ti": "#BFC2C7",
    "V": "#A6A6AB",
    "Cr": "#8A99C7",
    "Mn": "#9C7AC7",
    "Fe": "#E06633",
    "Co": "#F090A0",
    "Ni": "#50D050",
    "Cu": "#C88033",
    "Zn": "#7D80B0",
    "Ga": "#C28F8F",
    "Ge": "#668F8F",
    "As": "#BD80E3",
    "Se": "#FFA100",
    "Br": "#A62929",
    "Kr": "#5CB8D1",
    "Rb": "#702EB0",
    "Sr": "#00FF00",
    "Y": "#94FFFF",
    "Zr": "#94E0E0",
    "Nb": "#73C2C9",
    "Mo": "#54B5B5",
    "Tc": "#3B9E9E",
    "Ru": "#248F8F",
    "Rh": "#0A7D8C",
    "Pd": "#006985",
    "Ag": "#C0C0C0",
    "Cd": "#FFD98F",
    "In": "#A67573",
    "Sn": "#668080",
    "Sb": "#9E63B5",
    "Te": "#D47A00",
    "I": "#940094",
    "Xe": "#429EB0",
    "Cs": "#57178F",
    "Ba": "#00C900",
    "La": "#70D4FF",
    "Ce": "#FFFFC7",
    "Pr": "#D9FFC7",
    "Nd": "#C7FFC7",
    "Pm": "#A3FFC7",
    "Sm": "#8FFFC7",
    "Eu": "#61FFC7",
    "Gd": "#45FFC7",
    "Tb": "#30FFC7",
    "Dy": "#1FFFC7",
    "Ho": "#00FF9C",
    "Er": "#00E675",
    "Tm": "#00D452",
    "Yb": "#00BF38",
    "Lu": "#00AB24",
    "Hf": "#4DC2FF",
    "Ta": "#4DA6FF",
    "W": "#2194D6",
    "Re": "#267DAB",
    "Os": "#266696",
    "Ir": "#175487",
    "Pt": "#D0D0E0",
    "Au": "#FFD123",
    "Hg": "#B8B8D0",
    "Tl": "#A6544D",
    "Pb": "#575961",
    "Bi": "#9E4FB5",
    "Po": "#AB5C00",
    "At": "#754F45",
    "Rn": "#428296",
    "Fr": "#420066",
    "Ra": "#007D00",
    "Ac": "#70ABFA",
    "Th": "#00BAFF",
    "Pa": "#00A1FF",
    "U": "#008FFF",
    "Np": "#0080FF",
    "Pu": "#006BFF",
    "Am": "#545CF2",
    "Cm": "#785CE3",
    "Bk": "#8A4FE3",
    "Cf": "#A136D4",
    "Es": "#B31FD4",
    "Fm": "#B31FBA",
    "Md": "#B30DA6",
    "No": "#BD0D87",
    "Lr": "#C70066",
    "Rf": "#CC0059",
    "Db": "#D1004F",
    "Sg": "#D90045",
    "Bh": "#E00038",
    "Hs": "#E6002E",
    "Mt": "#EB0026"
}

def apply_y_scale(y_values, scale_type):
    if scale_type == "Logarithmic":
        # Add 1 to avoid log(0) and return 0 for 0 values
        return np.log10(y_values + 1)
    elif scale_type == "Square Root":
        return np.sqrt(y_values)
    else:  # Linear
        return y_values


def convert_intensity_scale(intensity_values, scale_type):
    if intensity_values is None or len(intensity_values) == 0:
        return intensity_values

    converted = np.copy(intensity_values)
    min_positive = 1

    if scale_type == "Square Root":
        converted[converted < 0] = 0
        converted = np.sqrt(converted)
    elif scale_type == "Logarithmic":
        converted[converted <= 1] = 1
        converted = np.log10(converted)
    return converted


def convert_to_hill_notation(formula_input):
    import re
    formula_parts = formula_input.strip().split()
    elements_dict = {}

    for part in formula_parts:
        match = re.match(r'([A-Z][a-z]?)(\d*)', part)
        if match:
            element = match.group(1)
            count = match.group(2) if match.group(2) else ""
            elements_dict[element] = count

    hill_order = []
    if 'C' in elements_dict:
        if elements_dict['C']:
            hill_order.append(f"C{elements_dict['C']}")
        else:
            hill_order.append("C")
        del elements_dict['C']
    if 'H' in elements_dict:
        if elements_dict['H']:
            hill_order.append(f"H{elements_dict['H']}")
        else:
            hill_order.append("H")
        del elements_dict['H']

    for element in sorted(elements_dict.keys()):
        if elements_dict[element]:
            hill_order.append(f"{element}{elements_dict[element]}")
        else:
            hill_order.append(element)

    return " ".join(hill_order)

def sort_formula_alphabetically(formula_input):
    formula_parts = formula_input.strip().split()
    return " ".join(sorted(formula_parts))

MINERALS = {
    # Cubic structures
    225: {  # Fm-3m
        "Rock Salt (NaCl)": "Na Cl",
        "Fluorite (CaF2)": "Ca F2",
        "Anti-Fluorite (Li2O)": "Li2 O",
    },
    229: {  # Im-3m
        "BCC Iron": "Fe",
    },
    221: {  # Pm-3m
        "Perovskite (SrTiO3)": "Sr Ti O3",
        "ReO3 type": "Re O3",
        "Inverse-perovskite (Ca3TiN)": "Ca3 Ti N",
        "Cesium chloride (CsCl)": "Cs Cl"
    },
    227: {  # Fd-3m
        "Diamond": "C",

        "Normal spinel (MgAl2O4)": "Mg Al2 O4",
        "Inverse spinel (Fe3O4)": "Fe3 O4",
        "Pyrochlore (Ca2NbO7)": "Ca2 Nb2 O7",
        "β-Cristobalite (SiO2)": "Si O2"

    },
    216: {  # F-43m
        "Zinc Blende (ZnS)": "Zn S",
        "Half-anti-fluorite (Li4Ti)": "Li4 Ti"
    },
    215: {  # P-43m


    },
    230: {  # Ia-3d
        "Garnet (Ca3Al2Si3O12)": "Ca3 Al2 Si3 O12",
    },
    205: {  # Pa-3
        "Pyrite (FeS2)": "Fe S2",
    },
    224:{
        "Cuprite (Cu2O)": "Cu2 O",
    },
    # Hexagonal structures
    194: {  # P6_3/mmc
        "HCP Magnesium": "Mg",
        "Ni3Sn type": "Ni3 Sn",
        "Graphite": "C",
        "MoS2 type": "Mo S2",
        "Nickeline (NiAs)": "Ni As",
    },
    186: {  # P6_3mc
        "Wurtzite (ZnS)": "Zn S"
    },
    191: {  # P6/mmm


        "AlB2 type": "Al B2",
        "CaCu5 type": "Ca Cu5"
    },
    #187: {  # P-6m2
#
 #   },
    156: {
        "CdI2 type": "Cd I2",
    },
    164: {
    "CdI2 type": "Cd I2",
    },
    166: {  # R-3m
    "Delafossite (CuAlO2)": "Cu Al O2"
    },
    # Tetragonal structures
    139: {  # I4/mmm
        "β-Tin (Sn)": "Sn",
        "MoSi2 type": "Mo Si2"
    },
    136: {  # P4_2/mnm
        "Rutile (TiO2)": "Ti O2"
    },
    123: {  # P4/mmm
        "CuAu (L10)": "Cu Au"
    },
    141: {  # I41/amd
        "Anatase (TiO2)": "Ti O2",
        "Zircon (ZrSiO4)": "Zr Si O4"
    },
    122: {  # P-4m2
        "Chalcopyrite (CuFeS2)": "Cu Fe S2"
    },
    129: {  # P4/nmm
        "PbO structure": "Pb O"
    },

    # Orthorhombic structures
    62: {  # Pnma
        "Aragonite (CaCO3)": "Ca C O3",
        "Cotunnite (PbCl2)": "Pb Cl2",
        "Olivine (Mg2SiO4)": "Mg2 Si O4",
        "Barite (BaSO4)": "Ba S O4",
        "Perovskite (GdFeO3)": "Gd Fe O3"
    },
    63: {  # Cmcm
        "α-Uranium": "U",
        "CrB structure": "Cr B",
        "TlI structure": "Tl I",
    },
   # 74: {  # Imma
   #
   # },
    64: {  # Cmca
        "α-Gallium": "Ga"
    },

    # Monoclinic structures
    14: {  # P21/c
        "Baddeleyite (ZrO2)": "Zr O2",
        "Monazite (CePO4)": "Ce P O4"
    },
    206: {  # C2/m
        "Bixbyite (Mn2O3)": "Mn2 O3"
    },
    15: {  # C2/c
        "Gypsum (CaSO4·2H2O)": "Ca S H4 O6",
        "Scheelite (CaWO4)": "Ca W O4"
    },

    1: {
        "Kaolinite": "Al2 Si2 O9 H4"

    },
    # Triclinic structures
    2: {  # P-1
        "Wollastonite (CaSiO3)": "Ca Si O3",
        #"Kaolinite": "Al2 Si2 O5"
    },

    # Other important structures
    167: {  # R-3c
        "Calcite (CaCO3)": "Ca C O3",
        "Corundum (Al2O3)": "Al2 O3"
    },
    176: {  # P6_3/m
        "Apatite (Ca5(PO4)3OH)": "Ca5 P3 O13 H"
    },
    58: {  # Pnnm
        "Marcasite (FeS2)": "Fe S2"
    },
    198: {  # P213
        "FeSi structure": "Fe Si"
    },
    88: {  # I41/a
        "Scheelite (CaWO4)": "Ca W O4"
    },
    33: {  # Pna21
        "FeAs structure": "Fe As"
    },
    96: {  # P4/ncc
        "α-Cristobalite (SiO2)": "Si O2"
    },
    92: {
        "α-Cristobalite (SiO2)": "Si O2"
    },
    152: {  # P3121
        "Quartz (SiO2)": "Si O2"
    },
    148: {  # R-3
        "Ilmenite (FeTiO3)": "Fe Ti O3",
        "Dolomite (CaMgC2O6)": "Ca Mg C2 O6",
    },
    180: {  # P4_3 32
        "β-quartz (SiO2)": "Si O2"
    }
}

def show_xrdlicious_roadmap():
    st.markdown("""
### Roadmap
-------------------------------------------------------------------------------------------------------------------
#### Code optimization 

#### Wavelength Input: Energy Specification
* ⏳ Allow direct input of X-ray energy (keV) for synchrotron measurements, converting to wavelength automatically.

#### Improved Database Search
* ✅ Add search by keywords, space groups, ids... in database queries.

#### Expanded Correction Factors & Peak Shapes
* ⏳ Add more peak shape functions (Lorentzian, Pseudo-Voigt).
* ⏳ Introduce preferred orientation and basic absorption corrections.
* ⏳ Instrumental broadening - introduce Caglioti formula.
* ⏳ Calculate and apply peak shifts due to sample displacement error.

#### Enhanced Background Subtraction (Experimental Data)
* ⏳ Improve tools for background removal on uploaded experimental patterns.

#### Enhanced XRD Data Conversion
* ⏳ More accessible conversion interface - not hidden within the interactive plot.
* ⏳ Batch operations on multiple files at once (e.g., FDS/VDS, wavelength).

#### Basic Peak Fitting (Experimental Data)
* ⏳ Fitting: Advanced goal for fitting profiles or full patterns to refine parameters.
""")

def get_space_group_info(number):
    symbol = SPACE_GROUP_SYMBOLS.get(number, f"SG#{number}")
    return symbol

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


def select_spaced_points_original(frac_coords_list, n_points, mode, target_value=0.5, random_seed=None):
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

    if mode == "random":
        import random
        selected_indices = random.sample(range(n_available), n_points)

    elif mode == "farthest":
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


import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
import random
from typing import List, Tuple, Optional


def compute_periodic_distance_matrix_optimized(frac_coords, max_distance=None):

    n = len(frac_coords)
    if n > 10000 and max_distance is not None:
        from scipy.sparse import lil_matrix
        dist_matrix = lil_matrix((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                delta = frac_coords[i] - frac_coords[j]
                delta = delta - np.round(delta)
                dist = np.linalg.norm(delta)
                if dist <= max_distance:
                    dist_matrix[i, j] = dist_matrix[j, i] = dist
        return dist_matrix.tocsr()
    else:
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                delta = frac_coords[i] - frac_coords[j]
                delta = delta - np.round(delta)
                dist = np.linalg.norm(delta)
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        return dist_matrix


def select_spaced_points_fast_kdtree(frac_coords_list, n_points, mode,
                                     target_value=0.5, random_seed=None,
                                     min_distance_threshold=0.1):
    if not frac_coords_list or n_points == 0:
        return [], []

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    frac_coords_array = np.array(frac_coords_list)
    n_available = len(frac_coords_list)

    if n_available <= n_points:
        return frac_coords_list.copy(), list(range(n_available))
    if n_available > 50000:
        n_clusters = min(n_points * 10, n_available // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(frac_coords_array)

        # Select representative points from each cluster
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                # Randomly select one point from each cluster
                selected_idx = np.random.choice(cluster_indices)
                selected_indices.append(selected_idx)

        # Reduce to the pre-selected points
        frac_coords_array = frac_coords_array[selected_indices]
        available_indices = selected_indices
        n_available = len(available_indices)
    else:
        available_indices = list(range(n_available))

    extended_coords = []
    extended_indices = []

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                offset_coords = frac_coords_array + np.array([dx, dy, dz])
                extended_coords.append(offset_coords)
                extended_indices.extend(available_indices)

    extended_coords = np.vstack(extended_coords)
    tree = cKDTree(extended_coords)

    selected_indices = []

    if mode == "random":
        attempts = 0
        max_attempts = min(n_available * 5, 100000)

        while len(selected_indices) < n_points and attempts < max_attempts:
            attempts += 1
            candidate_idx = random.randint(0, n_available - 1)

            if len(selected_indices) == 0:
                selected_indices.append(available_indices[candidate_idx])
            else:
                candidate_coord = frac_coords_array[candidate_idx]

                distances, _ = tree.query(candidate_coord, k=len(selected_indices) + 1)
                min_dist_to_selected = np.min(distances[1:])  # Exclude self

                if min_dist_to_selected >= min_distance_threshold:
                    selected_indices.append(available_indices[candidate_idx])

    elif mode == "farthest":
        start_idx = random.randint(0, n_available - 1) if random_seed is not None else 0
        selected_indices.append(available_indices[start_idx])

        for _ in range(n_points - 1):
            remaining_indices = [i for i in range(n_available)
                                 if available_indices[i] not in selected_indices]
            if not remaining_indices:
                break

            best_idx = remaining_indices[0]
            best_min_dist = 0

            for candidate_idx in remaining_indices:
                candidate_coord = frac_coords_array[candidate_idx]
                min_dist_to_selected = float('inf')
                for sel_global_idx in selected_indices:
                    sel_local_idx = available_indices.index(sel_global_idx)
                    sel_coord = frac_coords_array[sel_local_idx]

                    delta = candidate_coord - sel_coord
                    delta = delta - np.round(delta)
                    dist = np.linalg.norm(delta)
                    min_dist_to_selected = min(min_dist_to_selected, dist)

                if min_dist_to_selected > best_min_dist:
                    best_min_dist = min_dist_to_selected
                    best_idx = candidate_idx

            selected_indices.append(available_indices[best_idx])

    elif mode == "nearest":
        start_idx = random.randint(0, n_available - 1) if random_seed is not None else 0
        selected_indices.append(available_indices[start_idx])

        for _ in range(n_points - 1):
            remaining_indices = [i for i in range(n_available)
                                 if available_indices[i] not in selected_indices]
            if not remaining_indices:
                break

            best_idx = remaining_indices[0]
            best_min_dist = float('inf')

            for candidate_idx in remaining_indices:
                candidate_coord = frac_coords_array[candidate_idx]
                min_dist_to_selected = float('inf')
                for sel_global_idx in selected_indices:
                    sel_local_idx = available_indices.index(sel_global_idx)
                    sel_coord = frac_coords_array[sel_local_idx]

                    delta = candidate_coord - sel_coord
                    delta = delta - np.round(delta)
                    dist = np.linalg.norm(delta)
                    min_dist_to_selected = min(min_dist_to_selected, dist)

                if min_dist_to_selected < best_min_dist:
                    best_min_dist = min_dist_to_selected
                    best_idx = candidate_idx

            selected_indices.append(available_indices[best_idx])

    else:  # moderate

        if n_available > 1000:

            kmeans = KMeans(n_clusters=n_points, random_state=random_seed, n_init=10)
            cluster_labels = kmeans.fit_predict(frac_coords_array)
            cluster_centers = kmeans.cluster_centers_

            for center in cluster_centers:
                distances = np.linalg.norm(frac_coords_array - center, axis=1)
                closest_idx = np.argmin(distances)
                if available_indices[closest_idx] not in selected_indices:
                    selected_indices.append(available_indices[closest_idx])
                if len(selected_indices) >= n_points:
                    break
        else:
            return select_spaced_points_original(frac_coords_list, n_points, mode,
                                                 target_value, random_seed)

    if n_available > 50000:
        selected_coords_out = [frac_coords_list[i] for i in selected_indices]
    else:
        selected_coords_out = [frac_coords_list[available_indices.index(i)]
                               for i in selected_indices]
        selected_indices = [available_indices.index(i) for i in selected_indices]

    return selected_coords_out, selected_indices


def select_spaced_points_hierarchical(frac_coords_list, n_points, mode,
                                      target_value=0.5, random_seed=None):
    if not frac_coords_list or n_points == 0:
        return [], []

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    n_available = len(frac_coords_list)

    if n_available <= n_points:
        return frac_coords_list.copy(), list(range(n_available))
    if n_available > 10000:
        subsample_size = min(10000, n_available)
        subsample_indices = np.random.choice(n_available, subsample_size, replace=False)
        subsample_coords = [frac_coords_list[i] for i in subsample_indices]

        selected_coords, selected_local_indices = select_spaced_points_fast_kdtree(
            subsample_coords, n_points, mode, target_value, random_seed)

        selected_indices = [subsample_indices[i] for i in selected_local_indices]

        return selected_coords, selected_indices
    else:
        return select_spaced_points_fast_kdtree(frac_coords_list, n_points, mode,
                                                target_value, random_seed)


def select_spaced_points_grid_based(frac_coords_list, n_points, mode,
                                    grid_resolution=10, random_seed=None):

    if not frac_coords_list or n_points == 0:
        return [], []

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    frac_coords_array = np.array(frac_coords_list)
    n_available = len(frac_coords_list)

    if n_available <= n_points:
        return frac_coords_list.copy(), list(range(n_available))

    grid_points = {}
    point_to_grid = {}

    for i, coord in enumerate(frac_coords_array):
        grid_cell = tuple(np.floor(coord * grid_resolution).astype(int))
        if grid_cell not in grid_points:
            grid_points[grid_cell] = []
        grid_points[grid_cell].append(i)
        point_to_grid[i] = grid_cell

    selected_indices = []

    if mode == "random":
        available_cells = list(grid_points.keys())
        random.shuffle(available_cells)

        for cell in available_cells:
            if len(selected_indices) >= n_points:
                break
            cell_points = grid_points[cell]
            selected_idx = random.choice(cell_points)
            selected_indices.append(selected_idx)

    elif mode in ["farthest", "nearest", "moderate"]:
        grid_representatives = []
        grid_rep_indices = []

        for cell, cell_points in grid_points.items():
            rep_idx = random.choice(cell_points)
            grid_representatives.append(frac_coords_list[rep_idx])
            grid_rep_indices.append(rep_idx)

        if len(grid_representatives) <= n_points:
            selected_coords = grid_representatives
            selected_indices = grid_rep_indices
        else:
            selected_coords, local_indices = select_spaced_points_fast_kdtree(
                grid_representatives, n_points, mode, random_seed=random_seed
            )
            selected_indices = [grid_rep_indices[i] for i in local_indices]

    selected_coords_out = [frac_coords_list[i] for i in selected_indices]
    return selected_coords_out, selected_indices




import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random


def insert_interstitials_ase_fast_optimized_v2(structure_obj, interstitial_element, n_interstitials,
                                               min_distance=2.0, grid_spacing=0.5, mode="random",
                                               min_interstitial_distance=1.0, log_area=None, random_seed=None):
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.core import Element

    if log_area:
        log_area.info(f"Optimized fast insertion: {n_interstitials} {interstitial_element} atoms...")

    new_structure = structure_obj.copy()

    try:
        ase_atoms = AseAtomsAdaptor.get_atoms(structure_obj)
        cell = ase_atoms.get_cell()
        positions = ase_atoms.get_positions()
        cell_lengths = ase_atoms.get_cell_lengths_and_angles()[:3]

        n_atoms = len(structure_obj)
        if n_atoms > 10000:
            adaptive_spacing = max(grid_spacing, grid_spacing * (n_atoms / 10000) ** 0.3)
            if log_area:
                log_area.info(
                    f"Large structure detected ({n_atoms} atoms). Adjusting grid spacing from {grid_spacing:.2f} to {adaptive_spacing:.2f} Å")
            grid_spacing = adaptive_spacing

        max_grid_points = 50000  # Configurable limit
        n_points = [int(length / grid_spacing) + 1 for length in cell_lengths]
        total_grid_points = np.prod(n_points)

        if total_grid_points > max_grid_points:
            scale_factor = (max_grid_points / total_grid_points) ** (1 / 3)
            n_points = [max(10, int(n * scale_factor)) for n in n_points]
            if log_area:
                log_area.info(f"Grid too large ({total_grid_points} points). Scaled to {np.prod(n_points)} points")

        if np.prod(n_points) > max_grid_points:
            return insert_interstitials_random_sampling(structure_obj, interstitial_element, n_interstitials,
                                                        min_distance, min_interstitial_distance, log_area, random_seed)

        if log_area:
            log_area.write(
                f"Creating optimized grid: {n_points[0]}×{n_points[1]}×{n_points[2]} = {np.prod(n_points)} points")

        grid_points = generate_grid_points_efficient(n_points)
        grid_cart = np.dot(grid_points, cell.array)


        tree = cKDTree(positions)
        min_distances_to_atoms = tree.query(grid_cart)[0]

        valid_indices = np.where(min_distances_to_atoms >= min_distance)[0]
        valid_points = grid_points[valid_indices]

        if log_area:
            log_area.write(f"Found {len(valid_points)} valid void sites (>{min_distance}Å from atoms)")

        if len(valid_points) == 0:
            if log_area: log_area.warning("No valid interstitial sites found")
            return new_structure

        selected_points = select_interstitial_sites_optimized(
            valid_points, n_interstitials, mode, min_interstitial_distance,
            cell, log_area, random_seed
        )

        for point in selected_points:
            new_structure.append(
                species=Element(interstitial_element),
                coords=point,
                coords_are_cartesian=False,
                validate_proximity=False
            )

        if log_area:
            log_area.success(f"Successfully inserted {len(selected_points)} {interstitial_element} atoms")

    except Exception as e:
        if log_area: log_area.error(f"Error in optimized interstitial insertion: {e}")
        return structure_obj

    return new_structure


def generate_grid_points_efficient(n_points):
    x = np.linspace(0, 1, n_points[0], endpoint=False)
    y = np.linspace(0, 1, n_points[1], endpoint=False)
    z = np.linspace(0, 1, n_points[2], endpoint=False)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return grid_points


def select_interstitial_sites_optimized(valid_points, n_interstitials, mode,
                                        min_interstitial_distance, cell, log_area, random_seed):
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    n_valid = len(valid_points)
    n_to_insert = min(n_interstitials, n_valid)

    if mode == "random":
        return select_random_with_distance_constraint(valid_points, n_to_insert,
                                                      min_interstitial_distance, cell, log_area)
    else:
        valid_points_list = [valid_points[i] for i in range(len(valid_points))]
        selected_coords, _ = select_spaced_points_optimized(valid_points_list, n_to_insert, mode, 0.5, random_seed)
        return selected_coords


def select_random_with_distance_constraint(valid_points, n_to_insert, min_distance, cell, log_area):
    selected_points = []

    if n_to_insert <= 0:
        return selected_points

    valid_cart = np.dot(valid_points, cell.array)

    first_idx = random.randint(0, len(valid_points) - 1)
    selected_points.append(valid_points[first_idx])
    selected_cart = [valid_cart[first_idx]]

    attempts = 0
    max_attempts = min(len(valid_points) * 5, 100000)

    while len(selected_points) < n_to_insert and attempts < max_attempts:
        attempts += 1
        candidate_idx = random.randint(0, len(valid_points) - 1)
        candidate_cart = valid_cart[candidate_idx]

        if len(selected_cart) > 0:
            distances = cdist([candidate_cart], selected_cart)[0]
            min_dist = np.min(distances)

            if min_dist >= min_distance:
                selected_points.append(valid_points[candidate_idx])
                selected_cart.append(candidate_cart)
        else:
            selected_points.append(valid_points[candidate_idx])
            selected_cart.append(candidate_cart)

    if log_area and len(selected_points) < n_to_insert:
        log_area.warning(f"Only found {len(selected_points)} sites with required spacing after {attempts} attempts")

    return selected_points


def insert_interstitials_random_sampling(structure_obj, interstitial_element, n_interstitials,
                                         min_distance, min_interstitial_distance, log_area, random_seed):
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.core import Element

    if log_area:
        log_area.info("Using random sampling method for very large structure")

    if random_seed is not None:
        np.random.seed(random_seed)

    new_structure = structure_obj.copy()
    ase_atoms = AseAtomsAdaptor.get_atoms(structure_obj)
    cell = ase_atoms.get_cell()
    positions = ase_atoms.get_positions()

    tree = cKDTree(positions)

    selected_points = []
    selected_cart = []
    attempts = 0
    max_attempts = n_interstitials * 1000

    while len(selected_points) < n_interstitials and attempts < max_attempts:
        attempts += 1

        random_frac = np.random.rand(3)
        random_cart = np.dot(random_frac, cell.array)


        dist_to_atoms = tree.query(random_cart)[0]
        if dist_to_atoms < min_distance:
            continue

        if len(selected_cart) > 0:
            distances = cdist([random_cart], selected_cart)[0]
            min_dist_to_selected = np.min(distances)
            if min_dist_to_selected < min_interstitial_distance:
                continue
        selected_points.append(random_frac)
        selected_cart.append(random_cart)

    for point in selected_points:
        new_structure.append(
            species=Element(interstitial_element),
            coords=point,
            coords_are_cartesian=False,
            validate_proximity=False
        )

    if log_area:
        log_area.info(
            f"Random sampling: inserted {len(selected_points)} of {n_interstitials} requested atoms after {attempts} attempts")

    return new_structure


def insert_interstitials_adaptive_grid(structure_obj, interstitial_element, n_interstitials,
                                       min_distance=2.0, initial_grid_spacing=0.5, mode="random",
                                       min_interstitial_distance=1.0, log_area=None, random_seed=None):
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.core import Element

    if log_area:
        log_area.info(f"Adaptive grid insertion: {n_interstitials} {interstitial_element} atoms...")

    new_structure = structure_obj.copy()

    try:
        ase_atoms = AseAtomsAdaptor.get_atoms(structure_obj)
        cell = ase_atoms.get_cell()
        positions = ase_atoms.get_positions()
        cell_lengths = ase_atoms.get_cell_lengths_and_angles()[:3]

        grid_spacings = [initial_grid_spacing * 2, initial_grid_spacing, initial_grid_spacing * 0.7]

        for grid_spacing in grid_spacings:
            n_points = [int(length / grid_spacing) + 1 for length in cell_lengths]
            total_points = np.prod(n_points)

            if log_area:
                log_area.write(f"Trying grid spacing {grid_spacing:.2f} Å ({total_points} points)")

            if total_points > 100000:  # Skip if too many points
                continue
            grid_points = generate_grid_points_efficient(n_points)
            grid_cart = np.dot(grid_points, cell.array)

            tree = cKDTree(positions)
            min_distances = tree.query(grid_cart)[0]
            valid_indices = np.where(min_distances >= min_distance)[0]

            n_valid = len(valid_indices)
            if log_area:
                log_area.write(f"Found {n_valid} valid sites")

            if n_valid >= n_interstitials * 2:
                valid_points = grid_points[valid_indices]
                selected_points = select_interstitial_sites_optimized(
                    valid_points, n_interstitials, mode, min_interstitial_distance,
                    cell, log_area, random_seed
                )


                for point in selected_points:
                    new_structure.append(
                        species=Element(interstitial_element),
                        coords=point,
                        coords_are_cartesian=False,
                        validate_proximity=False
                    )

                if log_area:
                    log_area.success(
                        f"Adaptive grid: inserted {len(selected_points)} atoms with spacing {grid_spacing:.2f} Å")

                return new_structure

        if log_area:
            log_area.warning("Grid method failed, falling back to random sampling")

        return insert_interstitials_random_sampling(structure_obj, interstitial_element, n_interstitials,
                                                    min_distance, min_interstitial_distance, log_area, random_seed)

    except Exception as e:
        if log_area: log_area.error(f"Error in adaptive grid insertion: {e}")
        return structure_obj


# Wrapper function that chooses the best method
def insert_interstitials_ase_fast(structure_obj, interstitial_element, n_interstitials,
                                   min_distance=2.0, grid_spacing=0.5, mode="random",
                                   min_interstitial_distance=1.0, log_area=None, random_seed=None):

    n_atoms = len(structure_obj)

    if log_area:
        log_area.info(f"Smart interstitial insertion for {n_atoms} atom structure")

    if n_atoms < 1000:
        return insert_interstitials_ase_fast_optimized_v2(structure_obj, interstitial_element, n_interstitials,
                                                          min_distance, grid_spacing, mode, min_interstitial_distance,
                                                          log_area, random_seed)
    elif n_atoms < 50000:
        # Medium structures: use adaptive grid
        return insert_interstitials_adaptive_grid(structure_obj, interstitial_element, n_interstitials,
                                                  min_distance, grid_spacing, mode, min_interstitial_distance,
                                                  log_area, random_seed)
    else:
        # Large structures: use random sampling
        return insert_interstitials_random_sampling(structure_obj, interstitial_element, n_interstitials,
                                                    min_distance, min_interstitial_distance, log_area, random_seed)


def insert_interstitials_into_structure(structure_obj, interstitial_element, n_interstitials,
                                        which_interstitial_type_idx=0, mode="farthest",
                                        clustering_tol_val=0.75, min_dist_val=0.5, target_value=0.5,
                                        log_area=None, random_seed=None, include_manual_sites=True):
    from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator

    if log_area: log_area.info(f"Attempting to insert {n_interstitials} of {interstitial_element}...")
    new_structure_int = structure_obj.copy()

    try:
        generator = VoronoiInterstitialGenerator(clustering_tol=clustering_tol_val, min_dist=min_dist_val)
        unique_interstitial_types = list(generator.generate(new_structure_int, "H"))

        if include_manual_sites:
            manual_sites = find_octahedral_sites(new_structure_int, min_distance=min_dist_val)
            if manual_sites:
                manual_type = ManualInterstitialType(manual_sites, site_type="Octahedral (Manual)")
                unique_interstitial_types.append(manual_type)
                if log_area:
                    log_area.write(f"Added manual octahedral sites as Type {len(unique_interstitial_types)}")

        if not unique_interstitial_types:
            if log_area: log_area.warning("No candidate sites found.")
            return new_structure_int

        if log_area:
            log_area.write(f"Found {len(unique_interstitial_types)} unique interstitial site types.")
            for i_type, interstitial_type_obj in enumerate(unique_interstitial_types):
                site_label = classify_interstitial_site(new_structure_int, interstitial_type_obj.site.frac_coords)
                type_name = getattr(interstitial_type_obj, 'site_type', 'Voronoi')
                log_area.write(
                    f"  Type {i_type + 1} ({type_name}): at {np.round(interstitial_type_obj.site.frac_coords, 3)}, {site_label}, with {len(interstitial_type_obj.equivalent_sites)} equivalent sites.")

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
                f"Invalid interstitial type index: {which_interstitial_type_idx}. Max is {len(unique_interstitial_types)}.")
            return new_structure_int

        if not frac_coords_to_consider:
            if log_area: log_area.warning("No interstitial sites available to select from.")
            return new_structure_int

        num_to_actually_insert = min(n_interstitials, len(frac_coords_to_consider))
        if num_to_actually_insert < n_interstitials and log_area:
            log_area.warning(
                f"Requested {n_interstitials} interstitials, but only {num_to_actually_insert} sites are available.")

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

    info_text = f"(**{atom_count} atoms)**\n"

    element_counts = {}
    for site in structure:
        element = site.specie.symbol
        element_counts[element] = element_counts.get(element, 0) + 1

    element_info = ", ".join([f"{elem}: {count}" for elem, count in sorted(element_counts.items())])
    info_text += f"**Elements:** ({element_info})\n\n"

    info_text += f"a={cell_params[0]:.3f} Å, b={cell_params[1]:.3f} Å, c={cell_params[2]:.3f} Å\n"
    info_text += f"α={cell_params[3]:.1f}°, β={cell_params[4]:.1f}°, γ={cell_params[5]:.1f}°\n"
    info_text += f"Vol={volume:.2f} Å³"

    return info_text



def insert_database(ELEMENTS):
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


def get_laue_group(space_group_number):
    laue_groups = {
        # Triclinic
        range(1, 3): "-1",
        # Monoclinic
        range(3, 16): "2/m",
        # Orthorhombic
        range(16, 75): "mmm",
        # Tetragonal
        range(75, 89): "4/m",
        range(89, 143): "4/mmm",
        # Trigonal
        range(143, 149): "-3",
        range(149, 168): "-3m",
        # Hexagonal
        range(168, 177): "6/m",
        range(177, 195): "6/mmm",
        # Cubic
        range(195, 207): "m-3",
        range(207, 231): "m-3m"
    }

    for sg_range, laue in laue_groups.items():
        if space_group_number in sg_range:
            return laue
    return "Unknown"


