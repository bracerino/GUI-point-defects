import streamlit as st
import psutil
st.set_page_config(
    page_title="XRDlicious submodule:  Point Defects Creation on Uploaded Crystal Structures (CIF, LMP, POSCAR, ...)",
    layout="wide"
)
# Remove top padding
st.markdown("""
    <style>
    .block-container {
        padding-top: 0rem;
    }
    </style>
""", unsafe_allow_html=True)
from helpers_defects import *

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from matminer.featurizers.structure import PartialRadialDistributionFunction
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.diffraction.neutron import NDCalculator
from collections import defaultdict
from itertools import combinations
import streamlit.components.v1 as components
import py3Dmol
from io import StringIO
import pandas as pd
import os
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

# Inject custom CSS for buttons.
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #0099ff;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 0.5em 1em;
        border: none;
        border-radius: 5px;
        height: 3em;
        width: 100%;
    }
    div.stButton > button:active,
    div.stButton > button:focus {
        background-color: #0099ff !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    div[data-testid="stDataFrameContainer"] table td {
         font-size: 22px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

components.html(
    """
    <head>
        <meta name="description" content="XRDlicious submodule:  Point Defects Creation on Uploaded Crystal Structures (CIF, LMP, POSCAR, ...)">
    </head>
    """,
    height=0,
)

st.markdown(
    "#### XRDlicious submodule:  Point Defects Creation on Uploaded Crystal Structures (CIF, LMP, POSCAR, ...)")
col1, col2 = st.columns([1.25, 1])

with col2:

    st.info(
        "🌀 Developed by [IMPLANT team](https://implant.fs.cvut.cz/). 📺 [Quick tutorial HERE.](https://www.youtube.com/watch?v=7ZgQ0fnR8dQ&ab_channel=Implantgroup)"
    )
with col1:
    st.info("Visit the main [XRDlicious](http://xrdlicious.com) page")


if "first_run_note" not in st.session_state:
    st.session_state["first_run_note"] = True

if st.session_state["first_run_note"] == True:
    colh1, colh2 = st.columns([1, 3])
    with colh1:
        image = Image.open("images/Rb.png")
        st.image(image)
    with colh2:
        st.info("""
        Upload your crystal structure file (CIF, VASP, LMP, XYZ (with cell information), ...) into the sidebar. Then you can set the supercell size and create randomized point defects (interstitials (Voronoi method), substitutes, vacancies)
        """)
    st.session_state["first_run_note"] = False


pattern_details = None



# st.divider()

# Add mode selection at the very beginning
st.sidebar.markdown("## 🍕 XRDlicious")
# mode = st.sidebar.radio("Select Mode", ["Basic", "Advanced"], index=0)
mode = "Advanced"



calc_mode = "**🔬 Structure Visualization**"

# Initialize session state keys if not already set.
if 'mp_options' not in st.session_state:
    st.session_state['mp_options'] = None
if 'selected_structure' not in st.session_state:
    st.session_state['selected_structure'] = None
if 'uploaded_files' not in st.session_state or st.session_state['uploaded_files'] is None:
    st.session_state['uploaded_files'] = []  # List to store multiple fetched structures

# Create two columns: one for search and one for structure selection and actions.
# st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <hr style="border: none; height: 6px; background-color: #3399ff; border-radius: 8px; margin: 20px 0;">
    """,
    unsafe_allow_html=True
)

col3, col1, col2 = st.columns(3)

if 'full_structures' not in st.session_state:
    st.session_state.full_structures = {}

st.sidebar.subheader("📁📤 Upload Your Structure Files")
uploaded_files_user_sidebar = st.sidebar.file_uploader(
    "Upload Structure Files (CIF, POSCAR, XSF, PW, CFG, ...):",
    type=None,
    accept_multiple_files=True,
    key="sidebar_uploader"
)


if uploaded_files_user_sidebar:
    for file in uploaded_files_user_sidebar:
        # Only add the file if it hasn't been processed before.
        if file.name not in st.session_state.full_structures:
            try:
                # Replace load_structure with your structure-parsing function.
                structure = load_structure(file)
                st.session_state.full_structures[file.name] = structure
            except Exception as e:
                st.error(f"Failed to parse {file.name}: {e}")

structure_cell_choice = st.sidebar.radio(
    "Structure Cell Type:",
    options=["Conventional Cell", "Primitive Cell (Niggli)", "Primitive Cell (LLL)", "Primitive Cell (no reduction)"],
    index=1,  # default to Conventional
    help="Choose whether to use the crystallographic Primitive Cell or the Conventional Unit Cell for the structures. For Primitive Cell, you can select whether to use Niggli or LLL (Lenstra–Lenstra–Lovász) "
         "lattice basis reduction algorithm to produce less skewed representation of the lattice. The MP database is using Niggli-reduced Primitive Cells."
)
convert_to_conventional = structure_cell_choice == "Conventional Cell"
pymatgen_prim_cell_niggli = structure_cell_choice == "Primitive Cell (Niggli)"
pymatgen_prim_cell_lll = structure_cell_choice == "Primitive Cell (LLL)"
pymatgen_prim_cell_no_reduce = structure_cell_choice == "Primitive Cell (no reduction)"








if uploaded_files_user_sidebar:
    uploaded_files = st.session_state['uploaded_files'] + uploaded_files_user_sidebar
    if 'full_structures' not in st.session_state:
        st.session_state.full_structures = {}
    for file in uploaded_files_user_sidebar:
        try:
            structure = load_structure(file)
            # Use file.name as the key (or modify to a unique identifier if needed)
            st.session_state['full_structures'][file.name] = structure
        except Exception as e:
            st.error(f"Failed to parse {file.name}: {e}")
else:
    uploaded_files = st.session_state['uploaded_files']

# Column 2: Select structure and add/download CIF.


if uploaded_files:
    st.write(f"📄 **{len(uploaded_files)} file(s) uploaded.**")



st.sidebar.markdown("### Final List of Structure Files:")
st.sidebar.write([f.name for f in uploaded_files])

st.sidebar.markdown("### 🗑️ Remove structure(s) added from online databases")

files_to_remove = []
for i, file in enumerate(st.session_state['uploaded_files']):
    col1, col2 = st.sidebar.columns([4, 1])
    col1.write(file.name)
    if col2.button("❌", key=f"remove_{i}"):
        files_to_remove.append(file)

if files_to_remove:
    for f in files_to_remove:
        st.session_state['uploaded_files'].remove(f)
    st.rerun()  # 🔁 Force Streamlit to rerun and refresh UI

if uploaded_files:
    species_set = set()
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        structure = load_structure(file)
        for atom in structure:
            if atom.is_ordered:
                species_set.add(atom.specie.symbol)
            else:
                for sp in atom.species:
                    species_set.add(sp.symbol)
    species_list = sorted(species_set)
    # st.subheader("📊 Detected Atomic Species")
    # st.write(", ".join(species_list))
else:
    species_list = []

if "current_structure" not in st.session_state:
    st.session_state["current_structure"] = None

if "original_structures" not in st.session_state:
    st.session_state["original_structures"] = {}

if "base_modified_structure" not in st.session_state:
     st.session_state["base_modified_structure"] = None

if calc_mode == "**🔬 Structure Visualization**":
    # show_structure = st.sidebar.checkbox("Show Structure Visualization Tool", value=True)
    show_structure = True
    if uploaded_files:
        if "helpful" not in st.session_state:
            st.session_state["helpful"] = False
        if show_structure:
            col_viz, col_download = st.columns(2)

            # Initialize session state keys if not set
            if "current_structure" not in st.session_state:
                st.session_state["current_structure"] = None
            if "selected_file" not in st.session_state:
                st.session_state["selected_file"] = None

            with col_viz:
                file_options = [file.name for file in uploaded_files]
                st.subheader("Select Structure for Interactive Visualization:")
                if len(file_options) > 5:
                    selected_file = st.selectbox("", file_options)
                else:
                    selected_file = st.radio("", file_options)
            visualize_partial = st.checkbox("Enable enhanced partial occupancy visualization", value=False)
            # Check if selection changed
            if selected_file != st.session_state["selected_file"]:
                st.session_state["current_structure"] = None
                # Update the session state
                st.session_state["selected_file"] = selected_file

                # Read the new structure
                try:
                    structure = read(selected_file)
                    mp_struct = AseAtomsAdaptor.get_structure(structure)
                except Exception as e:
                    mp_struct = load_structure(selected_file)

                # Save to session state
                st.session_state["current_structure"] = mp_struct
                st.session_state["original_structures"][selected_file] = mp_struct.copy()
            else:
                # Use the stored structure
                mp_struct = st.session_state["current_structure"]
            selected_file = st.session_state.get("selected_file")
            original_structures = st.session_state["original_structures"]
            if st.session_state.get("reset_requested", False):
                if selected_file in original_structures:
                    st.session_state["current_structure"] = original_structures[selected_file].copy()
                    st.session_state["supercell_n_a"] = 1
                    st.session_state["supercell_n_b"] = 1
                    st.session_state["supercell_n_c"] = 1
                    st.session_state["last_multiplier"] = (1, 1, 1)
                    st.session_state["helpful"] = False

                st.session_state["reset_requested"] = False
                st.rerun()

            selected_id = selected_file.split("_")[0]  # assumes filename like "mp-1234_FORMULA.cif"
            # print(st.session_state.get('full_structures', {}))
            # if 'full_structures' in st.session_state:
            # mp_struct = st.session_state.get('full_structures', {}).get(selected_file)
            # mp_struct = AseAtomsAdaptor.get_structure(structure)
            # mp_struct = st.session_state.get('uploaded_files', {}).get(selected_file.name)
            # enable_supercell = st.checkbox("Wish to Create Supercell?", value=False)
            # if st.session_state["current_structure"] is not None:
            #     mp_struct = st.session_state["current_structure"]
            # if "original_structure" in st.session_state:
            #    mp_struct = st.session_state["original_structure"].copy()
            show_3d_visualization = st.checkbox("Show 3D Structure Visualization", value=True,
                                                help="Toggle to show/hide the interactive 3D structure viewer")
            apply_symmetry_ops = st.checkbox("🔁 Apply symmetry-based standardization", value=True, disabled = True)
            show_atomic = st.checkbox("Show atomic positions (labels on structure and list in table)", value=False)
            if mp_struct:
                if apply_symmetry_ops:
                    if convert_to_conventional:
                        # analyzer = SpacegroupAnalyzer(mp_struct)
                        # converted_structure = analyzer.get_conventional_standard_structure()
                        converted_structure = get_full_conventional_structure(mp_struct, symprec=0.1)
                    elif pymatgen_prim_cell_niggli:
                        analyzer = SpacegroupAnalyzer(mp_struct)
                        converted_structure = analyzer.get_primitive_standard_structure()
                        converted_structure = converted_structure.get_reduced_structure(reduction_algo="niggli")
                    elif pymatgen_prim_cell_lll:
                        analyzer = SpacegroupAnalyzer(mp_struct)
                        converted_structure = analyzer.get_primitive_standard_structure()
                        converted_structure = converted_structure.get_reduced_structure(reduction_algo="LLL")
                    else:
                        analyzer = SpacegroupAnalyzer(mp_struct)
                        converted_structure = analyzer.get_primitive_standard_structure()
                else:
                  #  if "base_modified_structure" not in st.session_state:
                        # Initially, set the base to the current modified structure.
                  #      st.session_state["base_modified_structure"] = mp_struct.copy()
                    converted_structure = mp_struct
                structure = AseAtomsAdaptor.get_atoms(converted_structure)
                base_for_supercell = st.session_state["base_modified_structure"]

                colb1, colb2, colb3 = st.columns(3)

                if "selected_file" not in st.session_state or selected_file != st.session_state["selected_file"]:
                    # Update the selected file in session state.
                    st.session_state["selected_file"] = selected_file
                    try:
                        structure = read(selected_file)
                        mp_struct = AseAtomsAdaptor.get_structure(structure)
                    except Exception as e:
                        mp_struct = load_structure(selected_file)

                    # Save both the current (working) and the pristine original structures.
                    st.session_state["current_structure"] = mp_struct
                    st.session_state["original_structures"][selected_file] = mp_struct.copy()

                    # Reset the supercell parameters in session state.
                    st.session_state["supercell_n_a"] = 1
                    st.session_state["supercell_n_b"] = 1
                    st.session_state["supercell_n_c"] = 1

                    # Also reset the last_multiplier (if you're tracking changes)
                    st.session_state["last_multiplier"] = (1, 1, 1)

                # Later in your code, when drawing the supercell creation UI:
                if visualize_partial == False:
                    with colb1:
                        col1, col2, col3 = st.columns(3)
                        st.markdown("**Optional: Create Supercell**")

                        # Use session state keys so that these inputs can be reset when a new structure is selected.

                        if st.session_state["helpful"] != True:
                            with col1:
                                n_a = st.number_input("Repeat along a-axis", min_value=1, max_value=10,
                                                      value=st.session_state.get("supercell_n_a", 1), step=1,
                                                      key="supercell_n_a")
                            with col2:
                                n_b = st.number_input("Repeat along b-axis", min_value=1, max_value=10,
                                                      value=st.session_state.get("supercell_n_b", 1), step=1,
                                                      key="supercell_n_b")
                            with col3:
                                n_c = st.number_input("Repeat along c-axis", min_value=1, max_value=10,
                                                      value=st.session_state.get("supercell_n_c", 1), step=1,
                                                      key="supercell_n_c")
                        else:
                            with col1:
                                n_a = st.number_input("Repeat along a-axis", min_value=1, max_value=10,
                                                      value=1, step=1, key="supercell_n_a")
                            with col2:
                                n_b = st.number_input("Repeat along b-axis", min_value=1, max_value=10,
                                                      value=1, step=1, key="supercell_n_b")
                            with col3:
                                n_c = st.number_input("Repeat along c-axis", min_value=1, max_value=10,
                                                      value=1, step=1, key="supercell_n_c")
                        base_atoms = len(structure)
                        supercell_multiplier = n_a * n_b * n_c
                        total_atoms = base_atoms * supercell_multiplier
                        st.info(f"Structure will contain **{total_atoms} atoms**.")
                        supercell_structure = structure.copy()  # ASE Atoms object
                        supercell_pmg = converted_structure.copy()  # pymatgen Structure object

                    current_multiplier = (n_a, n_b, n_c)
                    if "last_multiplier" not in st.session_state:
                        st.session_state["last_multiplier"] = current_multiplier
                        update_supercell = True
                    elif st.session_state["last_multiplier"] != current_multiplier:
                        st.session_state["last_multiplier"] = current_multiplier
                        update_supercell = True
                    else:
                        update_supercell = False

                    # if (n_a, n_b, n_c) != (8, 1, 1):
                    if st.session_state["helpful"] != True:
                        from pymatgen.transformations.standard_transformations import SupercellTransformation

                        # if update_supercell == True:

                        supercell_matrix = [[n_a, 0, 0], [0, n_b, 0], [0, 0, n_c]]
                        transformer = SupercellTransformation(supercell_matrix)
                        supercell_pmg = transformer.apply_transformation(supercell_pmg)
                        mp_struct = supercell_pmg
                        structure = AseAtomsAdaptor.get_atoms(mp_struct)
                        st.session_state["current_structure"] = mp_struct
                    # if update_supercell == False:
                    #   mp_struct = supercell_pmg
                    #   structure = AseAtomsAdaptor.get_atoms(mp_struct)
                    #   st.session_state["current_structure"] = mp_struct
                    else:
                        # If supercell not enabled, just use the original structure
                        supercell_structure = structure.copy()
                        supercell_pmg = converted_structure.copy()
                        mp_struct = supercell_pmg
                        structure = AseAtomsAdaptor.get_atoms(mp_struct)
                        st.session_state["helpful"] = False

                    from pymatgen.core import Structure, Element

                    with colb2:
                        def wrap_coordinates(frac_coords):
                            """Wrap fractional coordinates into [0,1)."""
                            coords = np.array(frac_coords)
                            return coords % 1


                        def compute_periodic_distance_matrix(frac_coords):
                            """Compute pairwise distances considering periodic boundary conditions."""
                            n = len(frac_coords)
                            dist_matrix = np.zeros((n, n))
                            for i in range(n):
                                for j in range(i, n):
                                    delta = frac_coords[i] - frac_coords[j]
                                    delta = delta - np.round(delta)
                                    dist = np.linalg.norm(delta)
                                    dist_matrix[i, j] = dist_matrix[j, i] = dist
                            return dist_matrix


                        def select_spaced_points(frac_coords, n_points, mode, target_value=0.5):
                            coords_wrapped = wrap_coordinates(frac_coords)
                            dist_matrix = compute_periodic_distance_matrix(coords_wrapped)
                            import random
                            selected_indices = [random.randrange(len(coords_wrapped))]
                            # selected_indices = [0]  # Always select the first candidate as the start.
                            for _ in range(1, n_points):
                                remaining = [i for i in range(len(coords_wrapped)) if i not in selected_indices]
                                if mode == "farthest":
                                    next_index = max(remaining,
                                                     key=lambda i: min(dist_matrix[i, j] for j in selected_indices))
                                elif mode == "nearest":
                                    next_index = min(remaining,
                                                     key=lambda i: min(dist_matrix[i, j] for j in selected_indices))
                                elif mode == "moderate":
                                    next_index = min(remaining, key=lambda i: abs(
                                        sum(dist_matrix[i, j] for j in selected_indices) / len(
                                            selected_indices) - target_value))
                                else:
                                    raise ValueError(
                                        "Invalid selection mode. Use 'farthest', 'nearest', or 'moderate'.")
                                selected_indices.append(next_index)
                            # Return both the selected coordinates and their local indices.
                            selected_coords = np.array(coords_wrapped)[selected_indices].tolist()
                            return selected_coords, selected_indices


                        # ---------- Interstitial Functions ----------

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


                        def insert_interstitials_into_structure(structure, interstitial_element, n_interstitials,
                                                                which_interstitial=0, mode="farthest",
                                                                clustering_tol=0.75,
                                                                min_dist=0.5):
                            from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
                            with colb3:
                                with st.spinner(f"Calculating available interstitials positions, please wait. 😊"):
                                    generator = VoronoiInterstitialGenerator(clustering_tol=clustering_tol,
                                                                             min_dist=min_dist)

                                    frac_coords = []
                                    frac_coords_dict = {}
                                    unique_int = []
                                    idx = 0
                                    # Collect candidate sites from the generator.
                                    with st.expander("See the unique interstitial positions", icon="🧠"):
                                        for interstitial in generator.generate(structure, "H"):
                                            frac_coords_dict[idx] = []
                                            unique_int.append(interstitial.site.frac_coords)
                                            label = classify_interstitial_site(structure, interstitial.site.frac_coords)
                                            rounded_coords = [round(float(x), 3) for x in interstitial.site.frac_coords]
                                            st.write(
                                                f"🧠 Unique interstitial site (**Type {idx + 1}**)  at {rounded_coords}, {label} (#{len(interstitial.equivalent_sites)} sites)")
                                            for site in interstitial.equivalent_sites:
                                                frac_coords.append(site.frac_coords)
                                                frac_coords_dict[idx].append(site.frac_coords)
                                            idx += 1

                                    st.write(f"**Total number of available interstitial positions:**", len(frac_coords))

                                    if which_interstitial == 0:
                                        frac_coords_use = frac_coords
                                    else:
                                        frac_coords_use = frac_coords_dict.get(which_interstitial - 1, [])

                                    # Select the desired number of points.
                                    selected_points, _ = select_spaced_points(frac_coords_use, n_points=n_interstitials,
                                                                              mode=mode)
                                    new_structure = structure.copy()
                                    for point in selected_points:
                                        new_structure.append(
                                            species=Element(interstitial_element),
                                            coords=point,
                                            coords_are_cartesian=False  # Input is fractional.
                                        )
                                return new_structure


                        # ---------- Vacancy Functions ----------

                        def remove_vacancies_from_structure(structure, vacancy_percentages, selection_mode="farthest",
                                                            target_value=0.5):
                            with colb3:
                                with st.spinner(f"Creating substitutes, please wait. 😊"):
                                    new_structure = structure.copy()
                                    indices_to_remove = []
                                    for el, perc in vacancy_percentages.items():
                                        # Get indices of sites for the element.
                                        el_indices = [i for i, site in enumerate(new_structure.sites) if
                                                      site.specie.symbol == el]
                                        n_sites = len(el_indices)
                                        n_remove = int(round(n_sites * perc / 100.0))
                                        st.write(f"🧠 Removed {n_remove} atoms of {el}.")
                                        if n_remove < 1:
                                            continue
                                        # Get the fractional coordinates of these sites.
                                        el_coords = [new_structure.sites[i].frac_coords for i in el_indices]
                                        # If fewer removals than available sites, use the selection function.
                                        if n_remove < len(el_coords):
                                            _, selected_local_indices = select_spaced_points(el_coords,
                                                                                             n_points=n_remove,
                                                                                             mode=selection_mode,
                                                                                             target_value=target_value)
                                            # Map the selected local indices back to global indices.
                                            selected_global_indices = [el_indices[i] for i in selected_local_indices]
                                        else:
                                            selected_global_indices = el_indices
                                        indices_to_remove.extend(selected_global_indices)
                                    # Remove sites in descending order (so that indices remain valid).
                                    for i in sorted(indices_to_remove, reverse=True):
                                        new_structure.remove_sites([i])
                            return new_structure


                        # ==================== Substitute Functions ====================
                        with colb3:
                            st.markdown(f"### Log output:")


                        def substitute_atoms_in_structure(structure, substitution_dict, selection_mode="farthest",
                                                          target_value=0.5):
                            with colb3:
                                with st.spinner(f"Creating substitutes, please wait. 😊"):
                                    new_species = [site.species_string for site in structure.sites]
                                    new_coords = [site.frac_coords for site in structure.sites]
                                    for orig_el, settings in substitution_dict.items():
                                        perc = settings.get("percentage", 0)
                                        sub_el = settings.get("substitute", "").strip()
                                        if perc <= 0 or not sub_el:
                                            continue
                                        indices = [i for i, site in enumerate(structure.sites) if
                                                   site.specie.symbol == orig_el]
                                        n_sites = len(indices)
                                        n_substitute = int(round(n_sites * perc / 100.0))
                                        st.write(f"🧠 Replaced {n_substitute} atoms of {orig_el} with {sub_el}.")

                                        if n_substitute < 1:
                                            continue
                                        el_coords = [new_coords[i] for i in indices]
                                        if n_substitute < len(el_coords):
                                            _, selected_local_indices = select_spaced_points(el_coords,
                                                                                             n_points=n_substitute,
                                                                                             mode=selection_mode,
                                                                                             target_value=target_value)
                                            selected_global_indices = [indices[i] for i in selected_local_indices]
                                        else:
                                            selected_global_indices = indices
                                        for i in selected_global_indices:
                                            new_species[i] = sub_el
                                    # Rebuild the structure using the original lattice, updated species, and coordinates.
                                    new_structure = Structure(structure.lattice, new_species, new_coords,
                                                              coords_are_cartesian=False)
                            return new_structure


                        # ==================== Streamlit UI ====================

                        current_atom_count = len(
                            structure)

                        if current_atom_count > 32:
                            operation_options = ["Create Vacancies", "Substitute Atoms"]
                            disabled_message = f"⚠️ **Interstitials option disabled:** Current structure has {current_atom_count} atoms (limit: 32). Voronoi method is computationally too extensive."
                        else:
                            operation_options = ["Insert Interstitials (Voronoi method)", "Create Vacancies",
                                                 "Substitute Atoms"]
                            disabled_message = None

                        # Display warning message if interstitials is disabled
                        if disabled_message:
                            st.warning(disabled_message)

                        # Choose among the three operation modes.
                        operation_mode = st.selectbox("Choose Operation Mode",
                                                      operation_options, help="""
                    #Interstitials settings
                    - **Element**: The chemical symbol of the interstitial atom you want to insert (e.g., `N` for nitrogen).
                    - **# to Insert**: The number of interstitial atoms to insert into the structure.
                    - **Type (0=all, 1=first...)**: Selects a specific interstitial site type.  
                      - `0` uses all detected interstitial sites.  
                      - `1` uses only the first unique type, `2` for second, etc.

                    - **Selection Mode**: How to choose which interstitial sites to use:  
                      - `farthest`: picks sites farthest apart from each other.  
                      - `nearest`: picks sites closest together.  
                      - `moderate`: balances distances around a target value.

                    - **Clustering Tol**: Tolerance for clustering nearby interstitial candidates together (higher = more merging).
                    - **Min Dist**: Minimum allowed distance between interstitials and other atoms when generating candidate sites. Do not consider any candidate site that is closer than this distance to an existing atom.

                    #Vacancy settings
                    - **Vacancy Selection Mode**: Strategy for choosing which atoms to remove:
                      - `farthest`: removes atoms that are farthest apart, to maximize spacing.
                      - `nearest`: removes atoms closest together, forming local vacancy clusters.
                      - `moderate`: selects atoms to remove so that the average spacing between them is close to a target value.

                    - **Target (moderate mode)**: Only used when `moderate` mode is selected.  
                      This value defines the average spacing (in fractional coordinates) between vacancies.

                    - **Vacancy % for [Element]**: Percentage of atoms to remove for each element.  
                      For example, if there are 20 O atoms and you set 10%, two O atoms will be randomly removed based on the selection mode.

                    #Substitution settings
                    - **Substitution Selection Mode**: Strategy to determine *which* atoms of a given element are substituted:
                      - `farthest`: substitutes atoms spaced far apart from each other.
                      - `nearest`: substitutes atoms that are close together.
                      - `moderate`: substitutes atoms spaced at an average distance close to the specified target.

                    - **Target (moderate mode)**: Only used when `moderate` mode is selected.  
                      It defines the preferred average spacing (in fractional coordinates) between substituted atoms.

                    - **Substitution % for [Element]**: How many atoms (as a percentage) of a given element should be substituted.

                    - **Substitute [Element] with**: The element symbol you want to use as a replacement.  
                      Leave blank or set substitution % to 0 to skip substitution for that element.
                            """)

                        if operation_mode == "Insert Interstitials (Voronoi method)":
                            st.markdown("""
                            **Insert Interstitials Settings**
                            """)

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                interstitial_element_to_place = st.text_input("Element", value="N")
                            with col2:
                                number_of_interstitials_to_insert = st.number_input("# to Insert", value=2, min_value=1)
                            with col3:
                                which_interstitial_to_use = st.number_input("Type (0=all, 1=first...)", value=0,
                                                                            min_value=0)

                            col4, col5, col6 = st.columns(3)
                            with col4:
                                selection_mode = st.selectbox("Selection Mode",
                                                              options=["farthest", "nearest", "moderate"],
                                                              index=0)
                            with col5:
                                clustering_tol = st.number_input("Clustering Tol", value=0.75, step=0.05, format="%.2f")
                            with col6:
                                min_dist = st.number_input("Min Dist", value=0.5, step=0.05, format="%.2f")

                        elif operation_mode == "Create Vacancies":
                            st.markdown("""

                            """)
                            # Row 1: Two columns for vacancy mode and target value
                            col1, col2 = st.columns(2)
                            vacancy_selection_mode = col1.selectbox("Vacancy Selection Mode",
                                                                    ["farthest", "nearest", "moderate"], index=0)
                            if vacancy_selection_mode == "moderate":
                                vacancy_target_value = col2.number_input("Target (moderate mode)", value=0.5, step=0.05,
                                                                         format="%.2f")
                            else:
                                vacancy_target_value = 0.5

                            # Row 2: One column per element for vacancy percentage input
                            elements = sorted({site.specie.symbol for site in mp_struct.sites})
                            cols = st.columns(len(elements))
                            vacancy_percentages = {
                                el: cols[i].number_input(f"Vacancy % for {el}", value=0.0, min_value=0.0,
                                                         max_value=100.0,
                                                         step=1.0, format="%.1f")
                                for i, el in enumerate(elements)}

                        elif operation_mode == "Substitute Atoms":
                            st.markdown("""
                            **Substitution Settings**
                            """)
                            # Row 1: Substitution mode and target value
                            col1, col2 = st.columns(2)
                            substitution_selection_mode = col1.selectbox("Substitution Selection Mode",
                                                                         ["farthest", "nearest", "moderate"], index=0)
                            if substitution_selection_mode == "moderate":
                                substitution_target_value = col2.number_input("Target (moderate mode)", value=0.5,
                                                                              step=0.05,
                                                                              format="%.2f")
                            else:
                                substitution_target_value = 0.5
                            # Row 2: One column per element (each showing two inputs: percentage and target)
                            elements = sorted({site.specie.symbol for site in mp_struct.sites})
                            cols = st.columns(len(elements))
                            substitution_settings = {}
                            for i, el in enumerate(elements):
                                with cols[i]:
                                    sub_perc = st.number_input(f"Substitution % for {el}", value=0.0, min_value=0.0,
                                                               max_value=100.0, step=1.0, format="%.1f",
                                                               key=f"sub_perc_{el}")
                                    sub_target = st.text_input(f"Substitute {el} with", value="",
                                                               key=f"sub_target_{el}")
                                substitution_settings[el] = {"percentage": sub_perc, "substitute": sub_target.strip()}

                        # ==================== Execute Operation ====================
                        with colb1:
                            if st.button("🔄 Reset to Original Structure"):
                                st.session_state["reset_requested"] = True
                                st.rerun()
                        if operation_mode == "Insert Interstitials (Voronoi method)":
                            if st.button("Insert Interstitials"):
                                updated_structure = insert_interstitials_into_structure(mp_struct,
                                                                                        interstitial_element_to_place,
                                                                                        number_of_interstitials_to_insert,
                                                                                        which_interstitial_to_use,
                                                                                        mode=selection_mode,
                                                                                        clustering_tol=clustering_tol,
                                                                                        min_dist=min_dist)

                                mp_struct = updated_structure
                                st.session_state["current_structure"] = updated_structure
                                #base_for_supercell = st.session_state["base_modified_structure"]
                                with colb3:
                                    st.success("Interstitials inserted and structure updated!")
                                st.session_state["helpful"] = True

                        elif operation_mode == "Create Vacancies":
                            if st.button("Create Vacancies"):
                                updated_structure = remove_vacancies_from_structure(mp_struct,
                                                                                    vacancy_percentages,
                                                                                    selection_mode=vacancy_selection_mode,
                                                                                    target_value=vacancy_target_value)

                                mp_struct = updated_structure
                                st.session_state["current_structure"] = updated_structure
                                #base_for_supercell = st.session_state["base_modified_structure"]
                                st.session_state["last_multiplier"] = (1, 1, 1)
                                with colb3:
                                    st.success("Vacancies created and structure updated!")
                                st.session_state["helpful"] = True
                            # st.rerun()
                        elif operation_mode == "Substitute Atoms":
                            if st.button("Substitute Atoms"):
                                updated_structure = substitute_atoms_in_structure(mp_struct,
                                                                                  substitution_settings,
                                                                                  selection_mode=substitution_selection_mode,
                                                                                  target_value=substitution_target_value)

                                mp_struct = updated_structure
                                st.session_state["current_structure"] = updated_structure
                                #base_for_supercell = st.session_state["base_modified_structure"]
                                with colb3:
                                    st.success("Substitutions applied and structure updated!")
                                st.session_state["helpful"] = True
                            #  st.rerun()

            # Checkbox option to show atomic positions (labels on structure and list in table)


            if show_3d_visualization:
                xyz_io = StringIO()
                if st.session_state["current_structure"] is not None:
                    structure = AseAtomsAdaptor.get_atoms(st.session_state["current_structure"])
                write(xyz_io, structure, format="xyz")
                xyz_str = xyz_io.getvalue()
                view = py3Dmol.view(width=1200, height=800)
                view.addModel(xyz_str, "xyz")
                view.setStyle({'model': 0}, {"sphere": {"radius": 0.3, "colorscheme": "Jmol"}})
                #view.setStyle({'model': 0}, {"line": {}})
                cell = structure.get_cell()  # 3x3 array of lattice vectors
                add_box(view, cell, color='black', linewidth=4)
                view.zoomTo()
                view.zoom(1.2)



                offset_distance = 0.3  # distance to offset
                overlay_radius = 0.15  # radius for the overlay spheres

                # Create a new list to store atomic info for the table (labels with occupancy info)
                atomic_info = []
                inv_cell = np.linalg.inv(cell)

                visual_pmg_structure_partial_check = load_structure(selected_file)

                # Check whether any site in the structure has partial occupancy.
                has_partial_occ = any(
                    (len(site.species) > 1) or any(occ < 1 for occ in site.species.values())
                    for site in visual_pmg_structure_partial_check.sites
                )

                # If partial occupancy is detected, notify the user and offer an enhanced visualization option.
                if has_partial_occ:
                    st.info(
                        f"Partial occupancy detected in the uploaded structure. Note that the conversion between cell representions will not be possible now.\n To continue ")

                else:
                    visualize_partial = False

                if visualize_partial:
                    visual_pmg_structure = load_structure(selected_file)
                    from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation

                    # try:
                    #    structure_with_oxi = visual_pmg_structure.add_oxidation_state_by_guess()
                    # except Exception as e:
                    # Optionally, handle the exception if oxidation states cannot be assigned.
                    #    raise ValueError(f"Could not assign oxidation states to the structure: {e}")

                    # Convert to ordered structure
                    # ordered_structure = OrderDisorderedStructureTransformation(no_oxi_states=True)
                    # ordered_structure = order_trans.apply_transformation(structure_with_oxi)

                    from pymatgen.core import Structure

                    # Get lattice from original structure
                    lattice = visual_pmg_structure.lattice

                    # Build new species list: choose the species with highest occupancy at each site
                    species = []
                    coords = []

                    for site in visual_pmg_structure.sites:
                        # Pick the species with the highest occupancy
                        dominant_specie = max(site.species.items(), key=lambda x: x[1])[0]
                        species.append(dominant_specie)
                        coords.append(site.frac_coords)

                    # Create a new ordered structure
                    ordrd = Structure(lattice, species, coords)
                    structure = AseAtomsAdaptor.get_atoms(ordrd)

                    xyz_io = StringIO()
                    write(xyz_io, structure, format="xyz")
                    xyz_str = xyz_io.getvalue()
                    view = py3Dmol.view(width=1200, height=800)
                    view.addModel(xyz_str, "xyz")
                    view.setStyle({'model': 0}, {"sphere": {"radius": 0.3, "colorscheme": "Jmol"}})
                    cell = structure.get_cell()  # 3x3 array of lattice vectors
                    add_box(view, cell, color='black', linewidth=4)
                    view.zoomTo()
                    view.zoom(1.2)
                    # Enhanced visualization: iterate over the pymatgen structure to use occupancy info.

                    for i, site in enumerate(visual_pmg_structure.sites):
                        # Get Cartesian coordinates.
                        x, y, z = site.coords

                        # Build a string for species and occupancy details.
                        species_info = []
                        for specie, occ in site.species.items():
                            occ_str = f"({occ * 100:.0f}%)" if occ < 1 else ""
                            species_info.append(f"{specie.symbol}{occ_str}")
                        label_text = f"{'/'.join(species_info)}{i}"
                        if show_atomic:
                            # Add a label with the occupancy info.
                            view.addLabel(label_text, {
                                "position": {"x": x, "y": y, "z": z},
                                "backgroundColor": "white",
                                "fontColor": "black",
                                "fontSize": 10,
                                "borderThickness": 1,
                                "borderColor": "black"
                            })

                        frac = np.dot(inv_cell, [x, y, z])
                        atomic_info.append({
                            "Atom": label_text,
                            "Elements": "/".join(species_info),
                            "X": round(x, 3),
                            "Y": round(y, 3),
                            "Z": round(z, 3),
                            "Frac X": round(frac[0], 3),
                            "Frac Y": round(frac[1], 3),
                            "Frac Z": round(frac[2], 3)
                        })

                        # For sites with partial occupancy, overlay extra spheres.
                        species_dict = site.species
                        if (len(species_dict) > 1) or any(occ < 1 for occ in species_dict.values()):
                            num_species = len(species_dict)
                            # Distribute offsets around a circle (here in the xy-plane).
                            angles = np.linspace(0, 2 * np.pi, num_species, endpoint=False)
                            offset_distance = 0.3  # adjust as needed
                            overlay_radius = 0.15  # adjust as needed
                            for j, ((specie, occ), angle) in enumerate(zip(species_dict.items(), angles)):
                                dx = offset_distance * np.cos(angle)
                                dy = offset_distance * np.sin(angle)
                                sphere_center = {"x": x + dx, "y": y + dy, "z": z}
                                view.addSphere({
                                    "center": sphere_center,
                                    "radius": overlay_radius,
                                    "color": jmol_colors.get(specie.symbol, "gray"),
                                    "opacity": 1.0
                                })
                else:
                    if show_atomic:
                        # Basic visualization (as before): iterate over ASE atoms.
                        for i, atom in enumerate(structure):
                            symbol = atom.symbol
                            x, y, z = atom.position
                            label_text = f"{symbol}{i}"
                            view.addLabel(label_text, {
                                "position": {"x": x, "y": y, "z": z},
                                "backgroundColor": "white",
                                "fontColor": "black",
                                "fontSize": 10,
                                "borderThickness": 1,
                                "borderColor": "black"
                            })
                            frac = np.dot(inv_cell, atom.position)
                            atomic_info.append({
                                "Atom": label_text,
                                "Elements": symbol,
                                "X": round(x, 3),
                                "Y": round(y, 3),
                                "Z": round(z, 3),
                                "Frac X": round(frac[0], 3),
                                "Frac Y": round(frac[1], 3),
                                "Frac Z": round(frac[2], 3)
                            })

                html_str = view._make_html()

                centered_html = f"""
                <div style="
                    display: flex;
                    justify-content: center;
                ">
                    <div style="
                        border: 3px solid black;
                        border-radius: 10px;
                        overflow: hidden;
                        padding: 0;
                        margin: 0;
                        display: inline-block;
                    ">
                        {html_str}
                    </div>
                </div>
                """

                unique_elements = sorted(set(structure.get_chemical_symbols()))
                legend_html = "<div style='display: flex; flex-wrap: wrap; align-items: center;justify-content: center;'>"
                for elem in unique_elements:
                    color = jmol_colors.get(elem, "#CCCCCC")
                    legend_html += (
                        f"<div style='margin-right: 15px; display: flex; align-items: center;'>"
                        f"<div style='width: 20px; height: 20px; background-color: {color}; margin-right: 5px; border: 1px solid black;'></div>"
                        f"<span>{elem}</span></div>"
                    )
                legend_html += "</div>"



            # Download CIF for visualized structure
            if mp_struct:
                visual_pmg_structure = converted_structure
            else:
                visual_pmg_structure = load_structure(selected_file)
            for site in visual_pmg_structure.sites:
                pass
                # print(site.species)  # This will show occupancy info
                # Write CIF content directly using pymatgen:
                # Otherwise, use the chosen conversion
            if convert_to_conventional:
                lattice_info = "conventional"
            elif pymatgen_prim_cell_niggli:
                lattice_info = "primitive_niggli"
            elif pymatgen_prim_cell_lll:
                lattice_info = "primitive_lll"
            elif pymatgen_prim_cell_no_reduce:
                lattice_info = "primitive_no_reduce"
            else:
                lattice_info = "primitive"

            cif_writer_visual = CifWriter(visual_pmg_structure, symprec=0.1, refine_struct=False)

            cif_content_visual = cif_writer_visual.__str__()

            # Prepare a file name (ensure it ends with .cif)
            download_file_name = selected_file.split('.')[0] + '_{}'.format(lattice_info) + '.cif'
            if not download_file_name.lower().endswith('.cif'):
                download_file_name = selected_file.split('.')[0] + '_{}'.format(lattice_info) + '.cif'
            if visualize_partial == False:
                with col_download:
                    with st.expander("Download Options", expanded=True):
                        file_format = st.radio(
                            "Select file format",
                            ("CIF", "VASP", "LAMMPS", "XYZ",),
                            horizontal=True
                        )

                        file_content = None
                        download_file_name = None
                        mime = "text/plain"

                        try:
                            if file_format == "CIF":
                                # Use pymatgen's CifWriter for CIF output.
                                from pymatgen.io.cif import CifWriter

                                cif_writer_visual = CifWriter(st.session_state["current_structure"], symprec=0.1,
                                                              refine_struct=False)
                                file_content = str(cif_writer_visual)

                                download_file_name = selected_file.split('.')[
                                                         0] + '_' + lattice_info + f'_Supercell_{n_a}_{n_b}_{n_c}.cif'

                                mime = "chemical/x-cif"
                            elif file_format == "VASP":
                                out = StringIO()
                                current_ase_structure = AseAtomsAdaptor.get_atoms(st.session_state["current_structure"])

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
                                        constraint = FixAtoms(indices=[])  # No atoms are fixed, so all will be T T T
                                        current_ase_structure.set_constraint(constraint)
                                write(out, current_ase_structure, format="vasp", direct=use_fractional, sort=True)
                                file_content = out.getvalue()
                                download_file_name = selected_file.split('.')[
                                                         0] + '_' + lattice_info + f'_Supercell_{n_a}_{n_b}_{n_c}.poscar'

                            elif file_format == "LAMMPS":
                                st.markdown("**LAMMPS Export Options**")

                                atom_style = st.selectbox("Select atom_style", ["atomic", "charge", "full"], index=0)
                                units = st.selectbox("Select units", ["metal", "real", "si"], index=0)
                                include_masses = st.checkbox("Include atomic masses", value=True)
                                force_skew = st.checkbox("Force triclinic cell (skew)", value=False)
                                current_ase_structure = AseAtomsAdaptor.get_atoms(st.session_state["current_structure"])
                                out = StringIO()
                                write(
                                    out,
                                    current_ase_structure,
                                    format="lammps-data",
                                    atom_style=atom_style,
                                    units=units,
                                    masses=include_masses,
                                    force_skew=force_skew
                                )
                                file_content = out.getvalue()

                                download_file_name = selected_file.split('.')[
                                                         0] + '_' + lattice_info + f'_Supercell_{n_a}_{n_b}_{n_c}' + f'_{atom_style}_{units}.lmp'

                            elif file_format == "XYZ":
                                current_ase_structure = AseAtomsAdaptor.get_atoms(st.session_state["current_structure"])
                                out = StringIO()
                                write(out, current_ase_structure, format="xyz")
                                file_content = out.getvalue()
                                download_file_name = selected_file.split('.')[
                                                         0] + '_' + lattice_info + f'_Supercell_{n_a}_{n_b}_{n_c}.xyz'

                        except Exception as e:
                            st.error(f"Error generating {file_format} file: {e}")

                        if file_content is not None:
                            st.download_button(
                                label=f"Download {file_format} file",
                                data=file_content,
                                file_name=download_file_name,
                                type="primary",
                                mime=mime
                            )
            # Get lattice parameters
            cell_params = structure.get_cell_lengths_and_angles()  # (a, b, c, α, β, γ)
            a_para, b_para, c_para = cell_params[:3]
            alpha, beta, gamma = [radians(x) for x in cell_params[3:]]

            volume = a_para * b_para * c_para * sqrt(
                1 - cos(alpha) ** 2 - cos(beta) ** 2 - cos(gamma) ** 2 +
                2 * cos(alpha) * cos(beta) * cos(gamma)
            )
            # Get lattice parameters

            lattice_str = (
                f"a = {cell_params[0]:.4f} Å<br>"
                f"b = {cell_params[1]:.4f} Å<br>"
                f"c = {cell_params[2]:.4f} Å<br>"
                f"α = {cell_params[3]:.2f}°<br>"
                f"β = {cell_params[4]:.2f}°<br>"
                f"γ = {cell_params[5]:.2f}°<br>"
                f"Volume = {volume:.2f} Å³"
            )

            left_col, right_col = st.columns([1, 3])

            with left_col:
                st.markdown(
                    f"<h3 style='text-align: center;'>Interactive Structure Visualization ({structure_cell_choice}) </h3>",
                    unsafe_allow_html=True)

                try:
                    mg_structure = AseAtomsAdaptor.get_structure(structure)
                    sg_analyzer = SpacegroupAnalyzer(mg_structure)
                    spg_symbol = sg_analyzer.get_space_group_symbol()
                    spg_number = sg_analyzer.get_space_group_number()
                    space_group_str = f"{spg_symbol} ({spg_number})"
                except Exception:
                    space_group_str = "Not available"
                try:
                    mg_structure = AseAtomsAdaptor.get_structure(structure)
                    sg_analyzer = SpacegroupAnalyzer(mg_structure)
                    spg_symbol = sg_analyzer.get_space_group_symbol()
                    spg_number = sg_analyzer.get_space_group_number()
                    space_group_str = f"{spg_symbol} ({spg_number})"

                    # New check
                    same_lattice = lattice_same_conventional_vs_primitive(mg_structure)
                    if same_lattice is None:
                        cell_note = "⚠️ Could not determine if cells are identical."
                        cell_note_color = "gray"
                    elif same_lattice:
                        cell_note = "✅ Note: Conventional and Primitive Cells have the SAME cell volume."
                        cell_note_color = "green"
                    else:
                        cell_note = "Note: Conventional and Primitive Cells have DIFFERENT cell volume."
                        cell_note_color = "gray"
                except Exception:
                    space_group_str = "Not available"
                    cell_note = "⚠️ Could not determine space group or cell similarity."
                    cell_note_color = "gray"

                st.markdown(f"""
                <div style='text-align: center; font-size: 22px; color: {"green" if same_lattice else "gray"}'>
                    <strong>{cell_note}</strong>
                </div>
                """, unsafe_allow_html=True)
                if show_3d_visualization:
                    st.markdown(f"""
                    <div style='text-align: center; font-size: 22px;'>
                        <p><strong>Lattice Parameters:</strong><br>{lattice_str}</p>
                        <p><strong>Legend:</strong><br>{legend_html}</p>
                        <p><strong>Number of Atoms:</strong> {len(structure)}</p>
                        <p><strong>Space Group:</strong> {space_group_str}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='text-align: center; font-size: 22px;'>
                        <p><strong>Lattice Parameters:</strong><br>{lattice_str}</p>
                        <p><strong>Number of Atoms:</strong> {len(structure)}</p>
                        <p><strong>Space Group:</strong> {space_group_str}</p>
                    </div>
                    """, unsafe_allow_html=True)
                # If atomic positions are to be shown, display them as a table.
            if show_3d_visualization:
                if show_atomic:
                    df_atoms = pd.DataFrame(atomic_info)
                    st.subheader("Atomic Positions")
                    st.dataframe(df_atoms)

                with right_col:
                    st.components.v1.html(centered_html, height=600)



st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
import sys


def get_session_memory_usage():
    total_size = 0
    for key in st.session_state:
        try:
            total_size += sys.getsizeof(st.session_state[key])
        except Exception:
            pass
    return total_size / 1024  # in KB


memory_kb = get_session_memory_usage()
st.markdown(f"🧠 Estimated session memory usage: **{memory_kb:.2f} KB**")
st.markdown("""
**The XRDlicious application is open-source and released under the [MIT License](https://github.com/bracerino/prdf-calculator-online/blob/main/LICENCSE).**
""")


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # in MB


memory_usage = get_memory_usage()
st.write(
    f"🔍 Current memory usage: **{memory_usage:.2f} MB**. We are now using free hosting by Streamlit Community Cloud servis, which has a limit for RAM memory of 2.6 GBs. If we will see higher usage of our app and need for a higher memory, we will upgrade to paid server, allowing us to improve the performance. :]")

st.markdown("""

### Acknowledgments

This project uses several open-source tools and datasets. We gratefully acknowledge their authors: **[Matminer](https://github.com/hackingmaterials/matminer)** Licensed under the [Modified BSD License](https://github.com/hackingmaterials/matminer/blob/main/LICENSE). **[Pymatgen](https://github.com/materialsproject/pymatgen)** Licensed under the [MIT License](https://github.com/materialsproject/pymatgen/blob/master/LICENSE)."
 **[ASE (Atomic Simulation Environment)](https://gitlab.com/ase/ase)** Licensed under the [GNU Lesser General Public License (LGPL)](https://gitlab.com/ase/ase/-/blob/master/COPYING.LESSER). **[Py3DMol](https://github.com/avirshup/py3dmol/tree/master)** Licensed under the [BSD-style License](https://github.com/avirshup/py3dmol/blob/master/LICENSE.txt). **[Materials Project](https://next-gen.materialsproject.org/)** Data from the Materials Project is made available under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). **[AFLOW](http://aflow.org)** Licensed under the [GNU General Public License (GPL)](https://www.gnu.org/licenses/gpl-3.0.html).
""")
