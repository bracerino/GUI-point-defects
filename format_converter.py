import streamlit as st
import io
from io import StringIO, BytesIO
import zipfile
import tempfile
import os
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write, read
import numpy as np
import pandas as pd


def load_structure_from_file(file_data, filename):
    try:
        bytes_data = file_data.getvalue() if hasattr(file_data, 'getvalue') else file_data.read()
        file_lower = filename.lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            tmp_file.write(bytes_data)
            tmp_path = tmp_file.name

        try:
            if file_lower.endswith('.cif'):
                structure = Structure.from_file(tmp_path)
            elif file_lower.endswith(('.poscar', '.vasp', '.contcar')):
                from pymatgen.io.vasp import Poscar
                poscar = Poscar.from_file(tmp_path)
                structure = poscar.structure
            elif file_lower.endswith('.lmp'):
                ase_atoms = read(tmp_path, format='lammps-data')
                structure = AseAtomsAdaptor.get_structure(ase_atoms)
            elif file_lower.endswith('.xyz'):
                ase_atoms = read(tmp_path, format='xyz')
                structure = AseAtomsAdaptor.get_structure(ase_atoms)
            else:
                structure = Structure.from_file(tmp_path)

            os.unlink(tmp_path)
            return structure

        except Exception as e:
            os.unlink(tmp_path)
            raise e

    except Exception as e:
        return None, str(e)


def convert_structure_to_format(structure, output_format, options):
    try:
        if output_format == "CIF":
            if options.get('cif_refine', True):
                content = str(CifWriter(structure, symprec=options.get('cif_symprec', 0.01)))
            else:
                content = str(CifWriter(structure, symprec=None, refine_struct=False))
            extension = ".cif"

        elif output_format == "VASP (POSCAR)":
            ase_atoms = AseAtomsAdaptor.get_atoms(structure)

            if options.get('vasp_selective', False):
                from ase.constraints import FixAtoms
                ase_atoms.set_constraint(FixAtoms(indices=[]))

            sio = StringIO()
            write(sio, ase_atoms, format="vasp",
                  direct=options.get('vasp_fractional', True),
                  sort=options.get('vasp_sort', True))
            content = sio.getvalue()
            extension = ".poscar"

        elif output_format == "LAMMPS":
            ase_atoms = AseAtomsAdaptor.get_atoms(structure)
            sio = StringIO()
            write(sio, ase_atoms, format="lammps-data",
                  atom_style=options.get('lmp_atom_style', 'atomic'),
                  units=options.get('lmp_units', 'metal'),
                  masses=options.get('lmp_masses', True),
                  force_skew=options.get('lmp_force_skew', False))
            content = sio.getvalue()
            extension = f"_{options.get('lmp_atom_style', 'atomic')}.lmp"

        elif output_format == "XYZ":
            lattice_vectors = structure.lattice.matrix
            cart_coords = []
            elements = []
            for site in structure:
                cart_coords.append(structure.lattice.get_cartesian_coords(site.frac_coords))
                elements.append(site.specie.symbol)

            xyz_lines = [str(len(structure))]
            lattice_string = " ".join([f"{x:.6f}" for row in lattice_vectors for x in row])
            properties = "Properties=species:S:1:pos:R:3"
            xyz_lines.append(f'Lattice="{lattice_string}" {properties}')

            for element, coord in zip(elements, cart_coords):
                xyz_lines.append(f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")

            content = "\n".join(xyz_lines)
            extension = ".xyz"

        else:
            return None, None

        return content, extension

    except Exception as e:
        st.error(f"Error during conversion: {e}")
        return None, None


def render_converter_uploader():
    st.markdown("#### 📁 Upload Files for Conversion")

    uploaded_files = st.file_uploader(
        "Upload structure files (CIF, POSCAR, LAMMPS, XYZ)",
        type=['cif', 'poscar', 'vasp', 'contcar', 'lmp', 'xyz'],
        accept_multiple_files=True,
        key="converter_file_uploader",
        help="Upload multiple structure files to convert them to another format"
    )

    if uploaded_files:
        success_count = 0
        error_count = 0

        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.converter_uploaded_structures:
                result = load_structure_from_file(uploaded_file, uploaded_file.name)

                if isinstance(result, tuple):
                    structure, error = result
                    if structure is None:
                        st.error(f"❌ Failed to load {uploaded_file.name}: {error}")
                        error_count += 1
                    else:
                        st.session_state.converter_uploaded_structures[uploaded_file.name] = structure
                        success_count += 1
                else:
                    st.session_state.converter_uploaded_structures[uploaded_file.name] = result
                    success_count += 1

        if success_count > 0:
            st.success(f"✅ Successfully loaded {success_count} structure(s)")
        if error_count > 0:
            st.warning(f"⚠️ Failed to load {error_count} file(s)")


def render_format_converter_ui(main_structures_dict=None):
    st.markdown("### 🔄 Format Converter")
    st.info("Convert crystal structures between different file formats (CIF, POSCAR, LAMMPS, XYZ)")

    if 'converter_uploaded_structures' not in st.session_state:
        st.session_state.converter_uploaded_structures = {}

    if 'converter_files' not in st.session_state:
        st.session_state.converter_files = {}

    structure_source = st.radio(
        "Structure source:",
        options=["Converter's own uploader", "Main app structures"],
        key="conv_structure_source",
        horizontal=True,
        help="Choose whether to upload new files or use structures already loaded in the main app"
    )

    structures_to_use = {}

    if structure_source == "Converter's own uploader":
        render_converter_uploader()
        structures_to_use = st.session_state.converter_uploaded_structures

        if not structures_to_use:
            st.info("👆 Upload structure files above to begin conversion")
            return
    else:
        if main_structures_dict and len(main_structures_dict) > 0:
            st.success(f"✅ Using {len(main_structures_dict)} structure(s) from main app")
            structures_to_use = main_structures_dict
        else:
            st.warning(
                "⚠️ No structures loaded in main app. Please upload structures first or switch to converter's own uploader.")
            return

    if not structures_to_use:
        return

    st.markdown("---")
    st.markdown("#### ⚙️ Conversion Settings")

    output_format = st.selectbox(
        "Select output format",
        ["CIF", "VASP (POSCAR)", "LAMMPS", "XYZ"],
        key="converter_output_format"
    )

    col1, col2 = st.columns(2)

    options = {}

    if output_format == "VASP (POSCAR)":
        with col1:
            options['vasp_fractional'] = st.checkbox("Use fractional coordinates", value=True, key="conv_vasp_frac")
            options['vasp_sort'] = st.checkbox("Sort by element", value=True, key="conv_vasp_sort")
        with col2:
            options['vasp_selective'] = st.checkbox("Include selective dynamics (all free)", value=False,
                                                    key="conv_vasp_sel")

    elif output_format == "LAMMPS":
        with col1:
            options['lmp_atom_style'] = st.selectbox("atom_style", ["atomic", "charge", "full"], key="conv_lmp_style")
            options['lmp_units'] = st.selectbox("units", ["metal", "real", "si"], key="conv_lmp_units")
        with col2:
            options['lmp_masses'] = st.checkbox("Include masses", value=True, key="conv_lmp_masses")
            options['lmp_force_skew'] = st.checkbox("Force triclinic (skew)", value=False, key="conv_lmp_skew")

    elif output_format == "CIF":
        with col1:
            options['cif_symprec'] = st.number_input("Symmetry precision", min_value=0.001, max_value=0.1,
                                                     value=0.01, step=0.001, format="%.3f", key="conv_cif_symprec")
        with col2:
            options['cif_refine'] = st.checkbox("Refine structure", value=False, key="conv_cif_refine")

    st.markdown("---")

    st.info(f"Ready to convert {len(structures_to_use)} structure(s) to {output_format}")

    if st.button("🔄 Convert All Files", type="primary", key="convert_structures_btn"):
        st.session_state.converter_files = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        structure_names = list(structures_to_use.keys())
        total_files = len(structure_names)
        conversion_errors = []

        for idx, filename in enumerate(structure_names):
            status_text.text(f"Converting {filename}... ({idx + 1}/{total_files})")

            structure = structures_to_use[filename]
            base_name = filename.rsplit('.', 1)[0]

            content, extension = convert_structure_to_format(structure, output_format, options)

            if content and extension:
                new_filename = f"{base_name}{extension}"
                st.session_state.converter_files[new_filename] = content
            else:
                conversion_errors.append(filename)

            progress_bar.progress((idx + 1) / total_files)

        status_text.empty()
        progress_bar.empty()

        if len(st.session_state.converter_files) > 0:
            st.success(f"✅ Successfully converted {len(st.session_state.converter_files)} file(s)!")
            st.warning(
                f"⚠️ **Please download the ZIP file now!** Use the download button below to save all converted files.")

        if conversion_errors:
            st.error(f"❌ Failed to convert {len(conversion_errors)} file(s): {', '.join(conversion_errors)}")

    if st.session_state.converter_files:
        st.markdown("---")
        st.markdown("#### 💾 Download Converted Files")

        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename, content in st.session_state.converter_files.items():
                zip_file.writestr(filename, content)

        zip_buffer.seek(0)

        format_suffix = output_format.replace(" ", "_").replace("(", "").replace(")", "").lower()
        zip_filename = f"converted_structures_{format_suffix}.zip"

        st.download_button(
            label=f"📦 Download All Files as ZIP ({len(st.session_state.converter_files)} files)",
            data=zip_buffer.getvalue(),
            file_name=zip_filename,
            mime="application/zip",
            type="primary",
            key="conv_download_zip",
            use_container_width=True
        )
