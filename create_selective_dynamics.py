import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor
import zipfile
from io import BytesIO


def get_atom_info_dataframe(structure):
    ase_atoms = AseAtomsAdaptor.get_atoms(structure)
    atom_data = []

    inv_cell = np.eye(3)
    if np.linalg.det(ase_atoms.get_cell()) > 1e-9:
        inv_cell = np.linalg.inv(ase_atoms.get_cell())

    for i, atom in enumerate(ase_atoms):
        pos = atom.position
        frac = np.dot(pos, inv_cell)
        atom_data.append({
            "ID": i,
            "Element": atom.symbol,
            "X (Å)": f"{pos[0]:.4f}",
            "Y (Å)": f"{pos[1]:.4f}",
            "Z (Å)": f"{pos[2]:.4f}",
            "Frac X": f"{frac[0]:.4f}",
            "Frac Y": f"{frac[1]:.4f}",
            "Frac Z": f"{frac[2]:.4f}"
        })

    return pd.DataFrame(atom_data)


def write_poscar_with_selective_dynamics(structure, fixed_atoms, fixed_directions, use_fractional=True):
    ase_atoms = AseAtomsAdaptor.get_atoms(structure)

    sio = StringIO()
    write(sio, ase_atoms, format="vasp", direct=use_fractional, sort=True)
    poscar_lines = sio.getvalue().split('\n')

    n_atoms = len(ase_atoms)

    if 'Selective dynamics' not in '\n'.join(poscar_lines[:10]):
        for i, line in enumerate(poscar_lines):
            if line.strip().startswith(('Direct', 'Cartesian', 'direct', 'cartesian')):
                poscar_lines.insert(i, 'Selective dynamics')
                break

    coord_start_idx = None
    for i, line in enumerate(poscar_lines):
        if line.strip().startswith(('Direct', 'Cartesian', 'direct', 'cartesian')):
            coord_start_idx = i + 1
            break

    if coord_start_idx is not None:
        new_lines = poscar_lines[:coord_start_idx]

        for idx in range(n_atoms):
            if coord_start_idx + idx < len(poscar_lines):
                coord_line = poscar_lines[coord_start_idx + idx].strip()

                if coord_line:
                    parts = coord_line.split()
                    if len(parts) >= 3:
                        coords = ' '.join(parts[:3])

                        if idx in fixed_atoms:
                            flags = []
                            for direction in ['x', 'y', 'z']:
                                if direction in fixed_directions:
                                    flags.append('F')
                                else:
                                    flags.append('T')
                            flag_str = ' '.join(flags)
                        else:
                            flag_str = 'T T T'

                        new_lines.append(f"  {coords}   {flag_str}")

        new_lines.extend(poscar_lines[coord_start_idx + n_atoms:])
        poscar_lines = new_lines

    return '\n'.join(poscar_lines)


def render_selective_dynamics_ui(structures_dict, selected_file=None):
    st.markdown("### 🔒 Selective Dynamics for POSCAR")

    if not structures_dict:
        st.warning("No structures available. Upload structures first.")
        return

    is_batch_mode = len(structures_dict) > 1

    if is_batch_mode:
        st.info(f"**Batch Mode**: {len(structures_dict)} structures detected. Settings will apply to all structures.")

    structure_to_analyze = None
    if selected_file and selected_file in structures_dict:
        structure_to_analyze = structures_dict[selected_file]
    elif structures_dict:
        structure_to_analyze = list(structures_dict.values())[0]

    if structure_to_analyze is None:
        st.error("No structure available for analysis")
        return

    available_elements = sorted(list(set(site.specie.symbol for site in structure_to_analyze.sites)))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Fixation Settings")

        fixation_mode = st.radio(
            "Fix atoms by:",
            ["Element Type", "Specific Atom IDs"],
            key="sd_fixation_mode"
        )

        if fixation_mode == "Element Type":
            selected_elements = st.multiselect(
                "Select elements to fix:",
                options=available_elements,
                default=[],
                key="sd_elements"
            )
        else:
            atom_ids_input = st.text_area(
                "Enter atom IDs to fix (comma-separated or ranges):",
                placeholder="e.g., 0,1,2 or 0-5,10,15-20",
                key="sd_atom_ids"
            )

    with col2:
        st.markdown("#### Fixation Directions")

        col_x, col_y, col_z = st.columns(3)

        with col_x:
            fix_x = st.checkbox("Fix X", value=True, key="sd_fix_x")
        with col_y:
            fix_y = st.checkbox("Fix Y", value=True, key="sd_fix_y")
        with col_z:
            fix_z = st.checkbox("Fix Z", value=True, key="sd_fix_z")

        fixed_directions = []
        if fix_x:
            fixed_directions.append('x')
        if fix_y:
            fixed_directions.append('y')
        if fix_z:
            fixed_directions.append('z')

        if not fixed_directions:
            st.warning("⚠️ No directions selected - atoms will be free to move")

    with st.expander("📋 View Atom Information", expanded=False):
        df = get_atom_info_dataframe(structure_to_analyze)
        st.dataframe(df, use_container_width=True, height=400)

        st.markdown("**Element Summary:**")
        element_counts = df['Element'].value_counts().to_dict()
        for element, count in sorted(element_counts.items()):
            st.write(f"• **{element}**: {count} atoms")

    fixed_atoms = set()

    if fixation_mode == "Element Type":
        if selected_elements:
            for i, site in enumerate(structure_to_analyze.sites):
                if site.specie.symbol in selected_elements:
                    fixed_atoms.add(i)

            st.info(f"**Preview**: {len(fixed_atoms)} atoms will be fixed ({', '.join(selected_elements)})")
        else:
            st.warning("No elements selected - all atoms will be free")

    else:
        if atom_ids_input.strip():
            try:
                parsed_ids = set()
                parts = atom_ids_input.strip().split(',')

                for part in parts:
                    part = part.strip()
                    if '-' in part:
                        start, end = part.split('-')
                        parsed_ids.update(range(int(start.strip()), int(end.strip()) + 1))
                    else:
                        parsed_ids.add(int(part))

                max_id = len(structure_to_analyze) - 1
                valid_ids = {i for i in parsed_ids if 0 <= i <= max_id}
                invalid_ids = parsed_ids - valid_ids

                if invalid_ids:
                    st.warning(f"Invalid IDs (out of range): {sorted(invalid_ids)}")

                fixed_atoms = valid_ids

                if fixed_atoms:
                    st.info(
                        f"**Preview**: {len(fixed_atoms)} atoms will be fixed (IDs: {sorted(list(fixed_atoms)[:10])}{'...' if len(fixed_atoms) > 10 else ''})")
                else:
                    st.warning("No valid atom IDs - all atoms will be free")

            except ValueError as e:
                st.error(f"Error parsing atom IDs: {e}")
                fixed_atoms = set()
        else:
            st.warning("No atom IDs specified - all atoms will be free")

    use_fractional = st.checkbox("Use fractional coordinates", value=True, key="sd_fractional")

    st.markdown("---")

    if not is_batch_mode:
        if st.button("💾 Download POSCAR with Selective Dynamics", type="primary", key="sd_download_single"):
            if not fixed_directions:
                st.error("Please select at least one direction to fix")
                return

            try:
                poscar_content = write_poscar_with_selective_dynamics(
                    structure_to_analyze,
                    fixed_atoms,
                    fixed_directions,
                    use_fractional
                )

                filename_base = selected_file.split('.')[0] if selected_file else "structure"
                filename = f"{filename_base}_selective_dynamics.poscar"

                st.download_button(
                    label="💾 Save POSCAR File",
                    data=poscar_content,
                    file_name=filename,
                    mime="text/plain",
                    type="primary",
                    key="sd_actual_download_single"
                )

            except Exception as e:
                st.error(f"Error generating POSCAR: {e}")

    else:
        if st.button("📦 Download All POSCARs with Selective Dynamics (ZIP)", type="primary", key="sd_download_batch"):
            if not fixed_directions:
                st.error("Please select at least one direction to fix")
                return

            try:
                zip_buffer = BytesIO()

                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for structure_name, structure in structures_dict.items():
                        structure_fixed_atoms = set()

                        if fixation_mode == "Element Type":
                            if selected_elements:
                                for i, site in enumerate(structure.sites):
                                    if site.specie.symbol in selected_elements:
                                        structure_fixed_atoms.add(i)
                        else:
                            if atom_ids_input.strip():
                                try:
                                    parsed_ids = set()
                                    parts = atom_ids_input.strip().split(',')

                                    for part in parts:
                                        part = part.strip()
                                        if '-' in part:
                                            start, end = part.split('-')
                                            parsed_ids.update(range(int(start.strip()), int(end.strip()) + 1))
                                        else:
                                            parsed_ids.add(int(part))

                                    max_id = len(structure) - 1
                                    structure_fixed_atoms = {i for i in parsed_ids if 0 <= i <= max_id}

                                except ValueError:
                                    pass

                        poscar_content = write_poscar_with_selective_dynamics(
                            structure,
                            structure_fixed_atoms,
                            fixed_directions,
                            use_fractional
                        )

                        filename_base = structure_name.split('.')[0]
                        filename = f"{filename_base}_selective_dynamics.poscar"

                        zip_file.writestr(filename, poscar_content)

                zip_buffer.seek(0)

                st.download_button(
                    label="💾 Download ZIP Archive",
                    data=zip_buffer.getvalue(),
                    file_name="selective_dynamics_poscars.zip",
                    mime="application/zip",
                    key="sd_actual_download_batch"
                )

                st.success(f"✅ Generated {len(structures_dict)} POSCAR files with selective dynamics")

            except Exception as e:
                st.error(f"Error generating POSCARs: {e}")
