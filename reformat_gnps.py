""" reformat_nist_sdf.py

Reformat the NIST sdf file

"""

from pathlib import Path
import re
import argparse
import pandas as pd
import numpy as np
from typing import Iterator, List, Tuple
from itertools import groupby
from functools import partial
from collections import defaultdict
import json
import ms_pred.common as common

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from tqdm import tqdm
from pathos import multiprocessing as mp

NAME_STRING = r"<(.*)>"
COLLISION_REGEX = "([0-9]+\.[0-9]+)"
VALID_ELS = set(["C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F", "Br", "B",
                 "Se", "Fe", "Co", "As", "Na", "K"])
ION_MAP = {'[M+H-H2O]+': '[M-H2O+H]+',
           '[M+NH4]+': '[M+H3N+H]+',
           '[M+H-2H2O]+': '[M-H4O2+H]+', }
DEBUG_ENTRIES = 10000


def get_els(form):
    return {i[0] for i in re.findall("([A-Z][a-z]*)([0-9]*)", form)}


def chunked_parallel(input_list, function, chunks=100, max_cpu=16):
    """chunked_parallel

    Args:
        input_list : list of objects to apply function
        function : Callable with 1 input and returning a single value
        chunks: number of hcunks
        max_cpu: Max num cpus
    """

    cpus = min(mp.cpu_count(), max_cpu)
    pool = mp.Pool(processes=cpus)

    def batch_func(list_inputs):
        outputs = []
        for i in list_inputs:
            outputs.append(function(i))
        return outputs

    list_len = len(input_list)
    num_chunks = min(list_len, chunks)
    step_size = len(input_list) // num_chunks

    chunked_list = [
        input_list[i: i + step_size] for i in range(0, len(input_list), step_size)
    ]

    list_outputs = list(tqdm(pool.imap(batch_func, chunked_list), total=num_chunks))
    full_output = [j for i in list_outputs for j in i]
    return full_output


def build_mgf_str(
        meta_spec_list: List[Tuple[dict, List[Tuple[str, np.ndarray]]]],
        merge_charges=True,
        parent_mass_keys=("PEPMASS", "parentmass", "PRECURSOR_MZ"),
        precision=4,
) -> str:
    """build_mgf_str.

    Args:
        meta_spec_list (List[Tuple[dict, List[Tuple[str, np.ndarray]]]]): meta_spec_list

    Returns:
        str:
    """
    entries = []
    for meta, spec in tqdm(meta_spec_list):
        str_rows = ["BEGIN IONS"]

        # Try to add precusor mass
        for i in parent_mass_keys:
            if i in meta:
                pep_mass = float(meta.get(i, -100))
                str_rows.append(f"PEPMASS={pep_mass}")
                break

        for k, v in meta.items():
            str_rows.append(f"{k.upper().replace(' ', '_')}={v}")

        if merge_charges:
            spec_ar = np.vstack([i[1] for i in spec])
            mz_to_inten = {}
            for i, j in spec_ar:
                i = np.round(i, precision)
                mz_to_inten[i] = mz_to_inten.get(i, 0) + j

            spec_ar = [[i, j] for i, j in mz_to_inten.items()]
            spec_ar = np.vstack([i for i in sorted(spec_ar, key=lambda x: x[0])])


        else:
            raise NotImplementedError()
        str_rows.extend([f"{i} {j}" for i, j in spec_ar])
        str_rows.append("END IONS")

        str_out = "\n".join(str_rows)
        entries.append(str_out)

    full_out = "\n\n".join(entries)
    return full_out


def uncharged_formula(mol, mol_type="mol") -> str:
    """Compute uncharged formula"""
    if mol_type == "mol":
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "smiles":
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    else:
        raise ValueError()

    return re.findall(r"^([^\+,^\-]*)", chem_formula)[0]


def process_mgf(group: Iterator, meta_data: pd.DataFrame):
    """process_mgf.

    Parses spectra in the MGF file formate, with

    Args:
        group (Iterator) : One group from MGF file
        meta_data (pd.DataFrame): Mass Spec meta data
    """
    output_dict = dict()
    # Note: Sometimes we have multiple scans
    # This mgf has them collapsed
    cur_spectra = []
    spec_id = ''
    group = list(group)
    for line in group:
        line = line.strip()
        if not line:
            pass
        elif line == "END IONS" or line == "BEGIN IONS":
            pass
        elif "=" in line:
            k, v = [i.strip() for i in line.split("=", 1)]
            if k == 'TITLE':
                spec_id = v
        else:
            mz, intens = line.split()
            cur_spectra.append((float(mz), float(intens)))

    if len(cur_spectra) == 0:
        # print(f'Warning: no spectrum found for entry {group}')
        return {}
    if len(cur_spectra) == 1:
        # print(f'Warning: single peak found for entry {group}, skipping')
        return {}
    cur_spectra = np.vstack(cur_spectra)
    output_dict["Peaks"] = cur_spectra

    if len(spec_id) == 0:
        raise ValueError(f'no TITLE/Spec_ID entry found for {group}')
    meta_entry = meta_data.query(f"spectrum_id == \"{spec_id}\"")
    assert meta_entry.shape[0] == 1
    for key in meta_entry.columns:
        val = meta_entry[key].values[0]
        if key == 'spectrum_id':
            output_dict['spec_id'] = val
        elif key == 'InChIKey_smiles':
            output_dict['InChIKey'] = val
        else:
            output_dict[key] = str(val)

    # Apply filter
    if fails_filter(output_dict):
        return {}

    formula = uncharged_formula(output_dict['Smiles'], mol_type="smiles")
    output_dict['FORMULA'] = formula
    if formula is None:
        return {}

    form_els = get_els(output_dict['FORMULA'])
    if len(form_els.intersection(VALID_ELS)) != len(form_els):
        return {}

    return output_dict


def merge_data(collision_dict: dict):
    base_dict = None
    out_peaks = {}
    num_peaks = 0
    energies = []
    for energy, sub_dict in collision_dict.items():
        if base_dict is None:
            base_dict = sub_dict
        if energy in out_peaks:
            raise ValueError(f"Unexpected to see {energy} in {json.dumps(sub_dict, indent=2)}")
        out_peaks[energy] = np.array(sub_dict["Peaks"])
        energies.append(energy)
        num_peaks += len(out_peaks[energy])

    base_dict["Peaks"] = out_peaks
    base_dict["collision_energy"] = energies
    base_dict["NUM PEAKS"] = num_peaks

    peak_list = list(base_dict.pop("Peaks").items())
    info_dict = base_dict
    return (info_dict, peak_list)


def dump_to_file(entry: tuple, out_folder) -> dict:
    # Create output entry
    entry, peaks = entry
    output_name = entry["spec_id"]
    common_name = entry.get("Compound_Name", "")
    formula = entry["FORMULA"]
    ionization = entry["Adduct"]
    parent_mass = entry["Precursor_MZ"]
    out_entry = {
        "dataset": "gnps_2023",
        "spec": output_name,
        "name": common_name,
        "formula": formula,
        "ionization": ionization,
        "smiles": entry["Smiles"],
        "inchikey": entry["InChIKey"],
        "precursor": parent_mass,
    }

    # create_output_file
    # All keys to exclude from the comments
    exclude_comments = {"Peaks"}

    output_name = Path(out_folder) / f"{output_name}.ms"
    header_str = [
        f">compound {common_name}",
        f">formula {formula}",
        f">ionization {ionization}",
        f">parentmass {parent_mass}",
    ]
    header_str = "\n".join(header_str)
    comment_str = "\n".join(
        [f"#{k} {v}" for k, v in entry.items() if k not in exclude_comments]
    )

    # Maps collision energy to peak set
    peak_list = []
    for k, v in peaks:
        peak_entry = []
        peak_entry.append(f">collision {k}")
        peak_entry.extend([f"{row[0]} {row[1]}" for row in v])
        peak_list.append("\n".join(peak_entry))

    peak_str = "\n\n".join(peak_list)
    with open(output_name, "w") as fp:
        fp.write(header_str + "\n")
        fp.write(comment_str + "\n\n")
        fp.write(peak_str)
    return out_entry


def read_mgf(input_file, debug=False):
    key = lambda x: x.strip() == "BEGIN IONS"
    groups_to_process = []
    print("Reading in file")
    with open(input_file, "r") as fp:
        for index, (is_header, group) in tqdm(enumerate(groupby(fp, key))):
            if is_header:
                continue
            else:
                groups_to_process.append(list(group))
            if debug and index > DEBUG_ENTRIES:
                break
    return groups_to_process


def fails_filter(entry, valid_adduct=list(common.ion2mass.keys()),
                 max_mass=1500,
                 ):
    """ fails_filter. """
    if entry['Adduct'] not in valid_adduct:
        return True

    if float(entry['Precursor_MZ']) > max_mass:
        return True

    # QTOF, HCD,
    # if entry['INSTRUMENT TYPE'].upper() != "HCD":
    #     return True

    if entry['Smiles'] == 'nan':
        return True

    if entry['ppmBetweenExpAndThMass'] == 'nan' or float(entry['ppmBetweenExpAndThMass']) > 50:
        return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--spec-file", action="store",
                        default="")
    parser.add_argument("--meta-file", action="store",
                        default="")
    parser.add_argument("--workers", action="store",
                        default=32)
    parser.add_argument("--targ-dir", action="store",
                        default="../processed_data/")
    args = parser.parse_args()
    debug = args.debug
    workers = args.workers

    target_directory = args.targ_dir
    spec_file = args.spec_file
    meta_file = args.meta_file

    if len(spec_file) == 0 or len(meta_file) == 0:
        raise FileNotFoundError('Please specify the input file!')

    if debug:
        target_directory = Path(target_directory) / "debug"
    else:
        target_directory = Path(target_directory)

    target_directory.mkdir(exist_ok=True, parents=True)
    target_ms = target_directory / "spec_files"
    target_labels = target_directory / "labels.tsv"

    target_ms.mkdir(exist_ok=True, parents=True)

    groups_to_process = read_mgf(spec_file, debug=debug)
    meta_data = pd.read_csv(meta_file)
    if debug:
        meta_data = meta_data

    process_temp = partial(process_mgf, meta_data=meta_data)

    print("Parallelizing smiles processing")
    if debug:
        output_dicts = [process_temp(i) for i in tqdm(groups_to_process)]
    else:
        output_dicts = chunked_parallel(groups_to_process, process_temp,
                                        1000, max_cpu=workers)

    # Reformat output dicts
    # {inchikey: {adduct: {instrument: {compound source: {collision type: {collision energy: spectra} } } } } }
    parsed_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))))
    )

    print("Shuffling dict before merge")
    for output_dict in tqdm(output_dicts):
        if len(output_dict) == 0:
            continue
        inchikey = output_dict["InChIKey"]
        precusor_type = output_dict["Adduct"]
        instrument_type = output_dict["msMassAnalyzer"]
        compound_source = output_dict["Compound_Source"]
        collision_type = output_dict["msDissociationMethod"]
        collision_energy = output_dict["collision_energy"]
        col_energies = re.findall(COLLISION_REGEX, collision_energy)
        if len(col_energies) > 0:
            collision_energy = col_energies[-1]

        # Assign a new key e.g. '> collision 35.0 (1)' to duplicated entries
        if collision_energy in parsed_data[inchikey][precusor_type][instrument_type][compound_source][collision_type]:
            assigned_new_key = False
            i = 0
            while True:
                i += 1
                new_ce_key = f'{collision_energy} ({i})'
                if new_ce_key not in parsed_data[inchikey][precusor_type][instrument_type][compound_source][collision_type]:
                    collision_energy = new_ce_key
                    assigned_new_key = True
                    break
            if not assigned_new_key:
                raise RuntimeError(f'Failed to assign a unique collision energy key for '
                                   f'InChIKey={inchikey}, {collision_energy}')

        parsed_data[inchikey][precusor_type][instrument_type][compound_source][collision_type][
            collision_energy
        ] = output_dict

    # merge entries
    merged_entries = []
    print("Merging dicts")
    for inchikey, adduct_dict in tqdm(parsed_data.items()):
        for adduct, instrument_dict in adduct_dict.items():
            for instrument, source_dict in instrument_dict.items():
                for source, colli_type_dict in source_dict.items():
                    for coll_type, colli_eng_type in colli_type_dict.items():
                        output_dict = merge_data(colli_eng_type)
                        merged_entries.append(output_dict)

    print(f"Parallelizing export to file")
    dump_fn = partial(dump_to_file, out_folder=target_ms)
    if debug:
        output_entries = [dump_fn(i) for i in merged_entries]
    else:
        output_entries = chunked_parallel(merged_entries, dump_fn, 10000,
                                          max_cpu=workers)

    df = pd.DataFrame(output_entries)

    # Transform ions
    df['ionization'] = [ION_MAP.get(i, i) for i in df['ionization'].values]
    df.to_csv(target_labels, sep="\t", index=False)
