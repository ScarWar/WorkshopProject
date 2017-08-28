import os
import sys

from TEMPy.StructureBlurrer import StructureBlurrer
from TEMPy.StructureParser import PDBParser

if len(sys.argv) != 4:
    print(
        "usage: python simulate_maps.py [pdbs directory] [maps directory] \
[resolution]")
else:
    resolution = float(sys.argv[3])
    pdbs_dir = sys.argv[1]
    maps_dir = sys.argv[2]

    for filename in os.listdir(pdbs_dir):
        # Generate a Map instance based on a Gaussian blurring of a protein
        if filename.endswith(".pdb"):

            structure_id = filename[:-4]
            structure_instance = PDBParser.fetch_PDB(
                structure_id, pdbs_dir + filename, hetatm=True, water=False)
            blurrer = StructureBlurrer()
            sim_map = blurrer.gaussian_blur(structure_instance, resolution)
            sim_map = sim_map.normalise()
            sim_map.write_to_MRC_file(maps_dir + structure_id + ".map")

        else:
            continue
