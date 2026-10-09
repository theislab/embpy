import sys
from embpy.embedder import BioEmbedder
import logging

logging.basicConfig(level=logging.INFO)

print(f"embpy manual local test")
print(f"Platform: {sys.platform}")

embedder = BioEmbedder(device="cpu")
print(f"Instantiated BioEmbedder on cpu")

# Test 1: embed a molecule
molecule = "CCO"
print(f"\n--- Testing molecule embedding ---")
print(f"Input SMILES: {molecule}")
mol_emb = embedder.embed_molecule(molecule, model="chemberta2MTR")
print(f"Output shape: {mol_emb.shape}")
print(f"First 5 values: {mol_emb[:5]}")

# Test 2: embed a gene sequence using BioEmbedder directly
# For this we need to use 'esm2_650M' and a protein sequence, or just some DNA sequence with 'enformer_human_rough'
protein_seq = "MKWVTFISLLFLFSSAYSRGVFRR"
print(f"\n--- Testing protein sequence embedding ---")
print(f"Input Protein: {protein_seq}")
prot_emb = embedder.embed_protein(protein_seq, model="esm2_8M")
print(f"Output shape: {prot_emb.shape}")
print(f"First 5 values: {prot_emb[:5]}")

print("\nAll local manual tests passed successfully!")
