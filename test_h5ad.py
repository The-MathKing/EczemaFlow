import scanpy as sc
import sys
print("Loading 1...")
a = sc.read_h5ad("data/GSE206391/GSE206391_Preprocessed_data.h5")
print("Loading 2...")
b = sc.read_h5ad("data/GSE206391/GSE206391_Preprocessed_data.h5")
print("Success")
