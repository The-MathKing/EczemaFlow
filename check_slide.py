import scanpy as sc
adata = sc.read_h5ad("data/GSE206391/GSE206391_Preprocessed_data.h5")
print(adata.obs.groupby('object_slide').size())
