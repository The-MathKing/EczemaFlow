import anndata as ad
adata = ad.read_h5ad("data/GSE206391/GSE206391_Preprocessed_data.h5")
for p in [1, 2, 3, 5, 6, 8, 11, 15]:
    sub = adata[adata.obs['patient'] == p] if 'patient' in adata.obs else adata[adata.obs['patient_id'] == p]
    samples = sub.obs['sample'].unique() if 'sample' in adata.obs else sub.obs['slide_id'].unique()
    print(f"Patient {p}: {samples.tolist()}")
