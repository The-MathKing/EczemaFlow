import scanpy as sc
import pandas as pd
adata = sc.read_h5ad("data/GSE206391/GSE206391_Preprocessed_data.h5")
df = adata.obs[['object_slide', 'sample']].drop_duplicates()
slides = ['3-V19S23-005', '8-V19T12-006', '12-V19T12-021', '14-V19T12-024', 'SN-V11J13-120', 'SN-V11J13-122']
for sl in slides:
    print(f"'{sl}': {list(df[df['object_slide'] == sl]['sample'])}")
