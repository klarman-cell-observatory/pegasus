try:
    get_ipython
except NameError:
    import matplotlib
    matplotlib.use("Agg")

from .io import infer_file_format, read_input, write_output
from .tools import (aggregate_matrices, qc_metrics, get_filter_stats, 
filter_data, log_norm, select_features, pca, highly_variable_features, 
set_group_attribute, correct_batch, neighbors, calc_kBET, calc_kSIM, 
diffmap, reduce_diffmap_to_3d, calc_pseudotime, 
louvain, leiden, spectral_louvain, spectral_leiden, 
tsne, fitsne, umap, fle, net_tsne, net_fitsne, net_umap, net_fle,
de_analysis, markers, write_results_to_excel, find_markers, infer_path)
from .annotate_cluster import infer_cell_types, annotate
from .demuxEM import estimate_background_probs, demultiplex
from .misc import search_genes, search_de_genes

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
