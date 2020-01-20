import time
import numpy as np
from scipy.sparse import issparse
from anndata import AnnData
import logging

from pegasus.tools import estimate_feature_statistics, select_features, X_from_rep

logger = logging.getLogger("pegasus")
from pegasus.utils import decorators as pg_deco



def set_group_attribute(data: AnnData, attribute_string: str) -> None:
    """Set group attributes used in batch correction.

    Batch correction assumes the differences in gene expression between channels are due to batch effects. However, in many cases, we know that channels can be partitioned into several groups and each group is biologically different from others. In this case, *pegasus* will only perform batch correction for channels within each group.

    Parameters
    ----------
    data: ``anndata.AnnData``
        Annotated data matrix with rows for cells and columns for genes.

    attribute_string: ``str``
        Attributes used to construct groups:

        * ``None``
            Assume all channels are from one group.

        * ``attr`` 
            Define groups by sample attribute ``attr``, which is a keyword in ``data.obs``.

        * ``att1+att2+...+attrn`` 
            Define groups by the Cartesian product of these *n* attributes, which are keywords in ``data.obs``.

        * ``attr=value_11,...value_1n_1;value_21,...value_2n_2;...;value_m1,...,value_mn_m``
            In this form, there will be *(m+1)* groups. A cell belongs to group *i* (*i > 1*) if and only if its sample attribute ``attr``, which is a keyword in ``data.obs``, has a value among ``value_i1``, ... ``value_in_i``. A cell belongs to group 0 if it does not belong to any other groups.

    Returns
    -------
    None

        Update ``data.obs``:
        
        * ``data.obs["Group"]``: Group ID for each cell.

    Examples
    --------

    >>> pg.set_group_attribute(adata, attr_string = "Individual")

    >>> pg.set_group_attribute(adata, attr_string = "Individual+assignment")

    >>> pg.set_group_attribute(adata, attr_string = "Channel=1,3,5;2,4,6,8")    
    """
    
    if attribute_string.find("=") >= 0:
        attr, value_str = attribute_string.split("=")
        assert attr in data.obs.columns
        values = value_str.split(";")
        data.obs["Group"] = "0"
        for group_id, value in enumerate(values):
            vals = value.split(",")
            idx = np.isin(data.obs[attr], vals)
            data.obs.loc[idx, "Group"] = str(group_id + 1)
    elif attribute_string.find("+") >= 0:
        attrs = attribute_string.split("+")
        assert np.isin(attrs, data.obs.columns).sum() == len(attrs)
        data.obs["Group"] = data.obs[attrs].apply(lambda x: "+".join(x), axis=1)
    else:
        assert attribute_string in data.obs.columns
        data.obs["Group"] = data.obs[attribute_string]

def estimate_adjustment_matrices(data: AnnData) -> bool:
    """ Estimate adjustment matrices
    """

    if "plus" in data.varm.keys() or "muls" in data.varm.keys():
        # This only happens if this is for subclustering. Thus do not calculate factors, using factors calculated from parent for batch correction.
        assert "plus" in data.varm.keys() and "muls" in data.varm.keys()
        return True

    if ("gmeans" not in data.varm) or ("gstds" not in data.varm):
        estimate_feature_statistics(data, True)

    if data.uns["Channels"].size == 1:
        logger.warning(
            "Warning: data only contains 1 channel. Batch correction disabled!"
        )
        return False

    nchannel = data.uns["Channels"].size

    plus = np.zeros((data.shape[1], nchannel))
    muls = np.zeros((data.shape[1], nchannel))

    ncells = data.uns["ncells"]
    means = data.varm["means"]
    partial_sum = data.varm["partial_sum"]
    gmeans = data.varm["gmeans"]
    gstds = data.varm["gstds"]
    c2gid = data.uns["c2gid"]
    for i in range(data.uns["Channels"].size):
        if ncells[i] > 1:
            muls[:, i] = (partial_sum[:, i] / (ncells[i] - 1.0)) ** 0.5
        outliers = muls[:, i] < 1e-6
        normals = np.logical_not(outliers)
        muls[outliers, i] = 1.0
        muls[normals, i] = gstds[normals, c2gid[i]] / muls[normals, i]
        plus[:, i] = gmeans[:, c2gid[i]] - muls[:, i] * means[:, i]

    data.varm["plus"] = plus
    data.varm["muls"] = muls

    return True

def correct_batch_effects(data: AnnData, keyword: str, features: str = None) -> None:
    """ Apply calculated plus and muls to correct batch effects for a dense matrix
    """
    X = data.uns[keyword]
    m = X.shape[1]
    if features is not None:
        selected = data.var[features].values
        plus = data.varm["plus"][selected, :]
        muls = data.varm["muls"][selected, :]
    else:
        selected = np.ones(data.shape[1], dtype=bool)
        plus = data.varm["plus"]
        muls = data.varm["muls"]

    for i, channel in enumerate(data.uns["Channels"]):
        idx = np.isin(data.obs["Channel"], channel)
        if idx.sum() == 0:
            continue
        X[idx] = X[idx] * np.reshape(muls[:, i], newshape=(1, m)) + np.reshape(
            plus[:, i], newshape=(1, m)
        )
    # X[X < 0.0] = 0.0

def correct_batch(data: AnnData, features: str = None) -> None:
    """Batch correction on data.

    Parameters
    ----------
    data: ``anndata.AnnData``
        Annotated data matrix with rows for cells and columns for genes.

    features: `str`, optional, default: ``None``
        Features to be included in batch correction computation. If ``None``, simply consider all features.

    Returns
    -------
    ``None``

    Update ``data.X`` by the corrected count matrix.

    Examples
    --------
    >>> pg.correct_batch(adata, features = "highly_variable_features")
    """

    tot_seconds = 0.0

    # estimate adjustment parameters
    start = time.perf_counter()
    can_correct = estimate_adjustment_matrices(data)
    end = time.perf_counter()
    tot_seconds += end - start
    logger.info("Adjustment parameters are estimated.")

    # select dense matrix
    keyword = select_features(data, features)
    logger.info("Features are selected.")

    if can_correct:
        start = time.perf_counter()
        correct_batch_effects(data, keyword, features)
        end = time.perf_counter()
        tot_seconds += end - start
        logger.info(
            "Batch correction is finished. Time spent = {:.2f}s.".format(tot_seconds)
        )



@pg_deco.TimeLogger()
def run_harmony(
    data: AnnData,
    rep: str = 'pca',
    n_jobs: int = -1,
    n_clusters: int = None,
    random_state: int = 0,
) -> str:
    """Batch correction PCs using Harmony

    Parameters
    ----------
    data: ``anndata.AnnData``.
        Annotated data matrix with rows for cells and columns for genes.

    rep: ``str``, optional, default: ``"pca"``.
        Which representation to use as input of Harmony, default is PCA.

    n_jobs : ``int``, optional, default: ``-1``.
        Number of threads to use for the KMeans clustering used in Harmony. ``-1`` refers to using all available threads.

    n_clusters: ``int``, optional, default: ``None``.
        Number of Harmony clusters. Default is ``None``, which asks Harmony to estimate this number from the data.

    random_state: ``int``, optional, default: ``0``.
        Seed for random number generator

    Returns
    -------
    out_rep: ``str``
        The keyword in ``data.obsm`` referring to the embedding calculated by Harmony algorithm.

    Update ``data.obsm``:
        * ``data.obsm['X_' + out_rep]``: The embedding calculated by Harmony algorithm.

    Examples
    --------
    >>> pg.run_harmony(adata, rep = "pca", n_jobs = 10, random_state = 25)
    """
    try:
        from harmony import harmonize
    except ImportError:
        print("Need harmony! Try 'pip install harmony-pytorch'.")
    
    logger.info("Start integration using Harmony.")    
    out_rep = rep + '_harmony'
    data.obsm['X_' + out_rep] = harmonize(X_from_rep(data, rep), data.obs, 'Channel', n_clusters = n_clusters, n_jobs_kmeans = n_jobs, random_state = random_state)
    return out_rep
