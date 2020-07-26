.. automodule:: pegasus
    :noindex:

API
===

*Pegasus* can also be used as a python package. Import pegasus by::

	import pegasus as pg

Analysis Tools
--------------

Read and Write
~~~~~~~~~~~~~~

.. autosummary::
	:toctree: .

	read_input
	write_output
	aggregate_matrices

Preprocess
~~~~~~~~~~

.. autosummary::
	:toctree: .

	qc_metrics
	get_filter_stats
	filter_data
    identify_robust_genes
	log_norm
	highly_variable_features
	select_features
	pca


Batch Correction
~~~~~~~~~~~~~~~~

.. autosummary::
	:toctree: .

	set_group_attribute
	correct_batch
	run_harmony

Nearest Neighbors
~~~~~~~~~~~~~~~~~

.. autosummary::
	:toctree: .

	neighbors
	calc_kBET
	calc_kSIM

Diffusion Map
~~~~~~~~~~~~~

.. autosummary::
	:toctree: .

	diffmap
	reduce_diffmap_to_3d
	calc_pseudotime
	infer_path


Cluster algorithms
~~~~~~~~~~~~~~~~~~

.. autosummary::
	:toctree: .

	cluster
	louvain
	leiden
	spectral_louvain
	spectral_leiden

Visualization Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
	:toctree: .

	tsne
	fitsne
	umap
	fle
	net_tsne
	net_fitsne
	net_umap
	net_fle

Differential Expression Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
	:toctree: .

	de_analysis
	markers
	write_results_to_excel

Marker Detection based on Gradient Boost Machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
	:toctree: .

	find_markers

Annotate clusters:
------------------

.. autosummary::
	:toctree: .

	infer_cell_types
	annotate_cluster

Plotting
--------

Static Plots
~~~~~~~~~~~~~~~~~

.. autosummary::
	:toctree: .

	scatter
    scatter_groups
	compo_plot
    violin
	heatmap
	dotplot
    dendrogram

Interactive Plots
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: .

    volcano


Miscellaneous
-------------

.. autosummary::
	:toctree: .

	calc_signature_score
	search_genes
	search_de_genes
