# SuperFold

SuperFold is a derivative of [AlphaFold2](https://github.com/deepmind/alphafold), AlphaFold-Multimer, and [ColabFold](https://github.com/sokrypton/ColabFold) with some novel improvements. 
It is intended only for running single-sequence predictions (no MSA) very quickly and takes advantage 
of some time and memory saving hacks found by [krypton](https://github.com/sokrypton) and others in the community. 
This package is intended for use by IPD labs and was written with our computing resources 
(digs, janelia, perlmutter) in mind. 


## Usage

For details on available functions, please refer to the help message (`./superfold --help`)

Basic AlphaFold2 prediction. This will run all 5 AF2-Monomer models with 3 recycles and generate 5 outputs.
`./superfold /path/to/input.pdb`

The script can accept any number of pdb or fasta files listed one after another separated by spaces. 
If pdb files are supplied, it will automatically calculate RMSD of the prediction to the input using 
pymol `align` with 0 cycles. N.B. if your input file contains more than one chain with different 
sequences (a hetero-oligomer), the RMSD calculations will be inaccurate unless you add the 
`--simple_rmsd` flag. Additionally, a separate pdb file may be supplied with the 
`--reference_pdb /path/to/ref.pdb` flag and RMSD calculations will be performed to this structure as 
well using pymol `super` with 0 cycles. 

To run with more recycles, which are useful for predictions of large multi-domain or multi-chain proteins:
`./superfold /path/to/input.pdb --max_recycles 6` 
or whichever number you like. Small de novo proteins can often get away with 0 or 1 recycles. This param
is named `max_recycles` because superfold has limited early-stopping capabilities when used with the 
`--recycle_tol <float>` flag. This will cause the script to stop early if the prediction has converged, 
meaning that the RMSD between the current prediction and the previous (`tol` in the Prediction Results 
file) is less than the value supplied to `--recycle_tol`.

To use the Multimer Model:
`./superfold /path/to/input.pdb --type multimer --version multimer`
The Multimer models were not particular useful for predictions not using MSAs before the Multimer
update. I don't know if the update has impoved this or not. 

## Installation

1) Download this git repo `$git clone git@github.com:rdkibler/superfold_public.git`
2) `$cd superfold`
3) [Download the alphafold weights](#model-parameters) or find an existing path to the weights
4) `$realpath /path/to/alphafold_weights/ > alphafold_weights.pth`. "params/" should be a child dir of the alphafold_weights dir
5) (optional, if you don't want to install pyrosetta) `$conda config --add channels https://username:password@conda.graylab.jhu.edu`
use the username and password provided by Comotion when you licensed it. 
This is technically optional because we don't currently use pyrosetta because
of silent_tools, so you could remove the pyrosetta line from the .yml and be fine
6) `$conda create --name pyroml --file pyroml.yml`
7) `$source activate ~/.conda/envs/pyroml`
8) `$pip install absl-py dm-tree tensorflow ml-collections`
9) `$pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html`
10) `$pip install dm-haiku 
11) All the work is done by the `run_superfold.py` script and this is done by running the `superfold` script. The only job of the `superfold` script is to choose the path to the correct python interpreter. You could edit this file to run on your system(s), or `conda activate pyroml` prior to running `python run_superfold.py`. This is equivalent, but I don't like activating conda envionments if I don't need to.  

### Model parameters

The AlphaFold code and parameters are licensed by DeepMind. Please visit the 
original repo for details. 

The AlphaFold parameters are available from
https://storage.googleapis.com/alphafold/alphafold_params_2021-10-27.tar. This file
contains:

*   5 models which were used during CASP14, and were extensively validated for
    structure prediction quality (see Jumper et al. 2021, Suppl. Methods 1.12
    for details).
*   5 pTM models, which were fine-tuned to produce pTM (predicted TM-score) and
    (PAE) predicted aligned error values alongside their structure predictions
    (see Jumper et al. 2021, Suppl. Methods 1.9.7 for details).
*   5 AlphaFold-Multimer models that produce pTM and PAE values alongside their
    structure predictions.

Download and extract this file and place a path to the directory containing the 
model parameters in a file called `alphafold_weights.pth`

### SuperFold output

The outputs will be saved in the directory provided via the `--output_dir` flag 
(defaults to `output/`). The outputs include the unrelaxed structure, the relaxed structure
if the `--amber_relax` flag is used, a `reports.txt` metadata summary for all predictions if the `--summarize` flag is used
, and prediction metadata for each prediction in individual .json files.

The contents of each output file are as follows:

*   `*_unrelaxed.pdb` – A PDB format text file containing the predicted
    structure with chain IDs rearranged to best match that of the input
    PDB file, if provided. The b-factor column contains the per-residue
    pLDDT scores ranging from `0` to `100`, where `100` means most 
    confident. Note that, because low is better for real experimental 
    B-factor, care must be taken when running applications that interpret
    the B-factor column, such as molecular replacement.
*   `*_relaxed*.pdb` – A PDB format text file containing the predicted
    structure, after performing an Amber relaxation procedure on the unrelaxed
    structure prediction (see Jumper et al. 2021, Suppl. Methods 1.8.6 for
    details). The chain IDs are rearranged and the b-factor column is filled
    like for the `*_unrelaxed.pdb` files.
*   `*_prediction_results.json` – A JSON format text file containing the times taken to run
    each section of the AlphaFold pipeline.
    *   Mean pLDDT scores in `mean_plddt` serve as an overall per-target monomer 
        confidence score and is the average over all per-residue plddts per target. 
        The range of possible values is from `0` to `100`, where `100`
        means most confident). 
    *   Present only if using pTM models: predicted TMalign-score (`pTMscore` field
        contains a scalar). As a predictor of a global superposition metric,
        this score is designed to also assess whether the model is confident in
        the overall domain packing.
    *   Present only if using the `--output_pae` flag and using pTM models: 
        predicted pairwise aligned errors. `pae` contains a NumPy array of 
        shape [N_res, N_res] with the range of possible values from `0` to
        `max_predicted_aligned_error`, where `0` means most confident). This can
        serve for a visualisation of domain packing confidence within the
        structure.
    *   Present only if using pTM models: predicted pairwise aligned error summaries.
        *   `mean_pae`: the mean of the entire [N_res, N_res] pae matrix
        *   `pae_intrachain_X`: The mean of the pae matrix between residues corresponding to chain X
        *   `pae_interchain_XY`: The mean of the pae matrix between residues corresponding to chain X going to residues corresponding to chain Y and vise versa

## Acknowledgements
### Code contributors

*   Ryan Kibler
*   Adam Broerman
*   Phil Leung

### 3rd party libraries and packages
SuperFold communicates with and/or references the following separate libraries
and packages:

*   [Abseil](https://github.com/abseil/abseil-py)
*   [Biopython](https://biopython.org)
*   [Chex](https://github.com/deepmind/chex)
*   [Haiku](https://github.com/deepmind/dm-haiku)
*   [Immutabledict](https://github.com/corenting/immutabledict)
*   [JAX](https://github.com/google/jax/)
*   [matplotlib](https://matplotlib.org/)
*   [ML Collections](https://github.com/google/ml_collections)
*   [NumPy](https://numpy.org)
*   [OpenMM](https://github.com/openmm/openmm)
*   [OpenStructure](https://openstructure.org)
*   [pandas](https://pandas.pydata.org/)
*   [pymol3d](https://github.com/avirshup/py3dmol)
*   PyMol
*   [SciPy](https://scipy.org)
*   [Sonnet](https://github.com/deepmind/sonnet)
*   [TensorFlow](https://github.com/tensorflow/tensorflow)
*   [Tree](https://github.com/deepmind/tree)
*   [tqdm](https://github.com/tqdm/tqdm)

TODO: are there some that need to be added/removed?

We thank all their contributors and maintainers!

## License and Disclaimer

This is not an officially supported Google product.

Copyright 2021 DeepMind Technologies Limited.
