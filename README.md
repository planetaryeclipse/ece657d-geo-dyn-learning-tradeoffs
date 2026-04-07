# ece657d-geo-dyn-learning-tradeoffs

Note that this repo is intended to be run from PyCharm and is configured with an `.idea` repository. You can likely run all the scripts/notebooks in this repository without much issue but you will need to ensure that you have added `.src` to your `PYTHONPATH` as there are classes inside that folder that implemment custom differential geometry.

1. Dataset generation: to generate the datasets, refer to the files inside `src/notbooks/data_gen`. Running the files prefixed with `generate_trainin_data` will allow you to generate the data. However, please note that it was segmented into multiple files so that training could be run across multiple repos due to the processing time taking days to generate.
2. Run training: to generate training results, refer to the files inside `src/notebooks/training`. You will want to run the files prefixed `train_embed_geom_losses_dim_*` and `train_nonembed_dim_*`. You can ignore the `train_sparsity_*` files as this was a failed attempt at utilizing regularization on internal weights of the DynamicsMLP

Please note that the results of analyzing the training results can be found in the notebooks in the base directory. Do not wipe these as they required a custom directory in which the results were sorted. I will try and get these uploaded in some form that you can extract the results.

If there are any problems please don't hesitate to reach out!