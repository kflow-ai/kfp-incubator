# kfp-vectorize-dataset

This is a [Kubeflow Pipeline Component][kfp] that accepts a Platform UI dataset,
chunks each file in the dataset, vectorizes each chunk, and inserts the vectors
into a vector database.

If available, this component will use [Ray] to parallelize the vectorization process.

## Setup

This repo follows the conventions of [scripts-to-rule-them-all].

The setup scripts have only been tested on Apple Silicon Macs. To install
dependencies and configure the project for setup, simply run `script/setup`.

## Updating dependencies

[kfp]: https://www.kubeflow.org/docs/components/pipelines/v1/sdk/component-development/
[scripts-to-rule-them-all]: https://github.com/github/scripts-to-rule-them-all
[Ray]: https://ray.io/

## Running tests

To run the full test suite, simply run `script/test`. As a side effect, the
script will automatically install dependencies, run formatters and linters,
and report code coverage.
