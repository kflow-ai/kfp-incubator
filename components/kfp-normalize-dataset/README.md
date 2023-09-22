# kfp-normalize-dataset

This is a [Kubeflow Pipeline Component][kfp] that accepts a Platform UI dataset and ensures
that it consists of individual files for use in further processing.

## Setup

This repo follows the conventions of [scripts-to-rule-them-all].

The setup scripts have only been tested on Apple Silicon Macs. To install
dependencies and configure the project for setup, simply run `script/setup`.

## Updating dependencies

[kfp]: https://www.kubeflow.org/docs/components/pipelines/v1/sdk/component-development/
[scripts-to-rule-them-all]: https://github.com/github/scripts-to-rule-them-all

## Running tests

To run the full test suite, simply run `script/test`. As a side effect, the
script will automatically install dependencies, run formatters and linters,
and report code coverage.
