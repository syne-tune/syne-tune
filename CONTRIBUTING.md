# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

If you like to contribute a new scheduler, here is a
[tutorial](docs/tutorials/developer/README.md) how to do so.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment


## Contributing via Pull Requests

Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You ran `./dev_setup.sh` to automatically activate the right code formatting
2. You are working against the latest source on the *main* branch.
3. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
4. You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request, please:

1. Fork the repository. (Note, core members can directly push to a branch on the repository)
2. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Ensure local tests pass.
4. Pull request title and message must adhere to [Conventional Commits](https://www.conventionalcommits.org). This means that the title must begin with `feat(scope): title`, `fix(scope): title`, `docs(scope): title`, `refactor(scope): title` or `chore(scope): title`. Including `(scope)` is optional.
    * `feat`: indicates a feature added. For example, `feat(benchmarking): benchmark ASHA` 
    * `fix`: indicates a bug fixes. Be sure to describe the problem in the PR description.
    * `docs`: indicates updated documentation (docstrings or Markdown files)
    * `refactor`: indicates a feature-preserving refactoring
    * `chore`: something without directly visible user benefit. Typically used for build scripts, config.
5. For bugs, describe the bug, root cause, solution, potential alternatives considered but discarded.
6. For features, describe use case, most salient design aspects (especially if new), potential alternatives.
7. Pull request message should indicate which issues are fixed: `fixes #<issue>` or `closes #<issue>`.
8. If not obvious (i.e. from unit tests), describe how you verified that your change works.
9. If this PR includes breaking changes, they must be listed at the end in the following format
  (notice how multiple breaking changes should be formatted):

  ```
  BREAKING CHANGE: Description of what broke and how to achieve this behavior now
  * **scope-name:** Another breaking change
  * **scope-name:** Yet another breaking change
  ```
10. Send us a pull request, answering any default questions in the pull request interface.
11. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.
12. Make sure to update the PR title/description if things change. The PR title/description are going to be used as the commit title/message and will appear in the CHANGELOG, so maintain them all the way throughout the process.


GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).


## Finding contributions to work on

Looking at the existing issues is a great way to find something to contribute on. As our
projects, by default, use the default GitHub issue labels
(enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any
'help wanted' issues is a great place to start.

## Adding/updating Syne Tune examples

We are always interested in adding examples that demonstrate how to use Syne Tune.

Please add your example inside the `examples/` folder. 

If you create an example notebook, please put it under the `examples/notebooks` folder.

Please ensure that your example assumes no prior knowledge of hyperparameter optimization, to ensure that the library remains accessible to new users. 


## Contributing to the Documentation

We also greatly value contributions to our [documentation](https://github.com/awslabs/syne-tune/tree/main/docs/source).
In fact, some of our tutorials have been developed by external contributors. You
can build the docs locally (assuming you have installed Syne Tune as `dev` or
`extra`):

```bash
cd docs
rm -rf source/_apidoc
make clean
make html
```

Then, load `docs/build/html/index.html` in your browser. Here are some further
hints:

* When looking at a pull request on GitHub, you can also explore the documentation
  that this PR implies. Locate the check **docs/readthedocs.org:syne-tune**, click
  on `Details`, then on the small `View docs` link just under `Build took X seconds`
  (not the tall `View docs` link in the upper right).


Jupyter notebooks inside the `examples/notebooks` folder can be easily added as documentation pages. 
See `tune_xgboost.ipynb` for an example showing how this can be done. 


## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security issue notifications

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.
