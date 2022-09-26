# pytorch-template

My personal starting point for a computer vision deep learning project. It tries to automate the general structure of your codebase to some degree, such that you can spend your time dealing mostly with filling in the "content" of data loaders, visualizations, network architectures, logging, etc. rather than boilerplate code.

## Notes

* Even though I am importing a lot of things that I use somewhat often in `__init__.py`, feel free to prune and change this list, because the template itself is designed to rely as little as possible on obscure libraries or other things that tend to break and render your codebase unusable a year or so down the line.
