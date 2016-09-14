# Kaggle Bimbo xgboost

This project contains a solution for Kaggle Grupo Bimbo challenge. The solution itself is not designed
to get a good score (for that I have a complete solution, not really pretty one), but to try out if Luigi
can be a good tool for competitions.

## Notes

For running tasks I use Luigi. It's great for making tasks granular so that you dump all the results after the task
which guarantees you won't struggle with memory leaks.

## TODO

I'm planning to work on this more, both on Luigi plumbing and feature generation. Not sure when though.

## Usage

You can use any python distribution, whether its dockerized or not.

Here's what I'm doing with Anaconda installation:
```
~/anaconda/bin/python -m venv . --without-pip --system-site-packages
curl https://bootstrap.pypa.io/get-pip.py | ./bin/python
source ./bin/activate
```
