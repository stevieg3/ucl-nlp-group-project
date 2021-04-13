#!/usr/bin/env python3


if __name__ == "__main__":
    import bpython
    bpython.embed(locals_=dict(globals(), **locals()))
