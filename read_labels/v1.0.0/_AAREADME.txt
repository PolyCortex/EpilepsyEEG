File: _AAREADME.txt
Software: NEDC Print Labels (v1.0.0)

---
Change Log:

 (20190417) initial release.

---


Requirements:

  - Python 2.7x

How to Run NEDC Print Labels:

  - Run `nedc_print_labels.py` under `src/` with Python:

    * A demonstration of this software can be found in run.sh

      ~ This demonstration can be executed through `sh run.sh`
      
    * An Example Usage: `python src/nedc_print_labels.py x1.tse`
    
      ~ `python src/nedc_print_labels.py -help` to display a help message

Data Format:

  - NEDC Print Labels takes either .tse or .lbl files as input, or list files

    * A list file is a file where each line corresponds to a .tse or .lbl file

Processing Annotations:

  - Annotations can be loaded directly into memory using ann.get()

    * See Line 130 for example code.

    * ann.get() takes 3 arguments to specify which information will be loaded

      ~ Level: Specifies which level to load (NOTE: level is 0 for
                tse files)

      ~ Sublevel: Specifies which sublevel to load (NOTE: sublevel is 0 for
                   tse files)

      ~ Channel: Specifies which channel to load (NOTE: channel is -1 for
                  tse files)

    * Gets loaded into a Python List structure.

      ~ Each element is another list, a single annotation

      ~ e.g. [start, stop, {symbol1: prob1, symbol2: prob2, ... }]

Directories:

  - src: This directory includes all Python code for running this software.

  - src/help: This directory contains help and usage files to NEDC Print Labels

  - src/sys_tools: This directory holds NEDC system modules required to run
                    NEDC Print Labels.
