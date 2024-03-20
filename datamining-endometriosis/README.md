Endometriosis and Its Correlation with Lifestyle Factors and Health Indicators
==============
***A Datamining approach using Python and R***

**Authors:** *JONAS STYLBÄCK, ELLA VILLFÖR*

# About

This repository was part of a bachelor thesis exploring possible correlations of lifestyle factors and health indicators with endometriosis. It contains the code necessary to run the types of analyses outlined in the thesis. The thesis report itself can be permanently accessed [here](https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-344208), a copy of the report is also included in this repository.

The code is split up into two pipelines, one using the Python programming language and the other using the R programming language. Each pipeline is accompanied by a utility file containing custom functions for processing and analysing data. Each pipeline generate figures for result interpretation, these can then be found in the `media` directory (not included).

If you're interested in prerequisites, running instructions and the like, take a look at the zipped contents that can be downloaded at the [KTH DiVA portal](https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-344208).

```
repository
├── code
│   ├── pipeline_part_1.ipynb
│   ├── pipeline_part_2.ipynb
│   ├── python_utilities.py
│   └── r_utilities.R
├── data (not included)
├── media (not included)
├── README.md
└── report.pdf
```

# Further development

This repository could benefit from several changes to the codebase, if you're looking to adapt this project to your own needs you might consider the following:

1. Instead of processing the data as a Python dictionary initially, process the data directly into a Pandas DataFrame.
2. Rewrite `pipeline_part_2` in Python with the help of relevant Python modules, such as [Prince](https://github.com/MaxHalford/prince).
3. Rework some of the functions in `python_utilities.py` to reduce abstraction.
4. Consider expanding the multiple correspondence analysis to generate factor maps for other orthogonal planes alongside the first.

# Experience Gained

This was my first dive into data mining and R. I learned plenty of new Python tools for data analysis such as SciPy and Pandas, I also learned the importance of having high-quality data if any real insights are to be gleaned from the analysis. Additionally, I learned more about project management and how to create a virtual, hybrid environment for Python and R within Jupyter.