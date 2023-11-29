# AffectiveVR

`[Last update: November 29, 2023]`

***
    Period:     2023-11 - ...
    Status:     work in progress

    Author(s):  Lucy Roellecke
    Contact:    lucy.roellecke@fu-berlin.de

***

*In general, one can add README's in nearly every folder. The guiding principle should always be that any person who is not familiar with the project can find their way exclusively via the README's – 'This may be you one day'*

## Project description

*A brief general description of the project.*

## Project structure

*A brief description of the folder structure of the project (Where is what?). Anticipate new lab members who suppose to be able to orientate within this structure without your help. At the same time, avoid too detailed descriptions. Down the folder structure, there suppose to be further README's explaining subsequent folders & data.*

## Install research code as package

In case, there is no project-related virtual / conda environment yet, create one for the project:

```shell
conda create -n AVR_3.11 python=3.11
```

And activate it:

```shell
conda activate AVR_3.11
```

Then install the code of the research project as python package:

```shell
# assuming your current working dircetory is the project root
pip install -e ".[develop]"
```

**Note**: The `-e` flag installs the package in editable mode,
i.e., changes to the code will be directly reflected in the installed package.
Moreover, the code keeps its access to the research data in the underlying folder structure.
Thus, the `-e` flag is recommended to use.

*R*-projects should be initialized in the project root `.` with, e.g., `RStudio` as *existing directory*.
Corresponding *R*-scripts can be stored in `./code/Rscripts/`

Similarly, use this structure for Matlab or other programming languages, which are employed in this project.

## Publications

*List publications resulted from this project (including papers, posters, talks, ...)*

## Preregistration

*If applicable, was the project pre-registered and if yes, when and where (link)*

## Contributors/Collaborators

*Name people who are involved in this project, their position and/or contribution. Optional: add contact data*
