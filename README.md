# AffectiveVR

`[Last update: December 13, 2023]`

    Period:     2023-10 - 2024-09
    Status:     work in progress

    Author(s):  Lucy Roellecke
    Contact:    lucy.roellecke@fu-berlin.de


## Project description

*Affective VR* (AVR) aims to develop and test a tool for continous emotion ratings. The project proposes such a tool and assesses its effectiveness, usability and reliability using videos presented in virtual reality (VR).

## Project structure
The AVR project consists of three main stages: In **phase 1**, three different rating methods ('Grid', 'Flubber' and 'Proprioceptive') were tested with different videos in VR of each 1 min length. In **phase 2**, the 'Flubber' as winning rating method from phase 1 was tested for a longer VR experience of about 20 min with different videos playing after one another. In **phase 3**, the very same stimuli and rating method are used but additionally to the behavioral data, EEG and periphysiological data are acquired.

**Code** for all three phases can be found in `./code/AVR`.    
The directory `./code/AVR/preprocessing` contains scripts to preprocess annotation data and physiological data, respectively, of any dataset using continuous emotion ratings and videos.   
The scripts `cpa.py` and `cpa_averaged.py` perform a change point analysis for continuous ratings from phase 2 of the AVR dataset, and for other open-source datasets such as the CASE and the CEAP datasets (for the downloadlinks see the [README_code.md](../code/AVR/README_code.md) in the code directory).   
The directory `./code/AVR/datacomparison` contains files that compare continuous ratings from phase 1 of the AVR project with other open-source datasets (at the moment only the CEAP dataset).   
The directories `./code/AVR/datavisualization` and `./code/AVR/modelling` contain nothing so far.   
Additional R-scripts from phase 1 can be found in `./code/Rscripts`.

     📂 code
     ├── 📂 AVR
     │   ├── 📁 datacomparison
     │   ├── 📁 datavisualization
     │   ├── 📁 modelling
     │   └── 📁 preprocessing
     │       ├── 📁 annotation
     │       └── 📁 physiological 
     ├── 📁 configs
     ├── 📁 Rscripts
     │   └── 📁 phase1
     └── 📁 tests
     
The **results** of all analyses can be found in `./results`.    
There is one sub-directory in the main results directory for each dataset being analyzed: `./results/CASE`, `./results/CEAP`, `./results/EmoCompass`, and `./results/phase1` and `./results/phase2` for the AVR data.

     📂 results
     ├── 📁 CASE
     │   ├── 📁 cpa
     │   └── 📁 summary_stats
     ├── 📁 CEAP
     │   └── 📁 descriptives
     ├── 📁 EmoCompass
     ├── 📁 phase1
     │   ├── 📁 assessment_results
     │   ├── 📁 cocor_results
     │   ├── 📁 cor_results
     │   ├── 📁 cr_plots
     │   ├── 📁 datacomparison
     │   ├── 📁 datavisualization
     │   ├── 📁 descriptives
     │   └── 📁 icc_results
     └── 📁 phase2



## Install research code as package

In case there is no project-related virtual / conda environment yet, create one for the project:

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

## To Dos

- [x] write a more meaningful project structure paragraph
- [ ] adapt all other READMEs
- [ ] adapt main README before going public

## Contributors/Collaborators

Antonin Fourcade   
Francesca Malandrone   
Michael Gaebler   

## License
MIT