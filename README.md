# AffectiveVR

`[Last update: December 12, 2023]`

    Period:     2023-10 - 2024-09
    Status:     work in progress

    Author(s):  Lucy Roellecke
    Contact:    lucy.roellecke@fu-berlin.de


## Project description

*Affective VR* (AVR) aims to develop and test a tool for continous emotion ratings. The project proposes such a tool and assesses its effectiveness, usability and reliability using videos presented in virtual reality (VR).

## Project structure


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
     
     📂 data
     ├── 📁 CASE
     │   ├── 📁 initial
     │   ├── 📁 interpolated
     │   ├── 📁 metadata
     │   ├── 📁 non-interpolated
     │   ├── 📁 preprocessed
     │   └── 📁 raw
     ├── 📁 CEAP
     │   └── 📁 CEAP-360VR
     ├── 📁 EmoCompass
     │   ├── 📁 data_codebank
     │   ├── 📁 data_raw
     │   ├── 📁 emotion_compass_material
     │   ├── 📁 experiment_scripts
     │   ├── 📁 source_data
     │   ├── 📁 stimuli
     │   ├── 📁 Supplementary_Software_1
     │   └── 📁 Wiki images
     ├── 📁 phase1
     │   ├── 📁 AVR
     │   └── 📁 preprocessed
     └── 📁 phase2
         ├── 📁 AVR
         └── 📁 preprocessed

     📂 literature
     └──📁 pdfs

     📂 organisation
     ├── 📁 color_schemes
     ├── 📁 experiment_preparation
     ├── 📁 participation_forms
     └── 📁 project_proposal
     
     📂 publications
     ├── 📁 articles
     ├── 📁 poster
     └── 📁 presentations
     
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

- [ ] write a more meaningful project structure paragraph
- [ ] adapt all other READMEs
- [ ] adapt main README before going public

## Contributors/Collaborators

Antonin Fourcade   
Francesca Malandrone   
Michael Gaebler   

## License
MIT