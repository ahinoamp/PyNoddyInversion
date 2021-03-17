# PyNoddy_Inversion

![geologic model](/Github.PNG)

PyNoddy_Inversion is a repository for stochastic inversion of geological and geophysical data for constraining structural features in the subsurace.

As is shown in the image below, the code takes in gravity, magnetic, fault markers, stratigraphy markers and tracer connectivity information and tries to find a structural geologic model that matches the data.

![geologic model](/InversionExample.png)

# Run the code!

Check out this Binder jupyter notebook that gives you an example of running the code with all the necessary python packages for you:
https://mybinder.org/v2/gh/ahinoamp/PyNoddyInversion/master?filepath=code%2Frun_MCMC_thread.ipynb

All you need to do is run the cells and observe the results (and then run again - this is a stochastic workflow - so each time will be different). 

# Example Models

For examples of resultant models for the Patua Geothermal Field case study, see the folder Example_Resultant_Models\InteractiveHTML, where you can download an html file of the 3D model and then open it to view in your browser. That folder also includes a power point with images of all of the example models. Or click on one of these links to see some examples now:
- https://htmlpreview.github.io/?https://github.com/ahinoamp/PyNoddyInversion/blob/master/Example_Resultant_Models/InteractiveHTML/GeoModel_503.html
- https://htmlpreview.github.io/?https://github.com/ahinoamp/PyNoddyInversion/blob/master/Example_Resultant_Models/InteractiveHTML/GeoModel_2151.html
- https://htmlpreview.github.io/?https://github.com/ahinoamp/PyNoddyInversion/blob/master/Example_Resultant_Models/InteractiveHTML/GeoModel_4375.html
- https://htmlpreview.github.io/?https://github.com/ahinoamp/PyNoddyInversion/blob/master/Example_Resultant_Models/InteractiveHTML/GeoModel_6039.html
- https://htmlpreview.github.io/?https://github.com/ahinoamp/PyNoddyInversion/blob/master/Example_Resultant_Models/InteractiveHTML/GeoModel_1553.html

# GIS
If you'd like to download the GIS data associated with this project, you can do so here:
https://drive.google.com/file/d/197MIDjw85VurKN3FLpsL7mFN7DxwWKSx/view?usp=sharing

# Contact
If you have any questions or comments, I'd be happy to talk (ahinoamp followed by the domain gmail). 

# Code structure
This repository is seperated into
1. core code
2. data files
3. examples
4. scratch

Inside the core folder, the main files are: 

1. SimulationUtilties.py: A file containing the functions for calling the kinematic structural geology simulator Noddy and analyzing
the results in order to simulate gravity, magentic, granite top, tracer connectivity, and fault markers

Three optimisation/search algorithm files:

2.  GA_Noddy.py: the genetic algorithm workflow

3.  MCMC_Noddy.py: the Markov Chain Monte Carlo workflow as well as the simulated annealing workflow

4.  NSGA_Noddy.py: the Non-Dominated Sorting Genetic Algorithm II

Several files with general utility functions for the inversion:

5. GeneralInversionUtil.py: general utilties used in all the algorithms

6. SamplingHisFileUtil: utilities for sampling parameters and creating history files

7. HisTemplates.py: a class assisting in creating history files 

8. PriorUnvertaintyUtil.py: a utilty for defining the prior uncertainty of the different structural geological events

9. VizualizationUtilities.py: a utility for creating visualizations of the inversion process

Utilties for the specific inversion workflow

10. MCMC_Util.py: utilties specific to the MCMC workflow

11. PSO_GA.py: utilities specific to the GA and NSGA algorithms

Utilties for running many inversion chains at once with different hyper parameters:

12. ThreadMasterCombo.py



