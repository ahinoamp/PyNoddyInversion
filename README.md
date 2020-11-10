# PyNoddy_Inversion

![geologic model](/Github.PNG)

PyNoddy_Inversion is a repository for stochastic inversion of geological and geophysical data for constraining structural features in the subsurace.

As is shown in the image below, the code takes in gravity, magnetic, fault markers, stratigraphy markers and tracer connectivity information and tries to find a structural geologic model that matches the data.

![geologic model](/InversionExample.PNG)

For examples of resultant models for the Patua Geothermal Field case study, see the folder Example_Resultant_Models\InteractiveHTML, where you can download an html file of the 3D model and then open it to view in your browser. That folder also includes a power point with images of all of the example models.

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



