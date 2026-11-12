Collective Behaviour - Sheep Herding Simulation

Team Members

- Mia Miletić - [dolouress]
- Anja Pijak - [anjapijak]
- Kristina Đukić - [kristina-djukic]
- Ana Nikolić - [Ana63250469]

Starting Point - Paper: Jadhav, V., et al. (2024). Collective responses of flocking sheep (Ovis aries) to a herding dog (border collie) - https://doi.org/10.1038/s42003-024-07245-8

Project Description:
Our project will focus on reproducing and extending the agent-based sheep-herding model presented by Jadhav et al. (2024). The original model investigates the collective response of sheep to a herding dog, including interactions between individuals, flock cohesion, and directional coordination. We plan to recreate the simulation, analyze its behaviour, and compare selected results with those reported in the paper. Based on a review of related research, we will then introduce modifications to the original model to investigate how changes in individual behaviour or interaction rules influence the collective dynamics of the flock. Finally, we will evaluate the results through simulations and quantitative measurements. 

For our extension, we decided to focus on differences in how individual sheep respond to the herding dog. We plan to use fuzzy logic to make their responses depend on the distance from the dog and to introduce different sensitivity levels between sheep. We will then compare the original model with the modified versions to see how these changes affect the flock's collective behaviour.

Project Plan - Milestones:

Milestone 1 - Research and initial implementation
- Review the original paper and relevant literature on collective behaviour.
- Recreate the original MATLAB model in Python and run baseline simulations.
- Analyse polarisation, cohesion, and elongation, and compare Python and MATLAB results using dog-to-rearmost-sheep distance, lateral offset, and three directional angles.
- Define the fuzzy-logic extension and prepare the first report.

Milestone 2 - Model verification and extension
- Validate the Python reproduction by comparing the distributions of all eight metrics with the MATLAB model.
- Implement fuzzy dog-avoidance responses with different sensitivity levels for individual sheep.
- Test the original, homogeneous fuzzy, and heterogeneous fuzzy models, and define the evaluation methodology using repeated simulations.
- Prepare the second report with implementation and verification details.

Milestone 3 - Experiments and conclusions
- Run experiments comparing the original, homogeneous fuzzy, and heterogeneous fuzzy models using the eight metrics.
- Investigate how individual sensitivity affects directional leadership and collective flock movement.
- Analyse the results, compare them with the original research, and discuss limitations.
- Prepare the final report, presentation, narrated simulation video, and GitHub documentation.

Technology: Python, NumPy, MATLAB (reference implementation).
