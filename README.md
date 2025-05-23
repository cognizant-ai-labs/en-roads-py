# Decision Making for Climate Change

Immediate action is required to combat climate change. The technology behind [Cognizant NeuroAI](https://evolution.ml/) brings automatic decision-making to the [En-ROADS](https://en-roads.climateinteractive.org/) platform, a powerful climate change simulator. A decision-maker can be ready for any scenario: choosing an automatically generated policy that suits their needs best, with the ability to manually modify the policy and see its results. This tool is brought together under [Project Resilience](https://www.itu.int/en/ITU-T/extcoop/ai-data-commons/Pages/project-resilience.aspx), a United Nations initiative to use AI for good.

## En-ROADS Wrapper

En-ROADS is a climate change simulator developed by [Climate Interactive](https://www.climateinteractive.org/). We have created a wrapper around the SDK to make it simple to use in a Python application which can be found in `enroadspy`. See `enroads_runner.py` for the main class that runs the SDK. The SDK is not included in this repository and access to it is very limited. If you would like to run this code, please contact Project Resilience at [daniel.young2@cognizant.com](mailto:daniel.young2@cognizant.com).

NOTE: the "Qualifying path renewables" variable has the same on and off value which may cause some issues.

### Installation
This project was created with Python 3.10.14. Run `pip install -r requirements.txt` to install the required packages.

The En-ROADS SDK must be loaded for this project to run. This can be done with `python -m enroadspy.load_sdk`, after the proper permissions are granted and environment variables are set.

## Evolution

We use 2 methods to evolve the policies: one is our own open-source version of NSGA-II implemented in PyTorch which is in `evolution/`. The other uses the pymoo library and is found in `moo/`.

See the notebooks in `experiments/` for how to analyze the results of such evolution.

## Demo App

A demo app is available in `app/` which displays the results of a pymoo evolution run. Run the app with `python -m app.app`

In order to deploy the app there is a provided Dockerfile. However, first access must be configured to load the SDK from the S3 bucket where it is stored. To build the Docker image use `docker build -t enroads-demo .` Then to run the container use `docker run -p 8080:4057 --name enroads-demo-container enroads-demo`
