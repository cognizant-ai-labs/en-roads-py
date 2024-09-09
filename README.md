# Decision Making for Climate Change

Immediate action is required to combat climate change. The technology behind [Cognizant NeuroAI](https://evolution.ml/) brings automatic decision-making to the En-ROADS platform, a powerful climate change simulator. A decision-maker can be ready for any scenario: choosing an automatically generated policy that suits their needs best, with the ability to manually modify the policy and see its results. This tool is brought together under Project Resilience, a United Nations initiative to use AI for good.

## En-ROADS Wrapper

En-ROADS is a climate change simulator developed by Climate Interactive. We have created a wrapper around the SDK to make it simple to use in a Python application which can be found in `enroadspy`. See `enroads_runner.py` for the main class that runs the SDK. The SDK is not included in this repository and must be requested from [Climate Interactive](https://www.climateinteractive.org/).

The input data format is a crazy long JSON object which I copied out of the source code, pasted into `inputSpecs.py`, and parsed into `inputSpecs.jsonl`. This format is used by the rest of the repository.

### Installation
Run `pip install -r requirements.txt` to install the required packages. Then run `python -m enroadspy.download_sdk` to download the SDK. In order to download the SDK environment variables must be set which can be requested online from Climate Interactive.

## Evolution

We use 2 methods to evolve the policies: one is our own open-source version of NSGA-II implemented in PyTorch which is in `evolution/`. The other uses the pymoo library and is found in `moo/`.

See the notebooks in `experiments/` for how to analyze the results of such evolution.

## Demo App

A demo app is available in `app/` which displays the results of a pymoo evolution run. Run the app with `python -m app.app`

In order to deploy the app there is a provided Dockerfile. However, first environment variables must be set in order to download the SDK, which can be requested online. To build the Docker image use `docker build -t enroads-demo --build-arg ENROADS_URL=$ENROADS_URL --build-arg ENROADS_ID=$ENROADS_ID --build-arg ENROADS_PASSWORD=$ENROADS_PASSWORD .` Then to run the container use `docker run -p 8080:4057 --name enroads-demo-container enroads-demo`
