# On-device AI Voice Assistant


Clone the repository using the following command:

```bash
git clone https://github.com/synaptics-astra-demos-stg/japanese-assistant.git
```
Navigate to the Repository Directory:

```
cd japanese-assistant
```

## Download Japanese models
Download All the Japanese models from [here](https://synaptics.sharepoint.com.mcas.ms/sites/ProcessorMarketing/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FProcessorMarketing%2FShared%20Documents%2FSolutions%20Team%2FACL%20Project%201%20%28Vision%20%2B%20SLM%29%2FSLM%2DAgent%5FProj%2D1%2Fjp%2Dmodels%2Fjp%2Dmodels&viewid=a157d8f9%2Dca0a%2D49d0%2D982e%2D3666d751f54d&csf=1&web=1&e=bALQq3&CID=835f8e13%2D5112%2D4e21%2D98cd%2D89b3aa2d2df9&FolderCTID=0x012000C9E60BBA7188CE4C9E684FED50FDAAF5)

So , just download the `jp-models` , it will download a zip in your Host machine. ADB or copy the `jp-models` into Machina board inside `jp-assistant` folder.

This `jp-models` folder contains 3 folders:
1. washingBERT
2. moonshine_jp
3. fumi_f_ja

## Add Anthropic API Key

Add `.env` file which has Anthropic Claude API key  in `jp-assistant` folder

## Setup
Run `./install.sh` to setup environment and download models.



<!-- ## Download Tsuki TTS model
```
wget https://storage.googleapis.com/useful-sensors-public/tsuki/fumi-f-ja-51M.tar -P models/
cd models
tar -xvf fumi-f-ja-51M.tar
cd ..
``` -->


## Demo
Launch assistant with:
```sh
source .venv/bin/activate
python assistant.py
```

#### Run Options
* `--qa-file`: Path to Question-Answer pairs (default: [data/qa_350_2.json](data/qa_350_2.json)
* `--cpu-only`: Use CPU only models
* `-j`: Number of cores to use for CPU execution (default: all)

Run `python assistant.py --help` to view all available options

## Profiling
Profiling scripts are available at [profile/](profile/). Currently supported models:

| Model | Script | Supported Model Types |
| ----- | ------ | ----------------- |
| MiniLM | [profile/minilm.py](profile/minilm.py) | `.synap` (SyNAP), `.gguf` (llama.cpp) |
| Moonshine | [profile/moonshine.py](profile/moonshine.py) | `.synap` (SyNAP), `.onnx` (ORT) |

Run profiling with:
```sh
source .venv/bin/activate
python -m profile.<model>
```

#### Run Options
* `--model[s]`: Model(s) to profile, inference runner is selected based on model type
* `--run-forever`: Continuosly profile provided models in a loop until interrupted with `ctrl + c`
* `-j`: Number of cores to use for CPU execution (default: all)

> [!TIP]
> Use in conjunction with the [Astra resource usage visualizer](https://github.com/spal-synaptics/astra-visualizer) to get a live dashboard of CPU and NPU usage during inference
