# Corba

Example code for paper `Contagious Recursive Blocking Attacks on Multi-Agent Systems Based on Large Language Models`.

## Install

Clone this repository and create a new virtual environment.

```shell
git clone https://github.com/zhrli324/Corba.git
cd Corba
conda create -n corba python==3.10 -y
conda activate corba
```

Install necessary dependencies.

```shell
pip install -r requirements.txt
```

For Autogen, additionally install:

```shell
pip install -U "autogen-agentchat"
pip install "autogen-ext[openai]"
pip install autogen-openaiext-client
```

For CAMEL, additionally install:

```shell
pip install 'camel-ai[all]'
```

## Quick Start

For *Open-Source Frameworks Are Vulnerable*, please see `ChatMASs/mas_autogen.py` and `ChatMASs/mas_camel.py`.

For *Effectiveness under Complex Topologies*, please see `ChatMASs/topo.py`.

For *Open-ended LLM-MASs Are Also Susceptible*, please see `open-ended/run.py`.
