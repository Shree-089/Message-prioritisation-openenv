# Message Prioritization OpenEnv

## Overview

This project implements a **real-world reinforcement learning (RL)-style environment** for message prioritization, where an AI agent classifies messages as **important** or **not important**.

The system follows an **OpenEnv-inspired architecture** with structured observation, action, and reward modeling using **Pydantic**, along with a reproducible inference pipeline powered by an LLM.


## Problem Statement

In real-world communication platforms (e.g., WhatsApp, Slack), important messages often get lost among casual conversations.

This project simulates an AI agent that:

* Reads incoming messages
* Determines their importance
* Helps users focus on critical information


## Environment Design

### Observation Space

```
{
  "message": "string"
}
```


### Action Space

| Action | Meaning       |
| ------ | ------------- |
| 0      | Not Important |
| 1      | Important     |


### Reward Function

| Outcome                  | Reward |
| ------------------------ | ------ |
| Correct classification   | +1     |
| Incorrect classification | -1     |

✔ Provides continuous feedback
✔ Encourages correct decision-making


## Environment Interface

The environment follows an OpenEnv-style structure:

* reset() → returns initial observation
* step(action) → returns (observation, reward, done, info)
* state() → returns current observation


## Tasks

The dataset is divided into three tasks:

| Task   | Description                                             |
| ------ | ------------------------------------------------------- |
| Easy   | Clear distinction between important and casual messages |
| Medium | Mixed and ambiguous cases                               |
| Hard   | Subtle or low-signal messages                           |



## Agent

The agent uses an **OpenAI-compatible LLM API** to classify messages.

It is prompted with:

* Definitions of important vs non-important messages
* The current observation


## Setup & Execution

### 1. Install dependencies

pip install -r requirements.txt


### 2. Set environment variables

set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
set OPENAI_API_KEY=your_hf_token


### 3. Run the environment

python inference.py


## Output Format

The script logs execution in structured format:

[START]
[STEP]
[END]

This ensures reproducibility and compatibility with evaluation pipelines.

---

## Key Features

* Real-world task simulation
* Pydantic-based structured models
* OpenEnv-style environment interface
* Multi-task evaluation (easy / medium / hard)
* LLM-based decision agent
* Deterministic reward-based scoring


## Future Improvements

* Full OpenEnv validation support
* Advanced reward shaping
* Better task difficulty calibration
* UI for visualization


## Baseline Performance

The following results were obtained using the provided LLM-based agent:

| Task   | Score |
| ------ | ----- |
| Easy   | 0.90  |
| Medium | 0.00  |
| Hard   | 1.00  |

### Interpretation

* **Easy**: High accuracy due to clear signals (deadlines, meetings, urgent keywords)
* **Medium**: Mixed performance due to ambiguous and less explicit messages
* **Hard**: High accuracy due to dominance of non-important messages

These results demonstrate that:

* The reward function is meaningful
* The environment produces varied outcomes
* The agent behavior changes across difficulty levels

---

## Task Difficulty Design

* **Easy** → Messages contain clear indicators of importance (e.g., "deadline", "meeting")
* **Medium** → Messages include mixed signals and ambiguity
* **Hard** → Messages are mostly casual, requiring subtle distinction

This progression ensures increasing complexity for the agent across tasks.



