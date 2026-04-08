import os
from typing import List, Optional
from openai import OpenAI
from data import messages
from env import MessageEnv
from env import Action

# ENV VARIABLES 

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")

# CONFIG
TASK_NAME = "message-prioritization"
BENCHMARK = "custom_env"
SUCCESS_THRESHOLD = 0.5

# OPENAI CLIENT
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# LOGGING FUNCTIONS 
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# LLM AGENT
def llm_agent(state):
    prompt = f"""
You are an expert message prioritization agent.

IMPORTANT messages:
- deadlines
- meetings
- urgent tasks
- submissions
- official updates

NOT IMPORTANT:
- casual chat
- greetings
- jokes

Message features:
{state.message}

Return ONLY:
1 or 0
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a classifier."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=5,
        )

        output = (response.choices[0].message.content or "").strip()

        if "1" in output:
            return 1
        return 0

    except Exception:
        return 0

# TASK RUNNER
def run_task(task_name, task_messages):
    env = MessageEnv(task_messages)

    rewards = []
    steps_taken = 0

    log_start(task_name, BENCHMARK, MODEL_NAME)

    try:
        state = env.reset()

        for step in range(1, len(task_messages) + 1):
           
            action_value = llm_agent(state)
            action = Action(label=action_value)

            next_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            steps_taken = step

           
            log_step(step, str(action_value), reward, done, None)

            state = next_state

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)

        success = score >= SUCCESS_THRESHOLD

    finally:
        log_end(success, steps_taken, score, rewards)

# MAIN (3 TASKS)
def main():
    # Split dataset into 3 tasks
    n = len(messages)
    easy = messages[: n // 3]
    medium = messages[n // 3 : 2 * n // 3]
    hard = messages[2 * n // 3 :]

    run_task("easy", easy)
    run_task("medium", medium)
    run_task("hard", hard)


if __name__ == "__main__":
    main()