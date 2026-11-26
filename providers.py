"""Asks one choice question to not-bev, Jev (TypeSafe, via AIML API) and OpenAI's Decisions API. Keys are read from .env."""
import time
from functools import cache
from pathlib import Path

import requests
from dotenv import dotenv_values
from openai import OpenAI

from inference import answer

ENV = Path(__file__).with_name(".env")
jev_session = requests.Session()


def key(name):
    value = dotenv_values(ENV).get(name)
    if not value:
        raise RuntimeError(f"no {name} in .env")
    return value


@cache
def openai_client(api_key):
    return OpenAI(api_key=api_key)


def ask_nbev(network, config, device, state, instructions, criteria):
    question = {"type": "choice", "instructions": instructions, "criteria": criteria}
    return answer(network, config, state, question, device)["probabilities"]


def ask_jev(state, instructions, criteria):
    response = jev_session.post("https://api.aimlapi.com/v1/decisions", timeout=60,
                                headers={"Authorization": f"Bearer {key('AIMLAPI_KEY')}"},
                                json={"model": "typesafe/jev", "state": state, "questions": {
                                    "q": {"type": "choice", "instructions": instructions, "criteria": criteria}}})
    response.raise_for_status()
    return response.json()["answers"]["q"]["probabilities"]


def ask_openai(state, instructions, criteria):
    choices = [{"value": k, "description": v} if v else {"value": k} for k, v in criteria.items()]
    decision = openai_client(key("OPENAI_API_KEY")).decisions.create(
        model="gpt-6-luna", input=state, questions=[{"type": "choice", "instructions": instructions, "choices": choices}])
    return {p.value: p.probability for p in decision.answers[0].probabilities}


def from_dataset(row):
    criteria = dict(c.split(": ", 1) if ": " in c else (c, "") for c in row["choices"])
    return {"state": row["state"], "instructions": row["instructions"], "criteria": criteria}


def ask_all(network, config, device, state, instructions, criteria):
    models = {"not-bev": lambda: ask_nbev(network, config, device, state, instructions, criteria),
              "Jev": lambda: ask_jev(state, instructions, criteria),
              "OpenAI": lambda: ask_openai(state, instructions, criteria)}
    for name, ask in models.items():
        try:
            ask()  # warm-up
            start = time.perf_counter()
            probabilities = ask()
            ms = 1000 * (time.perf_counter() - start)
        except Exception as error:
            print(f"{name}: {error}\n")
            continue
        print(f"{name}: {max(probabilities, key=probabilities.get)}  ({ms:.0f} ms)")
        for option in criteria:
            p = probabilities.get(option, 0.0)
            print(f"   {p:.3f}  {'█' * round(p * 30):<30}  {option}")
        print()
