"""
Class Activity 3: Build a Mini Autonomous Travel Agent
=========================================================

Goal: Build a simple AI agent that can understand a travel request, create a
plan, use tools, observe results, remember preferences, reason about options,
and take an authorized action.

This file implements the full activity breakdown from the slide:

    1. Design the Agent Architecture
    2. Implement the Agent Lifecycle
    3. Add Memory
    4. Give the Agent Tools
    5. Introduce Autonomous Behavior
    6. Add Human-in-the-Loop

...and wires in all 10 core concepts the slide asks us to identify:

    1. Goal          -> the natural-language travel request
    2. LLM            -> `LLMReasoner` (a swappable "brain"; see note below)
    3. Memory         -> `AgentMemory`
    4. Planning       -> `Planner.make_plan()`
    5. Reasoning      -> `LLMReasoner.decide_next_step()`
    6. Tools          -> `TOOLBOX` (flights, hotels, weather, booking)
    7. Actions        -> `Agent._act()`
    8. Observations    -> `Agent._observe()` / the `Observation` dataclass
    9. Re-planning     -> `Planner.replan()`
    10. Human approval -> `HumanInTheLoop.request_approval()`

NOTE ON THE "LLM" PIECE
------------------------
A real deployment would replace `LLMReasoner` with an actual call to the
Anthropic API, e.g.:

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=AGENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": state_summary}],
    )

Because this sandbox has network access disabled, `LLMReasoner` here is a
small rule-based stand-in that mimics the *shape* of what an LLM call would
return (a decision: which tool to call next, with what arguments, or that
the plan is complete) so the rest of the agent loop -- planning, memory,
tools, observation, re-planning, human approval -- is fully real and
testable end-to-end. Swap `LLMReasoner.decide_next_step` for a real API
call and nothing else in the architecture needs to change.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# 1. DESIGN THE AGENT ARCHITECTURE
# ---------------------------------------------------------------------------
# The architecture is split into clean, single-responsibility pieces:
#
#   Goal (str) -> Planner -> [Step, Step, ...]
#                     ^              |
#                     |              v
#              Re-planning <-- Reasoner --> Tools --> Observation
#                                    |
#                                    v
#                         HumanInTheLoop (for sensitive actions)
#
# AgentMemory persists across the whole lifecycle and is readable/writable
# by every other component.


@dataclass
class Observation:
    """9. Observations — the result of running a tool."""
    step: str
    tool: str
    success: bool
    data: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def summary(self) -> str:
        status = "OK" if self.success else "FAILED"
        return f"[{status}] {self.tool} -> {self.data}"


class AgentMemory:
    """3. Add Memory — short-term (this trip) + long-term (preferences)."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        self.preferences: dict[str, Any] = {}
        self.history: list[Observation] = []
        self.scratchpad: dict[str, Any] = {}  # working memory for the current plan
        if self.path and self.path.exists():
            self._load()

    # --- long-term preferences ---
    def remember_preference(self, key: str, value: Any):
        self.preferences[key] = value
        self._save()

    def get_preference(self, key: str, default=None):
        return self.preferences.get(key, default)

    # --- short-term episodic memory ---
    def log_observation(self, obs: Observation):
        self.history.append(obs)

    def recent_observations(self, n: int = 5) -> list[Observation]:
        return self.history[-n:]

    # --- persistence (so preferences survive across runs, like a returning user) ---
    def _save(self):
        if not self.path:
            return
        self.path.write_text(json.dumps({"preferences": self.preferences}, indent=2))

    def _load(self):
        try:
            data = json.loads(self.path.read_text())
            self.preferences = data.get("preferences", {})
        except (json.JSONDecodeError, FileNotFoundError):
            self.preferences = {}


# ---------------------------------------------------------------------------
# 4. GIVE THE AGENT TOOLS
# ---------------------------------------------------------------------------
# Each tool is a plain function: (args, memory) -> result.
# Mock data is used by default. Swapping in a real API (e.g. a live flights
# or weather API) only requires replacing the body of one function — the
# rest of the agent is unaffected. Real-time fetching is optional, per the
# activity instructions, so it's stubbed but clearly marked.

_MOCK_FLIGHTS = [
    {"airline": "IndiGo", "flight": "6E 204", "price": 4200, "depart": "06:10", "arrive": "08:35", "stops": 0},
    {"airline": "Air India", "flight": "AI 505", "price": 5100, "depart": "09:00", "arrive": "11:20", "stops": 0},
    {"airline": "Vistara", "flight": "UK 815", "price": 3800, "depart": "14:45", "arrive": "19:10", "stops": 1},
    {"airline": "SpiceJet", "flight": "SG 112", "price": 3500, "depart": "22:30", "arrive": "00:55", "stops": 0},
]

_MOCK_HOTELS = [
    {"name": "Lakeview Residency", "price_per_night": 2200, "rating": 4.3, "area": "City Center"},
    {"name": "Budget Inn Express", "price_per_night": 1200, "rating": 3.6, "area": "Airport Road"},
    {"name": "The Grand Palm", "price_per_night": 6500, "rating": 4.8, "area": "Beachfront"},
    {"name": "Cozy Nest Homestay", "price_per_night": 1800, "rating": 4.5, "area": "Old Town"},
]


def tool_search_flights(args: dict, memory: AgentMemory) -> dict:
    """Mock flight search, filtered by the user's remembered budget if any."""
    budget = memory.get_preference("max_flight_price")
    candidates = [f for f in _MOCK_FLIGHTS if not budget or f["price"] <= budget]
    candidates.sort(key=lambda f: f["price"])
    return {"origin": args.get("origin"), "destination": args.get("destination"), "options": candidates}


def tool_search_hotels(args: dict, memory: AgentMemory) -> dict:
    """Mock hotel search, filtered by remembered preferred area if any."""
    preferred_area = memory.get_preference("preferred_area")
    candidates = _MOCK_HOTELS
    if preferred_area:
        filtered = [h for h in candidates if preferred_area.lower() in h["area"].lower()]
        candidates = filtered or candidates
    candidates.sort(key=lambda h: -h["rating"])
    return {"destination": args.get("destination"), "options": candidates}


def tool_get_weather(args: dict, memory: AgentMemory) -> dict:
    """
    Optional real-time tool. Network access is disabled in this sandbox, so
    this returns simulated seasonal data. To make it real, replace the body
    with e.g. a call to a weather API (no key required: Open-Meteo) using
    the destination's coordinates.
    """
    destination = args.get("destination", "your destination")
    # Simple deterministic "simulation" so results are reproducible for testing.
    seed = sum(ord(c) for c in destination) % 5
    forecast = ["Sunny, 29°C", "Light rain, 24°C", "Partly cloudy, 27°C", "Humid, 31°C", "Clear skies, 26°C"][seed]
    return {"destination": destination, "forecast": forecast, "source": "SIMULATED (real-time fetch optional/disabled)"}


def tool_book_trip(args: dict, memory: AgentMemory) -> dict:
    """
    The sensitive/irreversible tool — spends money. This is exactly why the
    activity calls for Human-in-the-Loop before this tool is ever executed.
    """
    return {
        "status": "CONFIRMED",
        "flight": args.get("flight"),
        "hotel": args.get("hotel"),
        "total_cost": args.get("total_cost"),
        "confirmation_id": f"TRIP-{abs(hash(json.dumps(args, sort_keys=True))) % 100000}",
    }


TOOLBOX: dict[str, Callable[[dict, AgentMemory], dict]] = {
    "search_flights": tool_search_flights,
    "search_hotels": tool_search_hotels,
    "get_weather": tool_get_weather,
    "book_trip": tool_book_trip,
}


# ---------------------------------------------------------------------------
# PLANNING  (component 4/10)  +  RE-PLANNING (component 9/10)
# ---------------------------------------------------------------------------

class Planner:
    """5. (part of) Introduce Autonomous Behavior — turns a goal into steps."""

    def make_plan(self, goal_info: dict) -> list[str]:
        plan = ["search_flights", "search_hotels"]
        if goal_info.get("check_weather", True):
            plan.append("get_weather")
        plan.append("propose_itinerary")  # internal reasoning step, not a tool
        plan.append("book_trip")
        return plan

    def replan(self, plan: list[str], current_step: str, observation: Observation) -> list[str]:
        """9. Re-planning — react to a failed or unsatisfying observation."""
        if not observation.success:
            # e.g. no flights under budget -> insert a relaxed retry before giving up
            if current_step == "search_flights" and "retry_search_flights" not in plan:
                idx = plan.index(current_step)
                plan.insert(idx + 1, "retry_search_flights")
        return plan


# ---------------------------------------------------------------------------
# REASONING (component 5/10) — the "LLM" stand-in
# ---------------------------------------------------------------------------

class LLMReasoner:
    """
    6. Reasoning — decides, at each step, which tool to call and with what
    arguments, by looking at the goal, memory, and observations so far.
    This is where a real LLM call would live (see module docstring).
    """

    def decide_next_step(self, step: str, goal_info: dict, memory: AgentMemory,
                          observations: list[Observation]) -> dict:
        if step in ("search_flights", "retry_search_flights"):
            return {"tool": "search_flights",
                    "args": {"origin": goal_info["origin"], "destination": goal_info["destination"]}}

        if step == "search_hotels":
            return {"tool": "search_hotels", "args": {"destination": goal_info["destination"]}}

        if step == "get_weather":
            return {"tool": "get_weather", "args": {"destination": goal_info["destination"]}}

        if step == "propose_itinerary":
            # Reason over prior observations to pick the best flight + hotel.
            flight_obs = next((o for o in observations if o.tool == "search_flights"), None)
            hotel_obs = next((o for o in observations if o.tool == "search_hotels"), None)
            best_flight = flight_obs.data["options"][0] if flight_obs and flight_obs.data["options"] else None
            best_hotel = hotel_obs.data["options"][0] if hotel_obs and hotel_obs.data["options"] else None
            return {"tool": None, "args": {}, "reasoning_output": {"best_flight": best_flight, "best_hotel": best_hotel}}

        if step == "book_trip":
            itinerary = memory.scratchpad.get("itinerary", {})
            flight = itinerary.get("best_flight")
            hotel = itinerary.get("best_hotel")
            nights = goal_info.get("nights", 1)
            total_cost = (flight["price"] if flight else 0) + (hotel["price_per_night"] * nights if hotel else 0)
            return {"tool": "book_trip",
                    "args": {"flight": flight, "hotel": hotel, "total_cost": total_cost}}

        return {"tool": None, "args": {}}


# ---------------------------------------------------------------------------
# 6. ADD HUMAN-IN-THE-LOOP  (component 10/10)
# ---------------------------------------------------------------------------

class HumanInTheLoop:
    """
    Gate for sensitive/irreversible actions. In a real product this would
    pause and wait for a UI click / chat reply. Here, an `approve_fn`
    callback is injected so the same agent can be driven either by a human
    or, for automated testing, by a scripted "virtual human".
    """

    def __init__(self, approve_fn: Callable[[str, dict], bool]):
        self.approve_fn = approve_fn

    def request_approval(self, action: str, details: dict) -> bool:
        return self.approve_fn(action, details)


SENSITIVE_TOOLS = {"book_trip"}  # any tool that spends money or is irreversible


# ---------------------------------------------------------------------------
# 2. IMPLEMENT THE AGENT LIFECYCLE  +  5. AUTONOMOUS BEHAVIOR
# ---------------------------------------------------------------------------

class TravelAgent:
    """
    The full lifecycle, run autonomously step-by-step:

        perceive (parse goal) -> plan -> reason -> act -> observe
              -> replan if needed -> ... -> human approval -> act -> done

    The agent runs the whole loop itself without needing a human to
    micromanage each tool call — that's the "autonomous" part. The human
    is only pulled in at the one point that matters: before booking.
    """

    def __init__(self, memory: AgentMemory, human: HumanInTheLoop, verbose: bool = True):
        self.memory = memory
        self.human = human
        self.planner = Planner()
        self.reasoner = LLMReasoner()
        self.verbose = verbose
        self.log: list[str] = []

    def _log(self, msg: str):
        self.log.append(msg)
        if self.verbose:
            print(msg)

    # 1. Goal — parse a natural-language request into structured goal_info
    def _perceive_goal(self, request: str) -> dict:
        origin_match = re.search(r"from ([A-Za-z ]+?) to", request, re.I)
        dest_match = re.search(r"to ([A-Za-z ]+?)(?: for| on|,|\.|$)", request, re.I)
        nights_match = re.search(r"(\d+)\s*night", request, re.I)
        budget_match = re.search(r"under\s*(?:₹|rs\.?|inr)?\s*([\d,]+)", request, re.I)

        goal_info = {
            "raw_request": request,
            "origin": (origin_match.group(1).strip() if origin_match else self.memory.get_preference("home_city", "Bengaluru")),
            "destination": dest_match.group(1).strip() if dest_match else "Goa",
            "nights": int(nights_match.group(1)) if nights_match else 3,
            "check_weather": True,
        }
        if budget_match:
            budget = int(budget_match.group(1).replace(",", ""))
            self.memory.remember_preference("max_flight_price", budget)
            goal_info["budget"] = budget
        return goal_info

    def _act(self, tool_name: str, args: dict) -> Observation:
        """7. Actions — execute a tool call."""
        try:
            result = TOOLBOX[tool_name](args, self.memory)
            return Observation(step=tool_name, tool=tool_name, success=True, data=result)
        except Exception as e:  # noqa: BLE001
            return Observation(step=tool_name, tool=tool_name, success=False, data=str(e))

    def _observe(self, obs: Observation):
        """8. Observations — record + reflect."""
        self.memory.log_observation(obs)
        self._log(f"  observed: {obs.summary()}")

    def run(self, request: str) -> dict:
        self._log(f"\n=== GOAL === {request}")
        goal_info = self._perceive_goal(request)
        self._log(f"  parsed goal: {goal_info}")

        plan = self.planner.make_plan(goal_info)
        self._log(f"  initial plan: {plan}")

        i = 0
        while i < len(plan):
            step = plan[i]
            self._log(f"\n-- step: {step}")

            decision = self.reasoner.decide_next_step(step, goal_info, self.memory, self.memory.history)

            # Non-tool reasoning steps (e.g. proposing the itinerary)
            if decision["tool"] is None:
                if "reasoning_output" in decision:
                    self.memory.scratchpad["itinerary"] = decision["reasoning_output"]
                    self._log(f"  reasoned itinerary: {decision['reasoning_output']}")
                i += 1
                continue

            tool_name = decision["tool"]
            args = decision["args"]

            # 6. Human-in-the-loop for sensitive tools
            if tool_name in SENSITIVE_TOOLS:
                approved = self.human.request_approval(tool_name, args)
                self._log(f"  human approval for '{tool_name}': {'APPROVED' if approved else 'DENIED'}")
                if not approved:
                    self._log("  agent halts: cannot take a sensitive action without approval.")
                    return {"status": "halted_no_approval", "log": self.log}

            obs = self._act(tool_name, args)
            self._observe(obs)

            # 9. Re-planning based on what was observed
            new_plan = self.planner.replan(plan, step, obs)
            if new_plan != plan:
                self._log(f"  re-planned: {new_plan}")
                plan = new_plan

            i += 1

        final_result = self.memory.history[-1].data if self.memory.history else None
        self._log(f"\n=== DONE === {final_result}")
        return {"status": "completed", "result": final_result, "log": self.log}
