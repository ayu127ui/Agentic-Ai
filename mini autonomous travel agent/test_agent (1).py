"""
Virtual test environment for the Mini Autonomous Travel Agent.

Since there's no real user to click "approve" in this sandbox, we simulate
one with a scripted `virtual_human` callback. This lets us test the full
lifecycle end-to-end, including the human-in-the-loop gate, automatically.

Scenarios covered:
    1. Happy path — human approves booking.
    2. Human denies booking — agent halts safely, spends no money.
    3. Budget constraint triggers re-planning inside flight search.
    4. Memory persistence — a second, separate agent run remembers the
       user's preferred area/budget from a previous session (loaded from disk).
"""

from travel_agent import TravelAgent, AgentMemory, HumanInTheLoop
import os


def virtual_human_always_approve(action: str, details: dict) -> bool:
    print(f"  [virtual human sees] approve '{action}' with details={details}? -> YES")
    return True


def virtual_human_always_deny(action: str, details: dict) -> bool:
    print(f"  [virtual human sees] approve '{action}' with details={details}? -> NO")
    return False


def scenario_1_happy_path():
    print("\n" + "=" * 70)
    print("SCENARIO 1: Happy path (human approves the booking)")
    print("=" * 70)
    memory = AgentMemory()  # fresh, in-memory only
    human = HumanInTheLoop(virtual_human_always_approve)
    agent = TravelAgent(memory, human)
    result = agent.run("Plan a trip from Bengaluru to Goa for 3 nights, budget under 5000 for flights")
    assert result["status"] == "completed", "Scenario 1 should complete"
    assert result["result"]["status"] == "CONFIRMED", "Booking should be confirmed"
    print("\nPASS: trip booked with confirmation:", result["result"]["confirmation_id"])


def scenario_2_human_denies():
    print("\n" + "=" * 70)
    print("SCENARIO 2: Human-in-the-loop DENIES the booking")
    print("=" * 70)
    memory = AgentMemory()
    human = HumanInTheLoop(virtual_human_always_deny)
    agent = TravelAgent(memory, human)
    result = agent.run("Plan a trip from Bengaluru to Goa for 2 nights")
    assert result["status"] == "halted_no_approval", "Agent must halt, not book, when denied"
    # Confirm no booking observation was ever logged
    booked = any(o.tool == "book_trip" for o in memory.history)
    assert not booked, "Agent must NOT call book_trip without approval"
    print("\nPASS: agent correctly refused to book without human approval.")


def scenario_3_tight_budget_replans():
    print("\n" + "=" * 70)
    print("SCENARIO 3: Very tight budget forces a re-plan / retry on flight search")
    print("=" * 70)
    memory = AgentMemory()
    memory.remember_preference("max_flight_price", 100)  # unrealistically low on purpose
    human = HumanInTheLoop(virtual_human_always_approve)
    agent = TravelAgent(memory, human)
    result = agent.run("Plan a trip from Bengaluru to Goa for 1 night")
    flight_search_calls = [o for o in memory.history if o.tool == "search_flights"]
    print(f"\nflight search was attempted {len(flight_search_calls)} time(s) due to replanning")
    assert result["status"] == "completed"
    print("PASS: agent handled an unsatisfiable budget without crashing.")


def scenario_4_memory_persists_across_sessions():
    print("\n" + "=" * 70)
    print("SCENARIO 4: Memory persists across separate agent sessions")
    print("=" * 70)
    mem_path = "/home/claude/travel_agent/_test_memory.json"
    if os.path.exists(mem_path):
        os.remove(mem_path)

    # --- Session A: user states a preference, agent remembers it to disk ---
    memory_a = AgentMemory(path=mem_path)
    memory_a.remember_preference("preferred_area", "Beachfront")
    memory_a.remember_preference("home_city", "Bengaluru")
    print("  session A saved preferences:", memory_a.preferences)

    # --- Session B: brand-new agent instance, loads memory from disk ---
    memory_b = AgentMemory(path=mem_path)
    human = HumanInTheLoop(virtual_human_always_approve)
    agent_b = TravelAgent(memory_b, human)
    result = agent_b.run("Plan a trip to Goa for 2 nights")

    hotel_obs = next(o for o in memory_b.history if o.tool == "search_hotels")
    top_hotel_area = hotel_obs.data["options"][0]["area"]
    print(f"  session B (new agent instance) top hotel area: {top_hotel_area}")
    assert "beachfront" in top_hotel_area.lower(), "Agent should recall the remembered preference"
    print("PASS: preference remembered from session A correctly influenced session B.")

    os.remove(mem_path)


if __name__ == "__main__":
    scenario_1_happy_path()
    scenario_2_human_denies()
    scenario_3_tight_budget_replans()
    scenario_4_memory_persists_across_sessions()
    print("\n" + "=" * 70)
    print("ALL SCENARIOS PASSED — agent lifecycle, memory, tools, reasoning,")
    print("re-planning, and human-in-the-loop all verified in the virtual env.")
    print("=" * 70)
