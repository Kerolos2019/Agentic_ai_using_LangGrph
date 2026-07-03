from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def phase_banner(phase_num: int, title: str):
    print(f"\n{'=' * 55}")
    print(f"  PHASE {phase_num}: {title}")
    print(f"{'=' * 55}")

def step_print(icon: str, label: str, detail: str = ""):
    print(f"\n{icon} [{label}] {detail}")


# ── State ─────────────────────────────────────────────
class ApprovalState(TypedDict):
    request: str
    draft: str
    approved: bool
    feedback: str
    final: str


# ── Node 1: LLM generates a draft ────────────────────
def create_draft(state: ApprovalState) -> dict:
    step_print("🤖", "LLM NODE", "Generating draft...")
    response = llm.invoke(f"Create a professional response for: {state['request']}")
    print(f"   Draft preview: {response.content[:100]}...")
    return {"draft": response.content}


# ── Node 2: Human review (interrupt anchor) ───────────
def human_review(state: ApprovalState) -> dict:
    # Graph is interrupted BEFORE this node runs.
    # When resumed, this just passes state through.
    step_print("👁️", "HUMAN NODE", "Human has reviewed the draft.")
    print(f"   Decision: {'APPROVED ✅' if state['approved'] else 'REJECTED ❌'}")
    return state


# ── Routing: Approve → finalize | Reject → revise ────
def route_human_decision(state: ApprovalState) -> Literal["finalize", "revise"]:
    if state["approved"]:
        print("\n→ Routing: APPROVE → finalize")
        return "finalize"
    else:
        print("\n→ Routing: REJECT → revise")
        return "revise"


# ── Node 3a: Approved — use draft as-is ──────────────
def finalize(state: ApprovalState) -> dict:
    step_print("✅", "FINALIZE NODE", "Draft approved — using as-is.")
    return {"final": state["draft"]}


# ── Node 3b: Rejected — revise based on feedback ─────
def revise(state: ApprovalState) -> dict:
    step_print("✏️", "REVISE NODE", "Revising based on human feedback...")
    print(f"   Feedback: \"{state['feedback']}\"")
    response = llm.invoke(
        f"Revise this draft based on feedback:\n\n"
        f"Draft: {state['draft']}\n\n"
        f"Feedback: {state['feedback']}"
    )
    print(f"   Revised draft preview: {response.content[:100]}...")
    return {"final": response.content}


# ── Build graph ───────────────────────────────────────
#
#   START → create_draft → ⏸ → human_review ──(approve)──→ finalize → END
#                                             ╚─(reject)──→ revise   → END
#
builder = StateGraph(ApprovalState)

builder.add_node("create_draft", create_draft)
builder.add_node("human_review", human_review)
builder.add_node("finalize", finalize)
builder.add_node("revise", revise)

builder.add_edge(START, "create_draft")
builder.add_edge("create_draft", "human_review")

# Conditional edge: human decision routes to different nodes
builder.add_conditional_edges(
    "human_review",
    route_human_decision,
    {
        "finalize": "finalize",
        "revise":   "revise",
    }
)

builder.add_edge("finalize", END)
builder.add_edge("revise", END)

# Compile — interrupt BEFORE human_review so human acts between draft and routing
memory = MemorySaver()
app = builder.compile(
    checkpointer=memory,
    interrupt_before=["human_review"]
)

# ── Save graph image ──────────────────────────────────
png_data = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)
print("Graph saved to graph.png")

# ── Config ────────────────────────────────────────────
config = {"configurable": {"thread_id": "hitl-routing-001"}}

print("\n" + "=" * 55)
print("  HUMAN-IN-THE-LOOP: APPROVE / REJECT ROUTING")
print("=" * 55)
print("\n  Graph:")
print("  START → [LLM] → ⏸ → [Human] ──approve──→ [finalize] → END")
print("                               ╚──reject──→ [revise]   → END")

# ════════════════════════════════════
# PHASE 1: Run until interrupt
# ════════════════════════════════════
phase_banner(1, "RUN UNTIL INTERRUPT")

result = app.invoke(
    {
        "request":  "Write a thank-you email for a job interview",
        "draft":    "",
        "approved": False,
        "feedback": "",
        "final":    "",
    },
    config,
)

step_print("⏸️", "PAUSED", "Graph frozen before human_review node.")
print(f"   Draft ready: {result['draft'][:120]}...")
print(f"   Waiting for human decision...")

# ════════════════════════════════════
# PHASE 2: Inspect paused state
# ════════════════════════════════════
phase_banner(2, "INSPECT PAUSED STATE")

current_state = app.get_state(config)
print(f"   Next node(s): {current_state.next}")
print(f"   approved:     {current_state.values['approved']}")
print(f"   feedback:     '{current_state.values['feedback']}'")
print(f"   final:        '{current_state.values['final']}'")

# ════════════════════════════════════
# PHASE 3a: Human REJECTS → revise
# ════════════════════════════════════
phase_banner(3, "SCENARIO A — HUMAN REJECTS")

feedback_text = input("\nPlease provide feedback")
print(f"   Human decision: REJECT")
print(f"   Feedback: \"{feedback_text}\"")

app.update_state(config, {"approved": False, "feedback": feedback_text})
print(f"\n   Resuming graph...")

result_a = app.invoke(None, config)

step_print("🔀", "ROUTED TO", "revise node (because approved=False)")
print(f"   Final result: {result_a['final'][:200]}...")



print("\n" + "=" * 55)
print("  SUMMARY")
print("=" * 55)
print("  approved=False → conditional edge → [revise]   → END")
print("  approved=True  → conditional edge → [finalize] → END")
print("  Both paths driven by human decision injected via update_state()")