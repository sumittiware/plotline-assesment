"""
The plan -> act -> observe loop, as an explicit LangGraph StateGraph
(DESIGN.md SS5.1) rather than the legacy AgentExecutor -- every LLM turn and
tool call/result is plain Python state in `trace`, not something buried in a
framework callback.

Node responsibilities:
- agent: one LLM turn, wrapped with invoke_llm_with_retry (timeout + retry
         with backoff, src/agent/resilience.py, DESIGN.md SS7). If the LLM
         still fails after all attempts, that's caught here (not raised) and
         routes to the fallback node via should_continue -- a real provider
         outage degrades the response instead of 500-ing the caller.
- tools: dispatches every tool_call from the last AI turn, one at a time,
         via dispatch_tool_with_retry (timeout + retry, but ONLY on timeout --
         see resilience.py for why). Whatever comes out of that (success or
         a still-failing exception) is caught into a structured
         {"error": ...} result so a bad call surfaces to the LLM, which can
         adapt, rather than crashing the whole run.
- fallback: reached either when steps_taken hits MAX_AGENT_STEPS, or when the
            LLM call exhausts its retries (llm_failed). Ends the run with a
            clearly-flagged degraded response -- a small deterministic
            template, NOT a second LLM call -- built from the last
            successfully-resolved segment if one exists.

Compliance override (DESIGN.md SS5.4): create_campaign calls targeting an
external channel (push/email) always get the top consent-compliance chunk
folded into guideline_citations, regardless of whether the LLM thought to
retrieve or cite it -- enforced mechanically here, not left to the prompt.

Client idempotency key (DESIGN.md SS6): run()'s optional idempotency_key,
when supplied, always overrides whatever create_campaign's args carry --
lets a caller retry the same /copilot/run request and land on the same
campaign regardless of what the LLM itself derived or omitted.

Post-hoc grounding (DESIGN.md SS5.3): before a create_campaign call is
actually dispatched, _check_grounding verifies segment_size matches the last
real query_segment result and every cited chunk_id was actually returned by
a search_guidelines call -- checking the STRUCTURED create_campaign
arguments, not grepping the free-text final answer for citation markers
(there's no defined marker format for that, and the structured args are what
actually get persisted, so checking those is both exact and more
consequential). A violation is rejected as a structured tool error fed back
to the LLM, same as any other tool failure -- not a second LLM call, and not
a silent pass-through of an ungrounded claim.

LangSmith tracing (DESIGN.md SS9, optional): run()'s run_id is passed through
as the LangGraph invocation's run_name/metadata, so a run correlates across
three surfaces -- the API response's run_id, our own stdout JSON logs
(src/observability/logging.py), and a LangSmith trace UI, if the caller has
set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY. If those env vars aren't
set (the default), this config is inert -- LangGraph/LangChain simply don't
send anything anywhere; no new dependency, no code branch needed here.
"""
import json
import uuid
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph

from src.agent.llm_client import MockLLMClient
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.resilience import dispatch_tool_with_retry, invoke_llm_with_retry
from src.config import (
    COMPLIANCE_DOC_SLUG,
    EXTERNAL_CHANNELS,
    LLM_TIMEOUT_SECONDS,
    MAX_AGENT_STEPS,
    TOOL_DEFAULT_TIMEOUT_SECONDS,
    TOOL_TIMEOUT_SECONDS,
)

TRUNCATE_CHARS = 300


class CopilotState(TypedDict):
    messages: List[BaseMessage]
    trace: List[dict]
    steps_taken: int
    degraded: bool
    forced_idempotency_key: Optional[str]
    llm_failed: bool
    run_id: str


def _summarize(result: Any) -> str:
    text = json.dumps(result, default=str)
    return text if len(text) <= TRUNCATE_CHARS else text[:TRUNCATE_CHARS] + "...(truncated)"


def last_tool_result(messages: List[BaseMessage], tool_name: str) -> Optional[dict]:
    """
    Untruncated lookup (unlike trace[]'s result_summary) -- used by the
    fallback template to pull the real segment size/filters, not a snippet,
    and by SS5.3 grounding. Public (no leading underscore): src/main.py
    imports this rather than reimplementing it -- it used to, and the two
    copies had already started drifting (one had defensive JSONDecodeError
    handling, the other didn't).
    """
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == tool_name:
            try:
                return json.loads(message.content)
            except (json.JSONDecodeError, TypeError):
                return None
    return None


def _dispatch_tool(tool_call: dict, tools_by_name: Dict[str, StructuredTool]) -> Any:
    tool = tools_by_name.get(tool_call["name"])
    if tool is None:
        return {"error": f"Unknown tool: {tool_call['name']}"}
    timeout_seconds = TOOL_TIMEOUT_SECONDS.get(tool_call["name"], TOOL_DEFAULT_TIMEOUT_SECONDS)
    try:
        return dispatch_tool_with_retry(lambda: tool.invoke(tool_call["args"]), timeout_seconds)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _ensure_compliance_citation(
    args: dict,
    tools_by_name: Dict[str, StructuredTool],
    trace: List[dict],
    messages: List[BaseMessage],
) -> dict:
    """
    Mechanical enforcement of the SS5.4 compliance rule: mutates a copy of
    create_campaign's args so guideline_citations always includes the
    consent-compliance doc for external channels, forcing one extra retrieval
    if the LLM's own citations don't already cover it.

    Also appends a ToolMessage to `messages` (not just a trace entry) for
    this forced retrieval -- _check_grounding scans messages for every
    search_guidelines result to verify citations are real, and this forced
    call is just as real as an LLM-initiated one.
    """
    if args.get("channel") not in EXTERNAL_CHANNELS:
        return args

    citations = list(args.get("guideline_citations") or [])
    if any(c.get("topic_slug") == COMPLIANCE_DOC_SLUG for c in citations):
        return args

    search_tool = tools_by_name.get("search_guidelines")
    if search_tool is None:
        return args

    forced_query = {"query": "consent, compliance and opt-outs", "k": 1, "topic_slug": COMPLIANCE_DOC_SLUG}
    result = _dispatch_tool({"name": "search_guidelines", "args": forced_query}, tools_by_name)
    trace.append(
        {
            "tool": "search_guidelines",
            "input": {**forced_query, "forced_compliance_retrieval": True},
            "result_summary": _summarize(result),
        }
    )
    messages.append(
        ToolMessage(
            content=json.dumps(result, default=str),
            tool_call_id="forced_compliance_retrieval",
            name="search_guidelines",
        )
    )
    if isinstance(result, list):
        citations.extend(result)

    args = dict(args)
    args["guideline_citations"] = citations
    return args


def _check_grounding(args: dict, messages: List[BaseMessage]) -> Optional[str]:
    """
    DESIGN.md SS5.3: refuse to dispatch a create_campaign call whose
    segment_size doesn't match the last real query_segment result, or whose
    guideline_citations include a chunk_id never actually returned by a
    search_guidelines call. Returns an error message (fed back to the LLM as
    a structured tool error, same shape as any other tool failure) if either
    check fails, else None.
    """
    last_segment = last_tool_result(messages, "query_segment")
    if last_segment is None:
        return (
            "No query_segment call found in this conversation -- call query_segment "
            "before proposing a segment_size."
        )

    claimed_size, real_size = args.get("segment_size"), last_segment.get("size")
    if claimed_size != real_size:
        return (
            f"segment_size={claimed_size!r} does not match the last query_segment result "
            f"(size={real_size!r}). Call query_segment again if the segment changed, or use "
            f"segment_size={real_size!r}."
        )

    retrieved_chunk_ids = set()
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == "search_guidelines":
            try:
                chunks = json.loads(message.content)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(chunks, list):
                retrieved_chunk_ids.update(c.get("chunk_id") for c in chunks if isinstance(c, dict))

    cited_chunk_ids = {c.get("chunk_id") for c in (args.get("guideline_citations") or [])}
    unverifiable = cited_chunk_ids - retrieved_chunk_ids - {None}
    if unverifiable:
        return (
            f"guideline_citations references chunk_id(s) {sorted(unverifiable)} that were never "
            "returned by a search_guidelines call in this conversation -- only cite chunk_ids "
            "from actual search_guidelines results."
        )
    return None


def build_graph(llm: Union[MockLLMClient, Any], tools: List[StructuredTool]):
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: CopilotState) -> CopilotState:
        try:
            response = invoke_llm_with_retry(
                lambda: llm_with_tools.invoke(state["messages"]), LLM_TIMEOUT_SECONDS
            )
        except Exception as e:
            # Retries exhausted -- don't crash the run. Flag it and let
            # should_continue route to the fallback node instead.
            state["llm_failed"] = True
            state["trace"].append({"type": "llm_failure", "error": f"{type(e).__name__}: {e}"})
            return state

        state["messages"].append(response)
        state["trace"].append(
            {
                "type": "llm_turn",
                "content": response.content,
                "tool_calls": [tc["name"] for tc in (response.tool_calls or [])],
            }
        )
        return state

    def call_tools(state: CopilotState) -> CopilotState:
        last = state["messages"][-1]
        for tool_call in last.tool_calls:
            args = tool_call["args"]
            grounding_error = None

            if tool_call["name"] == "create_campaign":
                args = _ensure_compliance_citation(args, tools_by_name, state["trace"], state["messages"])
                if state.get("forced_idempotency_key"):
                    # Client-supplied key (DESIGN.md SS6) always wins over
                    # whatever the LLM derived/omitted -- a caller retrying the
                    # same /copilot/run request must land on the same campaign.
                    args = dict(args)
                    args["idempotency_key"] = state["forced_idempotency_key"]
                grounding_error = _check_grounding(args, state["messages"])

            if grounding_error is not None:
                # SS5.3: reject rather than persist an ungrounded claim --
                # same shape as any other tool failure, so the LLM can adapt.
                result = {"error": grounding_error}
            else:
                result = _dispatch_tool({"name": tool_call["name"], "args": args}, tools_by_name)

            state["trace"].append(
                {
                    "tool": tool_call["name"],
                    "input": args,
                    "result_summary": _summarize(result),
                }
            )
            state["messages"].append(
                ToolMessage(
                    content=json.dumps(result, default=str),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
        state["steps_taken"] += 1
        return state

    def fallback(state: CopilotState) -> CopilotState:
        """
        Deterministic template path (DESIGN.md SS7 circuit breaker) -- a
        small explicit function, NOT a second LLM call. Uses the last
        successfully-resolved query_segment result, if any, so the caller
        gets something concrete rather than a bare apology.
        """
        state["degraded"] = True
        reason = "llm_provider_failure" if state.get("llm_failed") else "max_steps_exceeded"
        segment = last_tool_result(state["messages"], "query_segment")

        if segment and "error" not in segment:
            content = (
                f"Automated planning could not be completed ({reason.replace('_', ' ')}). "
                f"Using the last resolved segment ({segment.get('size')} users, filters: "
                f"{segment.get('filters_applied')}), here is a generic fallback draft: "
                "\"We'd love to see you again -- check out what's new!\" "
                "Please review and finalize this campaign manually."
            )
        else:
            content = (
                f"Automated planning could not be completed ({reason.replace('_', ' ')}), and no "
                "segment was resolved yet. Please retry the request."
            )

        state["messages"].append(AIMessage(content=content))
        state["trace"].append(
            {"type": "fallback", "reason": reason, "steps_taken": state["steps_taken"]}
        )
        return state

    def should_continue(state: CopilotState) -> str:
        if state.get("llm_failed"):
            return "fallback"
        if state["steps_taken"] >= MAX_AGENT_STEPS:
            return "fallback"
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    graph = StateGraph(CopilotState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.add_node("fallback", fallback)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END, "fallback": "fallback"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("fallback", END)
    return graph.compile()


def run(
    goal_text: str,
    llm: Union[MockLLMClient, Any],
    tools: List[StructuredTool],
    idempotency_key: Optional[str] = None,
    run_id: Optional[str] = None,
) -> CopilotState:
    run_id = run_id or f"run_{uuid.uuid4()}"
    compiled = build_graph(llm, tools)
    initial_state: CopilotState = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=goal_text)],
        "trace": [],
        "steps_taken": 0,
        "degraded": False,
        "forced_idempotency_key": idempotency_key,
        "llm_failed": False,
        "run_id": run_id,
    }
    # run_name/tags/metadata are LangSmith-facing (inert if tracing isn't
    # enabled) -- run_name carries our own run_id so a trace, if one exists,
    # correlates with the same id in the API response and stdout logs.
    config = {
        "recursion_limit": MAX_AGENT_STEPS * 3 + 5,
        "run_name": run_id,
        "tags": ["campaign-copilot"],
        "metadata": {"goal": goal_text[:200], "app_run_id": run_id},
    }
    return compiled.invoke(initial_state, config)
