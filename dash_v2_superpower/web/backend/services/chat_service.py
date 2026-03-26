"""web/backend/services/chat_service.py

Per-request agent loop with SSE event generation.

Key design:
- Does NOT modify global TOOL_HANDLERS in agent.py (concurrent safety).
- Builds local_handlers dict per request, so concurrent requests are isolated.
- asyncio.Queue bridges the sync thread to the async SSE generator.
- auto_compact and micro_compact are applied to manage context size.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List

# Repo root injected by main.py sys.path
from agent import (
    client, TOOLS, SYSTEM, MODEL,
    micro_compact, auto_compact, estimate_tokens, COMPACT_THRESHOLD,
)
from query_db import query_acs_data, get_column_names
from services.map_service import join_geojson, compute_stats

_SENTINEL = object()  # signals queue exhausted


def _infer_level(rows: List[Dict]) -> str:
    """Infer geographic level from ACS result rows."""
    if rows and rows[0].get("zipcode"):
        return "zipcode"
    if rows and rows[0].get("county"):
        return "county"
    return "state"


def _build_data_payload(rows: List[Dict], columns: List[str]) -> Dict[str, Any]:
    """Build the 'data' SSE payload: rows + GeoJSON + stats."""
    from services.map_service import _load_geojson  # lazy import to avoid circular

    level = _infer_level(rows)

    if level == "state":
        raw_gj = _load_geojson("states.geojson")
    elif level == "county":
        raw_gj = _load_geojson("counties.geojson")
    else:
        raw_gj = _load_geojson("zcta_all.geojson")

    geojson = join_geojson(raw_gj, rows, level=level)

    # Pick first non-identifier column for stats
    id_cols = {"zipcode", "city", "county", "state", "year", "geoid"}
    stat_col = next((c for c in columns if c not in id_cols), None)
    stats = compute_stats(rows, stat_col) if stat_col else None

    return {
        "rows": rows,
        "columns": columns,
        "geojson": geojson,
        "stats": stats,
    }


def _run_agent_sync(
    input_messages: List[Dict],
    result_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Synchronous agent loop — run in a thread via asyncio.to_thread."""

    def wrapped_query_acs(**kwargs):
        result = query_acs_data(**kwargs)
        cols = list(result[0].keys()) if result else []
        payload = _build_data_payload(result, cols)
        loop.call_soon_threadsafe(result_queue.put_nowait, payload)
        return result

    local_handlers = {
        "QueryACSData":   lambda **kw: wrapped_query_acs(**kw),
        "GetColumnNames": lambda **kw: get_column_names(),
        "Compact":        lambda **kw: "__COMPACT__",
    }

    system_msg = {"role": "system", "content": SYSTEM}
    manual_compact = False

    try:
        while True:
            micro_compact(input_messages)
            if estimate_tokens(input_messages) > COMPACT_THRESHOLD:
                input_messages[:] = auto_compact(input_messages)

            response = client.chat.completions.create(
                model=MODEL,
                messages=[system_msg] + input_messages,
                tools=TOOLS,
            )
            msg = response.choices[0].message
            input_messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                break

            manual_compact = False
            for tc in msg.tool_calls:
                handler = local_handlers.get(tc.function.name)
                try:
                    output = (
                        handler(**json.loads(tc.function.arguments))
                        if handler
                        else f"Unknown tool: {tc.function.name}"
                    )
                except Exception as e:
                    output = f"Error: {e}"

                if output == "__COMPACT__":
                    manual_compact = True
                    output = "Compressing..."

                input_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(output),
                })

            if manual_compact:
                input_messages[:] = auto_compact(input_messages)

    finally:
        loop.call_soon_threadsafe(result_queue.put_nowait, _SENTINEL)


async def chat_stream(
    message: str,
    history: List[Dict[str, str]],
) -> AsyncGenerator[Dict, None]:
    """
    Async generator yielding SSE dicts: {"event": ..., "data": ...}
    Events: text | data | error | done (always ends with done)
    """
    input_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history
    ] + [{"role": "user", "content": message}]

    result_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    agent_future = None

    try:
        agent_future = asyncio.create_task(
            asyncio.to_thread(_run_agent_sync, input_messages, result_queue, loop)
        )

        # Drain queue until sentinel
        while True:
            item = await result_queue.get()
            if item is _SENTINEL:
                break
            yield {"event": "data", "data": json.dumps(item, default=str)}

        await agent_future

        # Final text response (agent_loop returns None; result is in last message)
        last_content = ""
        for msg in reversed(input_messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                last_content = msg.get("content") or ""
                break
        yield {"event": "text", "data": json.dumps(last_content)}
        yield {"event": "done", "data": ""}

    except Exception as e:
        yield {"event": "error", "data": str(e)}
        yield {"event": "done", "data": ""}

    finally:
        if agent_future is not None and not agent_future.done():
            agent_future.cancel()
            try:
                await agent_future
            except (asyncio.CancelledError, Exception):
                pass
