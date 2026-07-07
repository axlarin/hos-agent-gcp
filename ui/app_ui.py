"""HOS Agent — Streamlit UI

Connects to the HOS Agent REST API (Cloud Run or local).
Deployed on Streamlit Community Cloud at:
  https://axlarin-hos-agent-gcp-app-ui.streamlit.app (or similar)

Tabs:
  💬 Chat   — multi-turn conversation with session history
  📊 Report — configurable health profile workflow report
  🧪 Evaluate — full 13-case evaluation suite with component scores
"""

import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HOS Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "https://hos-agent-481249180021.us-east1.run.app"

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏥 HOS Agent")
    st.caption("Medicare Health Outcomes Survey — multi-agent analysis system")
    st.divider()

    base_url = st.text_input(
        "API URL",
        value=API_URL,
        help="Change to http://localhost:8080 to use a local server",
    ).rstrip("/")

    timeout = st.slider(
        "Request timeout (s)", 30, 300, 120, 10,
        help="First call after idle takes 15–20 s (Cloud Run cold start)",
    )

    st.divider()

    if st.button("🔍 Check server health", use_container_width=True):
        with st.spinner("Checking…"):
            try:
                r = requests.get(f"{base_url}/health", timeout=15)
                h = r.json()
                if h.get("status") == "ok":
                    st.success("Server is healthy ✅")
                    st.json({k: v for k, v in h.items() if k != "status"})
                else:
                    st.warning(str(h))
            except requests.exceptions.Timeout:
                st.error("Timeout — server may be cold-starting, try again in 20 s")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect — check the API URL above")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.markdown(
        "**Available datasets**\n"
        "- `c25a` — Cohort 25 analytic PUF (263k rows)\n"
        "- `c26b` — Cohort 26 baseline PUF (344k rows)\n"
        "- `c27b` — Cohort 27 baseline PUF (341k rows)"
    )
    st.divider()
    st.caption("Built with Google ADK · Gemini 2.5 · FastAPI · ChromaDB")

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_chat, tab_report, tab_evaluate = st.tabs(
    ["💬 Chat", "📊 Report", "🧪 Evaluate"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — CHAT
# ─────────────────────────────────────────────────────────────────────────────

with tab_chat:
    st.subheader("Chat with the HOS Agent")
    st.caption(
        "Ask about HOS definitions, methodology, dataset columns, or request "
        "statistical analyses. The agent maintains conversation context across turns."
    )

    # Example questions
    with st.expander("Example questions to try"):
        st.markdown(
            "- What does PCS mean?\n"
            "- What is the HOS survey methodology?\n"
            "- What datasets are available?\n"
            "- What columns does c25a have?\n"
            "- What predicts general health status in c25a?\n"
            "- Is there a significant association between AGE and health status in c25a?\n"
            "- Compare health scores between males and females in c25a\n"
            "- Give me a comprehensive analysis of general health status in c25a"
        )

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Clear button
    if st.session_state.messages:
        if st.button("🗑 Clear conversation", key="clear_chat"):
            st.session_state.messages = []
            st.session_state.session_id = None
            try:
                requests.post(f"{base_url}/clear", timeout=10)
            except Exception:
                pass
            st.rerun()

    # Chat input
    question = st.chat_input("Ask a question about HOS data…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("⏳ Thinking… *(first call may take 15–20 s on a cold start)*")
            try:
                r = requests.post(
                    f"{base_url}/query",
                    json={"question": question, "session_id": st.session_state.session_id},
                    timeout=timeout,
                )
                r.raise_for_status()
                data = r.json()
                answer = data.get("answer", "*(no answer returned)*")
                st.session_state.session_id = data.get("session_id")
                placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except requests.exceptions.Timeout:
                msg = (
                    f"⏱ **Request timed out** after {timeout} s. "
                    "Hit **'Check server health'** in the sidebar to wake the instance, "
                    "wait for the green tick, then try again."
                )
                placeholder.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            except requests.exceptions.ConnectionError:
                msg = f"❌ **Cannot connect** to `{base_url}`. Check the API URL in the sidebar."
                placeholder.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 503:
                    msg = (
                        "🔄 **Server cold-starting (503).** "
                        "Hit **'Check server health'** in the sidebar, wait for the green tick, "
                        "then resend your question."
                    )
                else:
                    msg = f"❌ **HTTP error:** {exc}"
                placeholder.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            except Exception as exc:
                try:
                    detail = r.json().get("detail", str(exc))
                except Exception:
                    detail = str(exc)
                msg = f"❌ **Error:** {detail}"
                placeholder.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

    if st.session_state.session_id:
        st.caption(f"Session: `{st.session_state.session_id}`")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — REPORT
# ─────────────────────────────────────────────────────────────────────────────

with tab_report:
    st.subheader("Health Profile Report")
    st.caption(
        "Runs a fixed multi-step workflow on one target variable: "
        "frequency distribution → top predictors → group comparisons (sex, age) → "
        "Pearson correlates. Steps are skipped automatically when statistical "
        "assumptions are not met."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        r_dataset = st.selectbox(
            "Dataset", ["c25a", "c26b", "c27b"],
            help="Select the HOS cohort to analyse",
        )
        r_target = st.text_input(
            "Target column (plain English or column code)",
            value="general health status",
            help='e.g. "general health status", "physical health score", "B25VRGENHTH"',
        )

    with col_right:
        r_missing = st.slider(
            "Abort if target is > X% missing", 10, 90, 50, 5,
            help="Global gate: aborts the whole report if the target column exceeds this threshold",
        ) / 100
        r_max_unique = st.slider(
            "Max unique values for classification", 5, 50, 20, 5,
            help="Skips Random Forest if target has more unique values than this (likely continuous)",
        )

    run_report = st.button("▶ Generate report", type="primary", use_container_width=True)

    if run_report:
        if not r_target.strip():
            st.warning("Please enter a target column name.")
        else:
            with st.spinner(f"Running health_profile workflow on '{r_target}' in {r_dataset}…"):
                try:
                    r = requests.post(
                        f"{base_url}/report",
                        json={
                            "dataset": r_dataset,
                            "target_column": r_target,
                            "workflow": "health_profile",
                            "max_missing_pct": r_missing,
                            "max_unique_for_classification": r_max_unique,
                        },
                        timeout=timeout,
                    )
                    r.raise_for_status()
                    result = r.json()

                    # Summary metrics
                    summ = result.get("summary", {})
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("✅ Completed", summ.get("completed", 0))
                    m2.metric("⏭ Skipped", summ.get("skipped", 0))
                    m3.metric("❌ Failed", summ.get("failed", 0))
                    m4.metric("Total steps", summ.get("total", 0))

                    # Resolved column
                    resolved = result.get("resolved_column", "")
                    if resolved:
                        st.info(f"Column resolved: **{r_target}** → `{resolved}`")

                    # Execution trace
                    steps = result.get("steps", [])
                    if steps:
                        st.divider()
                        st.markdown("##### Execution trace")
                        trace_rows = []
                        for s in steps:
                            icon = {"completed": "✅", "skipped": "⏭", "failed": "❌"}.get(
                                s["status"], s["status"]
                            )
                            trace_rows.append({
                                "Step": s["name"],
                                "Status": f"{icon} {s['status']}",
                                "ms": round(s.get("duration_ms", 0)),
                                "Note": s.get("reason", ""),
                            })
                        st.dataframe(trace_rows, use_container_width=True, hide_index=True)

                    # Full report
                    st.divider()
                    st.markdown(result.get("report_markdown", ""))

                except requests.exceptions.Timeout:
                    st.error(f"⏱ Timed out after {timeout} s. Try again or increase the timeout.")
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to `{base_url}`.")
                except requests.exceptions.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code == 503:
                        st.error("🔄 Server cold-starting (503). Click 'Check server health' in the sidebar, wait for the green tick, then try again.")
                    else:
                        st.error(f"❌ HTTP error: {exc}")
                except Exception as exc:
                    try:
                        detail = r.json().get("detail", str(exc))
                    except Exception:
                        detail = str(exc)
                    st.error(f"❌ {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

with tab_evaluate:
    st.subheader("Evaluation Suite")
    st.caption(
        "Runs all 13 ground-truth test cases and scores each across five "
        "components. Multi-turn follow-up cases are reported separately."
    )

    deep_mode = st.radio(
        "Evaluation mode",
        options=["false", "auto", "true"],
        index=1,
        horizontal=True,
        help=(
            "**false** — embedding only, fast, no quota  |  "
            "**auto** — embedding + selective Gemini escalation (~6 calls)  |  "
            "**true** — Gemini judge on every case"
        ),
    )

    st.warning(
        "⏱ The suite runs 13 agent queries with 3-second pacing between each. "
        "Expect **8–12 minutes** total. Keep this tab open."
    )

    run_eval = st.button("▶ Run evaluation suite", type="primary")

    if run_eval:
        with st.spinner("Running suite — please wait (8–12 minutes)…"):
            try:
                r = requests.post(
                    f"{base_url}/evaluate/suite",
                    params={"deep": deep_mode},
                    timeout=900,
                )
                r.raise_for_status()
                data = r.json()

                scored = data.get("results", [])
                context_dep = data.get("context_dependent_results", [])
                instr = data.get("instrumentation_summary", {})

                passed = sum(1 for c in scored if c.get("evaluation", {}).get("passed"))
                total = len(scored)

                # Result banner
                if passed == total:
                    st.success(f"✅ {passed}/{total} scored cases passed")
                elif passed >= total * 0.8:
                    st.success(f"✅ {passed}/{total} scored cases passed")
                else:
                    st.warning(f"⚠️ {passed}/{total} scored cases passed")

                # Instrumentation summary
                if instr:
                    i1, i2, i3 = st.columns(3)
                    i1.metric("Mode", str(instr.get("deep_mode", "—")))
                    i2.metric("Embedding evaluations", instr.get("total_embedding_evaluations", "—"))
                    i3.metric("Gemini calls", instr.get("total_gemini_calls", "—"))

                st.divider()

                # Per-case results table
                def _comp(result, name):
                    comps = result.get("evaluation", {}).get("components", [])
                    c = next((x for x in comps if x["component"] == name), {})
                    v = c.get("score", 0)
                    esc = "🔺" if c.get("escalated") else ""
                    return f"{v:.2f}{esc}"

                rows = []
                for c in scored:
                    ev = c.get("evaluation", {})
                    rows.append({
                        "": "✅" if ev.get("passed") else "❌",
                        "Question": c["question"][:55] + ("…" if len(c["question"]) > 55 else ""),
                        "Overall": f"{ev.get('overall_score', 0):.3f}",
                        "Route": _comp(c, "routing"),
                        "Tool": _comp(c, "tool_selection"),
                        "Retrieval": _comp(c, "retrieval_relevance"),
                        "Faithful": _comp(c, "answer_faithfulness"),
                        "Quality": _comp(c, "answer_quality"),
                    })

                st.markdown("##### Scored cases")
                st.dataframe(rows, use_container_width=True, hide_index=True)

                # Context-dependent cases
                if context_dep:
                    st.divider()
                    st.markdown(
                        "##### Context-dependent cases *(excluded from pass count)*"
                    )
                    st.caption(
                        "These require conversation history from a prior turn — "
                        "failing them is a test-harness limitation, not an agent defect."
                    )
                    cd_rows = []
                    for c in context_dep:
                        ev = c.get("evaluation", {})
                        cd_rows.append({
                            "Question": c["question"],
                            "Overall": f"{ev.get('overall_score', 0):.3f}",
                        })
                    st.dataframe(cd_rows, use_container_width=True, hide_index=True)

                # Gemini escalation details
                escalations = instr.get("escalations", [])
                if escalations:
                    st.divider()
                    st.markdown("##### Gemini escalations (auto mode)")
                    st.caption("🔺 marks escalated dimensions in the table above.")
                    for e in escalations:
                        with st.expander(e["question"][:70]):
                            for reason in e.get("reasons", []):
                                st.markdown(f"- {reason}")

                # Answer preview
                st.divider()
                st.markdown("##### Answer preview")
                selected_q = st.selectbox(
                    "Select a case to see the full answer",
                    options=[c["question"] for c in scored],
                    key="eval_preview_select",
                )
                match = next((c for c in scored if c["question"] == selected_q), None)
                if match:
                    ev = match.get("evaluation", {})
                    st.markdown(f"**Overall: {ev.get('overall_score', 0):.3f}** — "
                                f"{'✅ PASS' if ev.get('passed') else '❌ FAIL'}")
                    st.markdown(match.get("answer", "*(no answer)*"))

            except requests.exceptions.Timeout:
                st.error(
                    "⏱ Suite timed out. This is unusual — the server may have "
                    "crashed mid-run. Check `/health` and try again."
                )
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to `{base_url}`.")
            except Exception as exc:
                try:
                    detail = r.json().get("detail", str(exc))
                except Exception:
                    detail = str(exc)
                st.error(f"❌ {detail}")
