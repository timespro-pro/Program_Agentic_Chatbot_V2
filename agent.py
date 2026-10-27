"""
Fast LLM layer.

Python performs the numerical analysis.

The LLM:
- interprets the analysis
- identifies likely business drivers
- structures stakeholder communication
- recommends Finance/Marketing investigation

The model is called only once per request.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Callable

import pandas as pd
import streamlit as st
from openai import OpenAI

from analysis_engine import build_analysis_context
from analysis_points import (
    REPORT_SECTIONS,
    SYSTEM_REPORT_RULES,
)


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = f"""

You are a senior EdTech Finance and Marketing
performance analyst.

Your job is to analyze marketing campaign and
program performance data and explain why a new
batch is performing better or worse than an old batch.

Python has already calculated the numerical metrics.

Do NOT recalculate them.

Do NOT invent numbers.

Do NOT invent missing data.

Do NOT invent causal explanations.

=========================================================
REPORT STRUCTURE
=========================================================

{chr(10).join(REPORT_SECTIONS)}

=========================================================
RESPONSE STYLE: CHAT VS FULL REPORT
=========================================================

This agent serves two different purposes. Choose the response style
based on the user's current question.

A) NORMAL CHAT / FOLLOW-UP QUESTION (DEFAULT):
Write like a helpful, sharp human analyst in a chat conversation.
Start with a direct answer in a natural paragraph. Use the most
relevant figures from the supplied evidence as proof, then add only
2-4 short bullets if they make the answer clearer. Usually keep the
answer to one short paragraph plus a few bullets (roughly 100-180
words; shorter is better when possible).

For normal chat:
- Do NOT generate a full report.
- Do NOT use report headings such as "Key Metrics Overview",
  "Identified Issues", "Summary of Analysis", or "Recommendations".
- Do NOT repeat all metrics or summarize the whole program unless asked.
- Do NOT create a long numbered list, multiple report sections, or a
  large table.
- Answer the exact question first, then explain the evidence briefly.
- Mention specific batch names and values when available.
- Explain percentage-point changes correctly for conversion rates.
- Distinguish observed facts from possible explanations.
- End with one practical next step only when useful.
- For "Is the issue conversion, volume, CPL, CAC?", give a concise
  judgement across those dimensions backed by key values, not a full report.
- Only use a table if the user explicitly asks for one or it is necessary.

B) FULL REPORT:
Use the full stakeholder report structure only when the user explicitly
requests a full/comprehensive analysis, a performance report, or asks to
generate the analysis. In that case, follow the required section order
and provide the appropriate detail.

For either style:
- Use numerical comparisons and percentage-point changes for rates.
- Never invent figures or causes.
- Avoid generic business language and repeating the same point.
- Higher spend is not automatically bad; assess it against output, CPL,
  CAC and conversion.
- Examine Leads -> Applications -> Enrollments, Spend -> CPL -> CAC,
  and source-level evidence where available.

=========================================================
FACT VS HYPOTHESIS
=========================================================

FACT:
Directly supported by the data.

OBSERVATION:
A relationship visible in the data.

HYPOTHESIS:
A possible explanation that requires investigation.

Never present a hypothesis as a fact.

=========================================================
SELECTED BATCH RULE
=========================================================

If the user selected:

New Batch = X
Old Batch = Y

then compare EXACTLY X against Y.

Do not silently substitute another batch.

=========================================================
NUMERICAL RULES
=========================================================

CAC:
Total spend / total enrollments.

CPL:
Total spend / total leads.

Never average source-level CAC or CPL.

Missing spend means unavailable, not zero.

Conversion movement is percentage points.

Higher spend is not automatically bad.

Higher spend should be evaluated against:
- output
- CPL
- CAC
- funnel conversion

Better than plan does not automatically mean
better than the previous batch.

=========================================================
BUSINESS INTERPRETATION
=========================================================

When CAC increases:

Check:

1. Did spend increase?
2. Did leads increase or decrease?
3. Did applications increase or decrease?
4. Did enrollments increase or decrease?
5. Did CPL increase?
6. Did Lead → Application conversion deteriorate?
7. Did Application → Enrollment conversion deteriorate?
8. Which source changed?
9. Is the deterioration broad-based or concentrated?

Do not jump directly to a causal explanation.

=========================================================

{SYSTEM_REPORT_RULES}
"""


# =========================================================
# SECRETS
# =========================================================

def _secret(
    name,
    default=None,
):

    try:

        value = st.secrets.get(
            name,
            default,
        )

        if value:
            return value

    except Exception:
        pass

    return os.getenv(
        name,
        default,
    )


# =========================================================
# CHAT HISTORY
# =========================================================

def _history(
    history,
):

    if not history:
        return []

    result = []

    # Keep only recent context.
    for item in history[-4:]:

        role = item.get(
            "role"
        )

        content = item.get(
            "content",
            "",
        )

        if role not in (
            "user",
            "assistant",
        ):

            continue

        if not content:
            continue

        content = str(
            content
        )

        # Prevent old long reports from
        # making every subsequent request slower.

        if len(content) > 1400:

            content = (
                content[-1400:]
            )

        result.append(

            {

                "role":
                    role,

                "content":
                    content,

            }

        )

    return result


# =========================================================
# FULL REPORT DETECTION
# =========================================================

def _full_report(
    question,
):

    q = (
        question or ""
    ).lower()

    keywords = (

        "comprehensive",

        "full analysis",

        "generate analysis",

        "performance report",

        "stakeholder-ready",

        "performance analysis",

        "overall performance",

        "business performance",

        "analyze program",

        "analyse program",

    )

    return any(
        keyword in q
        for keyword in keywords
    )


# =========================================================
# MAIN AGENT
# =========================================================

def ask_agent(

    question: str,

    df: pd.DataFrame,

    program_code:
        Optional[str] = None,

    old_batch_code:
        Optional[str] = None,

    check_df:
        Optional[pd.DataFrame] = None,

    chat_history:
        Optional[list] = None,

    extra_context:
        Optional[str] = None,

    stream_callback:
        Optional[
            Callable[
                [str],
                None
            ]
        ] = None,

):

    api_key = _secret(
        "OPENAI_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "OPENAI_API_KEY is missing."
        )

    model = _secret(
        "OPENAI_MODEL",
        "gpt-4o-mini",
    )

    client = OpenAI(
        api_key=api_key
    )

    # =====================================================
    # DETERMINISTIC CONTEXT
    # =====================================================

    context = build_analysis_context(

        df=df,

        program_code=
            program_code,

        old_batch_code=
            old_batch_code,

        check_df=
            check_df,

        question=
            question,
    )

    full_report = _full_report(
        question
    )

    # =====================================================
    # SMALL REQUEST PAYLOAD
    # =====================================================

    payload = {

        "question":
            question,

        "selected_new_batch":
            program_code,

        "selected_old_batch":
            old_batch_code,

        "full_report":
            full_report,

        "analysis_context":
            context,

        "additional_data_sources":
            (extra_context or "No additional cached report or Excel workbook context was available."),

        "recent_conversation":
            _history(
                chat_history
            ),

        "instruction": (
            "The user explicitly wants a full analysis/report. Produce the complete "
            "stakeholder report using the required section order."
            if full_report
            else
            "This is a normal chat reply, not a report. Answer the exact question directly "
            "in a natural conversational paragraph, then use at most 2-4 brief bullets if "
            "helpful. Aim for 100-180 words maximum, usually shorter. Do not add report "
            "headings, a full metrics overview, a long identified-issues list, a summary "
            "section, or a recommendations section. Support the answer with a few specific "
            "numbers from the supplied evidence. Use cached report and Excel context when "
            "relevant. Never invent figures; say plainly if the available sources do not "
            "contain the answer."
        ),
    }

    # =====================================================
    # ONE OPENAI CALL
    # =====================================================

    stream = client.responses.create(

        model=model,

        instructions=
            SYSTEM_PROMPT,

        input=json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        ),

        max_output_tokens=(

            1600

            if full_report

            else 600

        ),

        stream=True,
    )

    # =====================================================
    # STREAM RESPONSE
    # =====================================================

    answer_parts = []

    for event in stream:

        event_type = getattr(
            event,
            "type",
            "",
        )

        if (
            event_type
            == "response.output_text.delta"
        ):

            delta = getattr(
                event,
                "delta",
                "",
            )

            if delta:

                answer_parts.append(
                    delta
                )

                if stream_callback:

                    stream_callback(
                        delta
                    )

    answer = "".join(
        answer_parts
    ).strip()

    if not answer:

        raise RuntimeError(
            "The model returned an empty response."
        )

    return answer
