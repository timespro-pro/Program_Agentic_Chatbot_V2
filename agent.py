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
STAKEHOLDER REPORT STYLE
=========================================================

The output should look like a professional
business-performance analysis.

Use:

- concise sections
- tables where useful
- numerical comparisons
- percentage changes
- percentage-point changes for conversion
- clear red-flag areas
- specific Finance actions
- specific Marketing actions

Avoid:

- generic business language
- excessive narrative
- repeating the same point
- unsupported assumptions
- long introductions

The key question is:

WHERE DID THE NEW BATCH FALL OFF
COMPARED WITH THE OLD BATCH?

Follow the funnel:

Leads
↓
Applications
↓
Enrollments

Then examine:

Spend
↓
CPL
↓
CAC

Then:

Source
↓
Funnel
↓
Program

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

            "Produce the complete "
            "stakeholder report using "
            "the required section order."

            if full_report

            else

            "Answer only the user's specific question using the supplied evidence. "
            "Use additional cached report and Excel workbook context when relevant. "
            "Never invent figures; clearly say when the available sources do not contain the answer."

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
