import re
import pandas as pd
import streamlit as st
from transformers import pipeline

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-0.6B"

st.set_page_config(
    page_title="AI Support Ticket Assistant",
    page_icon="🎫",
    layout="wide",
)

SAMPLE_TICKETS = pd.DataFrame(
    [
        {
            "ticket_id": "TKT-1001",
            "ticket_text": (
                "I was charged twice for my monthly subscription this morning. "
                "Please check the duplicate payment and tell me how it can be corrected."
            ),
        },
        {
            "ticket_id": "TKT-1002",
            "ticket_text": (
                "I cannot sign in to my account. The password reset link says it has "
                "expired even though I requested it just now."
            ),
        },
        {
            "ticket_id": "TKT-1003",
            "ticket_text": (
                "My order was marked as delivered but I have not received the package. "
                "The order number is ORD-7842."
            ),
        },
        {
            "ticket_id": "TKT-1004",
            "ticket_text": (
                "The mobile app closes whenever I open the invoices page. This has "
                "happened three times today and I cannot download my invoice."
            ),
        },
        {
            "ticket_id": "TKT-1005",
            "ticket_text": (
                "I would like to return a headset I purchased last week. "
                "Please explain the return process."
            ),
        },
    ]
)

CATEGORIES = [
    "Billing",
    "Technical Issue",
    "Account Access",
    "Delivery",
    "Returns",
    "General",
]

PRIORITIES = ["Low", "Medium", "High"]


# ---------------------------------------------------------
# Load the Hugging Face model once and cache it
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Downloading/loading the Hugging Face model...")
def load_model():
    """
    Loads a small chat model from Hugging Face Hub.
    device=-1 forces CPU inference, so a CUDA GPU is not required.
    The first run needs internet access to download model files.
    """
    text_generator = pipeline(
        task="text-generation",
        model=MODEL_ID,
        device=-1,
    )
    return text_generator


# ---------------------------------------------------------
# Generate a response from the chat model
# ---------------------------------------------------------
def generate_text(text_generator, prompt, max_new_tokens=120):
    """
    Sends a chat-style message to the Transformers text-generation pipeline.
    /no_think asks Qwen3 to use its non-thinking response mode.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant helping a customer support team. "
                "Be concise, accurate, polite, and professional. "
                "Do not invent facts. Do not claim that a refund, replacement, "
                "investigation, or other action has already been completed. "
                "Return only the requested answer."
            ),
        },
        {
            "role": "user",
            "content": prompt + "\n\n/no_think",
        },
    ]

    output = text_generator(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generated_chat = output[0]["generated_text"]

    # In chat mode, Transformers returns the conversation as message dictionaries.
    if isinstance(generated_chat, list):
        answer = generated_chat[-1].get("content", "")
    else:
        answer = str(generated_chat)

    # Defensive cleanup in case a model response contains a think block.
    answer = re.sub(
        r"<think>.*?</think>",
        "",
        answer,
        flags=re.DOTALL,
    ).strip()

    return answer


# ---------------------------------------------------------
# Ticket analysis
# ---------------------------------------------------------
def normalize_category(model_output):
    cleaned = model_output.strip().lower()

    for category in CATEGORIES:
        if cleaned == category.lower() or category.lower() in cleaned:
            return category

    return "Needs human review"


def normalize_priority(model_output):
    for priority in PRIORITIES:
        if re.search(r"\b" + re.escape(priority) + r"\b", model_output, re.IGNORECASE):
            return priority

    return "Needs human review"


def analyse_ticket(text_generator, ticket_text):
    """
    Uses separate prompts for each task.
    The character limit helps prevent extremely long tickets from consuming
    excessive memory or context.
    """
    ticket_text = str(ticket_text).strip()

    if not ticket_text:
        raise ValueError("Ticket description cannot be empty.")

    if len(ticket_text) > 5000:
        ticket_text = ticket_text[:5000]

    summary = generate_text(
        text_generator,
        (
            "Summarize the following customer support ticket in one or two "
            "short sentences. Include the main issue and any important detail.\n\n"
            f"Ticket:\n{ticket_text}"
        ),
        max_new_tokens=100,
    )

    category_raw = generate_text(
        text_generator,
        (
            "Classify this customer support ticket into exactly ONE category "
            "from this list:\n"
            "Billing, Technical Issue, Account Access, Delivery, Returns, General.\n"
            "Return only the category name. Do not explain.\n\n"
            f"Ticket:\n{ticket_text}"
        ),
        max_new_tokens=20,
    )

    priority_raw = generate_text(
        text_generator,
        (
            "Classify the priority of this support ticket as exactly ONE of "
            "these values: Low, Medium, High.\n"
            "Use High only for urgent service disruption, security concerns, "
            "or major customer impact. Return only the priority value.\n\n"
            f"Ticket:\n{ticket_text}"
        ),
        max_new_tokens=15,
    )

    draft_response = generate_text(
        text_generator,
        (
            "Draft a polite and concise customer support reply to this ticket. "
            "Acknowledge the issue. Do not claim that an action has already "
            "been completed. If necessary information is missing, ask the "
            "customer to provide it. Do not invent company policies.\n\n"
            f"Ticket:\n{ticket_text}"
        ),
        max_new_tokens=150,
    )

    return {
        "summary": summary,
        "category": normalize_category(category_raw),
        "priority": normalize_priority(priority_raw),
        "draft_response": draft_response,
        "raw_category": category_raw,
        "raw_priority": priority_raw,
    }


# ---------------------------------------------------------
# Streamlit user interface
# ---------------------------------------------------------
st.title("🎫 AI-Powered Customer Support Ticket Assistant")
st.write(
    "Analyse customer support tickets using a pretrained Hugging Face chat model. "
    "Review all AI-generated suggestions before taking action."
)

with st.sidebar:
    st.header("Project information")
    st.markdown(f"**Model:** `{MODEL_ID}`")
    st.markdown("**Framework:** Hugging Face Transformers")
    st.markdown("**Interface:** Streamlit")
    st.markdown("**Inference:** Local CPU")
    st.divider()
    st.warning(
        "This is a learning prototype. Model outputs can be wrong. "
        "Do not enter real personal, financial, credential, or confidential data."
    )
    st.caption(
        "An internet connection is required for the first model download. "
        "The model is cached locally for later runs."
    )

single_tab, csv_tab = st.tabs(
    ["Analyse a ticket", "Upload / select CSV"]
)


# ---------------------------------------------------------
# Tab 1: Manual ticket analysis
# ---------------------------------------------------------
with single_tab:
    st.subheader("Enter customer ticket details")

    ticket_id = st.text_input(
        "Ticket ID",
        value="TKT-1001",
        key="manual_ticket_id",
    )

    ticket_text = st.text_area(
        "Customer issue",
        height=160,
        placeholder=(
            "Example: I was charged twice for my monthly subscription. "
            "Please check the duplicate payment."
        ),
        key="manual_ticket_text",
    )

    analyse_button = st.button(
        "Analyse ticket",
        type="primary",
        disabled=not ticket_text.strip(),
        key="analyse_manual",
    )

    if analyse_button:
        try:
            model = load_model()

            with st.spinner(
                "The model is analysing the ticket. This may take a little while on CPU..."
            ):
                result = analyse_ticket(model, ticket_text)

            st.session_state["manual_result"] = {
                "ticket_id": ticket_id.strip() or "Not provided",
                "ticket_text": ticket_text.strip(),
                **result,
            }

        except Exception as error:
            st.error(f"Could not load or run the model: {error}")
            st.info(
                "Check the terminal for the full error. If this is the first run, "
                "also check your internet connection and available memory."
            )

    if "manual_result" in st.session_state:
        result = st.session_state["manual_result"]

        st.divider()
        st.subheader(f"Analysis result — {result['ticket_id']}")

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric("Suggested category", result["category"])

        with metric_col2:
            st.metric("Suggested priority", result["priority"])

        st.markdown("#### Ticket summary")
        st.write(result["summary"])

        st.markdown("#### Draft customer response")
        st.caption("Edit and review this response. The application does not send it.")
        edited_response = st.text_area(
            "Edit response before use",
            value=result["draft_response"],
            height=160,
            key="manual_edited_response",
        )

        export_data = {
            "ticket_id": result["ticket_id"],
            "ticket_text": result["ticket_text"],
            "summary": result["summary"],
            "category": result["category"],
            "priority": result["priority"],
            "draft_response": edited_response,
        }

        export_df = pd.DataFrame([export_data])

        st.download_button(
            label="Download analysis as CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="ticket_analysis.csv",
            mime="text/csv",
        )

        with st.expander("View raw model category and priority outputs"):
            st.write("Raw category output:", result["raw_category"])
            st.write("Raw priority output:", result["raw_priority"])


# ---------------------------------------------------------
# Tab 2: CSV upload and sample ticket analysis
# ---------------------------------------------------------
with csv_tab:
    st.subheader("Upload a ticket CSV or use sample records")

    st.write("Sample tickets")
    st.dataframe(
        SAMPLE_TICKETS,
        use_container_width=True,
        hide_index=True,
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Your CSV must contain ticket_id and ticket_text columns.",
    )

    if uploaded_file is not None:
        try:
            ticket_df = pd.read_csv(uploaded_file)
        except Exception as error:
            st.error(f"Could not read the CSV file: {error}")
            ticket_df = pd.DataFrame()
    else:
        ticket_df = SAMPLE_TICKETS.copy()

    required_columns = {"ticket_id", "ticket_text"}

    if not ticket_df.empty and required_columns.issubset(ticket_df.columns):
        ticket_df = ticket_df[["ticket_id", "ticket_text"]].fillna("")

        ticket_options = ticket_df["ticket_id"].astype(str).tolist()

        selected_ticket_id = st.selectbox(
            "Select a ticket ID",
            options=ticket_options,
            key="csv_ticket_selector",
        )

        selected_row = ticket_df[
            ticket_df["ticket_id"].astype(str) == selected_ticket_id
        ].iloc[0]

        selected_ticket_text = str(selected_row["ticket_text"])

        st.text_area(
            "Selected ticket description",
            value=selected_ticket_text,
            height=130,
            disabled=True,
            key="selected_ticket_preview",
        )

        if st.button(
            "Analyse selected ticket",
            type="primary",
            key="analyse_csv_ticket",
            disabled=not selected_ticket_text.strip(),
        ):
            try:
                model = load_model()

                with st.spinner(
                    "The model is analysing the selected ticket..."
                ):
                    csv_result = analyse_ticket(
                        model,
                        selected_ticket_text,
                    )

                st.session_state["csv_result"] = {
                    "ticket_id": selected_ticket_id,
                    "ticket_text": selected_ticket_text,
                    **csv_result,
                }

            except Exception as error:
                st.error(f"Could not load or run the model: {error}")

        if "csv_result" in st.session_state:
            result = st.session_state["csv_result"]

            st.divider()
            st.subheader(f"Analysis result — {result['ticket_id']}")

            st.write(f"**Category:** {result['category']}")
            st.write(f"**Priority:** {result['priority']}")

            st.markdown("#### Summary")
            st.write(result["summary"])

            st.markdown("#### Draft customer response")
            st.write(result["draft_response"])

            csv_export = pd.DataFrame(
                [
                    {
                        "ticket_id": result["ticket_id"],
                        "ticket_text": result["ticket_text"],
                        "summary": result["summary"],
                        "category": result["category"],
                        "priority": result["priority"],
                        "draft_response": result["draft_response"],
                    }
                ]
            )

            st.download_button(
                "Download selected ticket analysis",
                data=csv_export.to_csv(index=False).encode("utf-8"),
                file_name=f"{result['ticket_id']}_analysis.csv",
                mime="text/csv",
            )

    elif not ticket_df.empty:
        st.error(
            "Invalid CSV format. Please include both columns: "
            "`ticket_id` and `ticket_text`."
        )


st.divider()
st.caption(
    "Prototype for learning and demonstration. AI-generated results require "
    "human validation and must not be treated as automated decisions."
)
