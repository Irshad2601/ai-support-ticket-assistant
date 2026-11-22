# AI-Powered Customer Support Ticket Assistant

A small learning project that uses a pretrained text-to-text model from the Hugging Face Hub and a Streamlit interface. It demonstrates how an LLM-style model can help a support team review incoming tickets.

## What it does

For one selected support ticket, the app generates:
- A short ticket summary
- A suggested category (Billing, Technical Issue, Account Access, Delivery, Returns, or General)
- A suggested priority (Low, Medium, or High)
- A draft customer response for a human to review and edit

You can type a ticket or select one from the sample CSV / upload your own CSV.

## Technology

- Python
- Hugging Face Transformers
- `google/flan-t5-base` pretrained model
- Streamlit
- Pandas

The app loads model weights and tokenizer from the Hugging Face Hub using Transformers. It does **not** use LangChain or LangGraph, and it does not train or fine-tune the model.

## Run locally

### 1. Install Python

Use Python 3.10 or 3.11. Confirm it is installed:

```bash
python --version
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start Streamlit

Run this command from the project folder:

```bash
streamlit run app.py
```

Streamlit will open the app in your browser. The first analysis may take longer because the pretrained model must be downloaded. An internet connection is needed for that first download. Subsequent runs normally use the local model cache.

## CSV format

Your CSV must include these two columns:

| ticket_id | ticket_text |
|---|---|
| TKT-2001 | Describe the customer's issue here |

The app analyses one selected ticket at a time to keep the demo manageable on a normal laptop.

## Important limitations

- This is a prototype for learning and demonstration, not a production customer-support system.
- `google/flan-t5-base` is a pretrained instruction-following text-to-text model. It may produce incomplete, inaccurate, or inconsistent labels and replies.
- Category and priority are model suggestions. Always review them.
- The app does not connect to a CRM, ticketing platform, email account, or customer database.
- It does not train, fine-tune, or deploy a model as a hosted service.
- Avoid real customer data, personal information, payment details, credentials, and confidential company information.
- Model inference runs on the machine running Streamlit after the model is downloaded; the initial download comes from Hugging Face Hub.

## Suggested next improvements

1. Add a confidence / review-needed indicator based on rule checks (not a claim of model confidence).
2. Add batch processing with progress tracking.
3. Add a feedback column so a reviewer can correct category and priority.
4. Evaluate results against a small, manually labelled test set before making any performance claims.
5. Add a proper privacy, security, and access-control design before considering real business use.

## Resume integrity

Only list features you have personally run and verified. While building, use **Independent Project — In Progress** only if you have actually started implementing it. Do not claim model fine-tuning, production deployment, autonomous agents, or measurable business impact unless you have completed and evidenced those activities.
