# Reader Agent

An AI-powered Git Project Manager that reads Writer memory from Firebase and answers questions about repositories, PRs, developers, risks, and code evidence.

The Reader connects to a GitHub repository or organization, loads the project memory from Firebase, and deep-dives into the cloned code when Firebase memory is not enough.

## Setup

### 1. Requirements

* Python 3.10+
* Git installed and available in your terminal
* A Firebase service account key
* One LLM API key: Gemini or OpenAI
* Optional: a GitHub Personal Access Token for private repos or higher GitHub API limits

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure Firebase

Download your Firebase service account key from Firebase Console:

```text
Project Settings → Service accounts → Generate new private key
```

Place it in:

```text
secrets/firebase-service-account.json
```

Do not commit this file.

### 3. Configure `.env`

Create a `.env` file in the Reader root:

```env
# Choose one LLM provider
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash

# Optional, for private repos / higher GitHub rate limits
GH_PAT=your_github_token
```

If using OpenAI instead:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
```

## Run the Reader

Start the Streamlit app:

```bash
streamlit run app.py
```

Then enter either:

```text
https://github.com/owner/repo
```

or a full organization/user URL:

```text
https://github.com/owner
```

For an organization URL, the Reader lists and clones the accessible repositories under that owner.

## How It Works

1. **Firebase Memory:** Reader loads the full Firebase memory produced by Writer.
2. **Repository Scoping:** Reader scopes Firebase memory to the currently connected GitHub repo(s), unless the user asks for a global Firebase overview.
3. **Code Deep-Dive:** If Firebase memory is not enough, Reader performs a read-only investigation of the connected repo(s).
4. **Answer Generation:** Reader combines Firebase memory and repo evidence into a PM / QA / developer answer.

## Features

* **Firebase-first memory:** Uses Writer's stored project intelligence as the main source of truth.
* **Repo and org support:** Accepts both single GitHub repos and full GitHub organizations/users.
* **Scoped memory:** Prevents unrelated Firebase data from leaking into answers about another repo.
* **Read-only code investigation:** Can inspect repo structure, files, line ranges, symbols, grep hits, commit history, and relevant evidence.
* **Disconnect cleanup:** Disconnecting clears local repo cache and session state to prevent stale repo usage.
* **LLM additional info:** Can save useful Reader-discovered insights back to Firebase under a separate Reader namespace.

## Example Questions

```text
What information exists in Firebase right now? List projects, developers, and PRs.
```

```text
Summarize PR 11 for a project manager. What changed, which files were touched, and what is the risk level?
```

```text
Using Firebase and the connected GitHub repos, verify whether PR 11 really changed scripts/agent_action.py and scripts/github_service.py.
```

```text
What should QA test based on this PR?
```

## Local Development

Useful quick checks:

```bash
python dump_firebase_structure.py
```

```bash
python test_llm.py
```

```bash
streamlit run app.py
```

## Notes

* The Reader does not use local mock JSON memory anymore.
* Firebase memory is provided by the Writer Agent.
* A local Git clone can show files and commits, but PR data only comes from Firebase unless GitHub PR API support is added.
* Never commit `.env` or `secrets/firebase-service-account.json`.
