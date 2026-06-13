# **🧠 frugaLLM Proxy**

**A zero-dependency, cost-optimized LLM router proxy for ANY app that supports custom OpenAI endpoints.**

Whether you're using AI coding assistants like **Cursor**, **Cline**, or **Continue.dev**, or chat interfaces like **AnythingLLM**, **Obsidian plugins**, or **Hermes Agent**, frugaLLM acts as a smart middleman between your app and OpenRouter.

It automatically discovers the **best free models** available right now and routes your everyday requests to them, only escalating to a paid premium model when you explicitly tell it to.

## **💸 The $11 Difference: Why use this?**

If you load $11 into your OpenRouter account and point your favorite AI tool directly at a premium model, every single interaction costs you money. Trivial tasks like "fix this typo," "write a git commit," or "format this JSON" will eat away at your balance just as fast as complex coding architecture problems.

**frugaLLM changes the game:**

* **Without the router:** Your $11 drains constantly. You hesitate to ask your AI small questions because of the micro-transactions.  
* **With the router:** 95% of your daily requests hit high-quality, dynamically discovered *free* models. You pay **$0.00** for everyday tasks. That $11 balance is saved exclusively for the 5% of tasks that actually require heavy-duty reasoning. Your $11 can now last you months instead of days.

## **🗺️ How It Works: The Architecture**

The router runs locally on your machine and pretends to be a standard OpenAI-compatible API. When your app sends a prompt, frugaLLM intercepts it and decides where it should *actually* go based on a simple 3-tier escalation path mapped to your agent types.

                     \[Your Favorite AI App\]  
                  (Cursor, Cline, AnythingLLM)  
                               │  
                               ▼ (http://localhost:5050/v1)  
                ╔══════════════════════════════╗  
                ║       🧠 frugaLLM Proxy      ║  
                ║    (Dynamic Model Router)    ║  
                ╚══════════════════════════════╝  
                 /             │             \\  
           "balanced"      "reasoning"    "pro" or //escalate  
        (Default Agents) (Expert Agents)    (Escalation)  
               │               │               │  
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  
        │ OpenRouter  │ │ OpenRouter  │ │ OpenRouter  │  
        │ (Free Tier) │ │ (Free Tier) │ │ (Paid Tier) │  
        │  owl-alpha  │ │ deepseek-r1 │ │ gemini-pro  │  
        └─────────────┘ └─────────────┘ └─────────────┘  
            $0.00           $0.00           $$$$$

### **🚦 The 3-Tier System**

| Tier | Trigger | Model | Cost |
| :---- | :---- | :---- | :---- |
| **Tier 1: Balanced** | Every auto request (Default Agents) | Best free general model available today | **$0.00** |
| **Tier 2: Reasoning** | Explicit model: reasoning\_free (Expert Agents) | Best free reasoning model available today | **$0.00** |
| **Tier 3: The Big Guns** | Typing //escalate in the prompt, repeated failures, or massive context | Your configured premium model | Paid |

**Set It and Forget It:** At startup, frugaLLM automatically scans OpenRouter's /models endpoint and seamlessly updates its roster with the best free models available *that day*, sorted by context window. You never have to hunt for promotions or manually update model lists.

## **🛠️ Requirements**

* Python 3.10+ (with the requests library installed)  
* An OpenRouter API key (a free-tier account works perfectly\!)

## **🚀 Easy Installation**

### **1\. Download & Configure**

Create a folder for the router anywhere on your machine (e.g., \~/frugaLLM), and place router\_server.py inside it.

Create a .env file in the same folder and add your key:

OPENROUTER\_API\_KEY=sk-or-your-key-here

### **2\. Point Your App to the Router**

Go into the settings of Cursor, Cline, AnythingLLM, or whatever app you use, and set up a custom model provider:

* **Base URL / API URL:** http://localhost:5050/v1  
* **API Key:** na (or anything, frugaLLM handles your real key securely)  
* **Model Name:** auto

### **3\. Run It as a Background Service (The AI Way 🤖)**

To make sure frugaLLM is always running without you having to keep a terminal window open, you'll want to run it as a background service.

**Don't know how to do that? Ask an AI\!**

Simply copy the path to your router\_server.py file, open ChatGPT, Claude, or Gemini, and paste this prompt:

*"I am on \[macOS / Windows / Ubuntu\]. Please write me a \[launchd plist / systemd service / Windows Task Scheduler script\] that runs a Python script located at /path/to/your/frugaLLM/router\_server.py continuously in the background. Tell me exactly where to save it and the terminal commands to start it."*

Follow the AI's instructions, and you'll have a rock-solid background service running in two minutes.

## **💬 Usage**

Once installed, just use your app normally\! Set your app's model to auto and frugaLLM will handle the rest.

If you want to manually force a specific tier during a chat or coding session, you can use these model names in your app's dropdown:

| If you want... | Set your app's model to... |
| :---- | :---- |
| **Normal tasks** | auto or balanced\_free (Tier 1\) |
| **Complex logic** | reasoning\_free (Tier 2\) |
| **Max power** | pro or simply type //escalate anywhere in your prompt |

💡 **Pro-Tip: Local LLMs**

While frugaLLM dominates OpenRouter savings, it also fully supports routing to local instances via the local tag. If you have the hardware, downloading an app like Ollama is an incredibly easy, low-effort/high-reward way to add completely private, offline, 100% free routing to your stack\!

## **🎛️ Admin Commands**

Want to manually poke the router while it's running in the background? Run these in your terminal:

\# Force the router to re-scan OpenRouter for new free models right now  
curl \-X POST http://localhost:5050/admin/refresh-models

\# View current router status and active model routing table  
curl \-s http://localhost:5050/health | python3 \-m json.tool

### **⏱️ Auto-Refresh Free Models (Zero Downtime)**

Because the OpenRouter free-tier directory changes constantly, you can keep frugaLLM perfectly up to date without *ever* restarting the app. Just set up a simple cron job to ping the refresh endpoint every 12 hours\!

Open your crontab (crontab \-e) and add this line to silently refresh the roster every day at midnight and noon:

0 0,12 \* \* \* curl \-X POST http://localhost:5050/admin/refresh-models \>/dev/null 2\>&1  
