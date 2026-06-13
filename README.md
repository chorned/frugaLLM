# **🧠 Hermes Router Plugin**

A zero-dependency, cost-optimized LLM router proxy for [Hermes Agent](https://hermes-agent.nousresearch.com).

This plugin acts as a smart middleman between Hermes and OpenRouter. It automatically discovers the **best free models** available right now and routes your everyday requests to them, only escalating to a paid model when you explicitly tell it to.

## **💸 The $11 Difference: Why use this?**

If you load $11 into your OpenRouter account and point Hermes directly at a premium model, every single interaction costs you money. Trivial tasks like "fix this typo," "summarize this log," or "format this text" will eat away at your balance just as fast as complex coding problems.

**The Hermes Router changes the game:**

* **Without the router:** Your $11 drains constantly. You hesitate to use Hermes for small tasks because of the micro-transactions.  
* **With the router:** 95% of your daily requests hit high-quality, dynamically discovered *free* models (Tier 1). You pay **$0.00** for everyday tasks. That $11 balance is saved exclusively for the 5% of tasks that actually require heavy-duty reasoning (Tier 3). Your $11 can now last you months instead of days, and you never have to hesitate to ask Hermes a question.

## **✨ Set It and Forget It**

The best part? You don't have to manage model lists or hunt for promotions. At startup, the router automatically scans OpenRouter's /models endpoint and seamlessly updates its roster with the best free models available *that day*, sorted by context window.

You just chat with Hermes like normal. The routing happens entirely in the background.

## **🚦 How It Works: The 3-Tier System**

The router runs locally on localhost:5050. Hermes talks to it exactly like a standard API endpoint, and the router decides where the request goes based on a simple escalation path:

| Tier | Trigger | Model | Cost |
| :---- | :---- | :---- | :---- |
| **Tier 1: Daily Driver** | Every auto request | Best free general model (e.g., openrouter/owl-alpha) | **$0.00** |
| **Tier 2: Deep Thinker** | Explicit model: reasoning\_free | Best free reasoning model (e.g., deepseek/deepseek-r1) | **$0.00** |
| **Tier 3: The Big Guns** | Typing //escalate, 3 consecutive failures, or context overflow | Your configured premium model (e.g., gemini-3.1-pro-preview) | Paid |

## **🛠️ Requirements**

* **macOS** (for the launchd background service — Linux users can easily adapt this to systemd)  
* A Hermes Agent installation at \~/.hermes/  
* Python 3.10+ with requests installed in your Hermes virtual environment  
* An OpenRouter API key (a free-tier account works perfectly\!)

## **🚀 Easy 5-Step Installation**

### **1\. Copy the Script**

Move the router script into your Hermes skills folder:

cp router\_server.py \~/.hermes/skills/router\_server.py

### **2\. Add Your Key**

Add your OpenRouter API key to your \~/.hermes/.env file:

OPENROUTER\_API\_KEY=sk-or-your-key-here

### **3\. Point Hermes to the Router**

Update your \~/.hermes/config.yaml to tell Hermes to use your new local proxy. This ensures you only pay for Tier 3 when you specifically ask for it:

routing:  
  allow\_pro: false   \# Only escalate if you type //escalate or on repeated failures

model:  
  api\_key: na  
  base\_url: http://localhost:5050/v1  
  default: auto  
  provider: custom

custom\_providers:  
\- api\_key: na  
  base\_url: http://localhost:5050/v1  
  model: auto  
  name: HermesRouter

### **4\. Install as a Background Service (macOS)**

To make sure the router is always running without you having to think about it, we'll set it up as a launch agent.

First, edit com.hermes.router-proxy.plist to match your specific username:

\<string\>/Users/YOUR\_USERNAME/.hermes/hermes-agent/venv/bin/python3\</string\>  
\<string\>/Users/YOUR\_USERNAME/.hermes/skills/router\_server.py\</string\>

Then, load the service:

mkdir \-p \~/Library/LaunchAgents  
cp com.hermes.router-proxy.plist \~/Library/LaunchAgents/  
launchctl load \~/Library/LaunchAgents/com.hermes.router-proxy.plist

*The proxy will now auto-start on login and restart automatically if it ever crashes.*

### **5\. Verify It's Running**

Run these quick checks to ensure everything is wired up:

\# Check the service is alive in macOS  
launchctl list | grep com.hermes.router-proxy

\# Check the router's health endpoint  
curl \-s http://localhost:5050/health | python3 \-m json.tool

\# Watch the live logs to see the routing in action  
tail \-f \~/.hermes/logs/router\_proxy.log

## **💬 Usage**

Once installed, there's nothing else you need to do\! Hermes will route your requests to Tier 1 automatically.

If you want to manually force a specific tier during a conversation, you can use these model labels:

| If you want... | Route to... |
| :---- | :---- |
| **Normal tasks** | auto or balanced\_free (Tier 1\) |
| **Complex logic** | reasoning\_free (Tier 2\) |
| **Local privacy** | local (Routes to local Ollama via hermes:latest) |
| **Max power** | pro or type //escalate in your chat message (Tier 3 paid model) |

## **🎛️ Admin Commands**

Want to manually poke the router? You can use these local endpoints:

\# Force the router to re-scan OpenRouter for new free models right now  
curl \-X POST http://localhost:5050/admin/refresh-models

\# Reset the consecutive failure safety counter  
curl \-X POST http://localhost:5050/admin/reset-failures

\# View current router status and active models  
curl http://localhost:5050/health

## **📋 Logs**

Keep an eye on what the router is doing (and how much time it's taking) in your logs:

\~/.hermes/logs/router\_proxy.log   \# Routing decisions and speed metrics  
\~/.hermes/logs/router\_proxy.err   \# Error output

**Example output:**

10:27:11 \[Router\] ── Request: model=auto, msgs=2, stream=True  
10:27:11 \[Router\] ☁️  auto → openrouter/owl-alpha \[Tier 1: default\]  
10:27:11 \[Router\]    Resolved → openrouter/owl-alpha @ OPENROUTER (1.0ms routing)  
10:27:29 \[Router\]    ✓ Done. Proxy overhead: 1.0ms | Upstream: 18040ms (18.0s)

## **🗑️ Uninstall**

If you ever need to remove the background service:

launchctl unload \~/Library/LaunchAgents/com.hermes.router-proxy.plist  
rm \~/Library/LaunchAgents/com.hermes.router-proxy.plist  