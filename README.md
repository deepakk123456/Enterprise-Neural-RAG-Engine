# ⚡ Distributed Neural-RAG Cluster Engine

A production-ready, highly modular **Retrieval-Augmented Generation (RAG)** pipeline designed with a microservices-inspired architecture. This framework mitigates standard RAG vulnerabilities through real-time security guardrails, optimizes document alignment via Cross-Encoder reranking models, and monitors cluster transactional health with an enterprise-grade telemetry dashboard.

---

## 🏗️ System Architecture & Core Modules

The engine splits monolithic RAG pipelines into detached, single-responsibility operational components:

* **`app.py` (Distributed Gateway):** Built on FastAPI; handles core API routing, document orchestration, vector space synchronization, and LLM communication loops.
* **`ui.py` (Telemetry Control Hub):** An advanced Streamlit visual node styled with a high-fidelity dark-mode UI. It features real-time area gradient plots map tracking mathematical decay curves during retrieval reranking.
* **`guardrails.py` (Security Firewall Matrix):** An inbound transaction filter that intercepts adversarial prompt injections, jailbreaks, and unauthorized database instructions (`drop table`, etc.) before vector matching.
* **`memory_manager.py` (Session Persistence Layer):** Uses an isolated SQLite connection framework to maintain multi-turn context boundaries, allowing seamless conversational memory without state bloat.
* **`evaluator.py` (RAG Precision Metrics):** Dynamically computes mathematical verification scales (Faithfulness Matrix & Context Recall alignment) for each synthesized token chunk.
* **`feedback_logger.py` (Audit Log System):** Records telemetry signals and alignment data into centralized JSON arrays for continuous evaluation models.

---

## 🛠️ Computational Overhead & Complexity Layout

The core architecture leverages the **Transformer attention mechanism** for retrieval context processing. The underlying mathematical matrices maintain the following benchmark characteristics:

| Core Component | Target Mathematical Operation | Computational Overhead |
| :--- | :--- | :--- |
| **Self-Attention** | $QK^T$ (Query-Key Dot Product) | $O(\text{sequence\_length}^2 \cdot d_{\text{model}})$ |
| **Self-Attention** | $V \cdot \text{softmax}(QK^T)$ | $O(\text{sequence\_length} \cdot d_{\text{model}}^2)$ |
| **Feed-Forward Layers** | Linear Projection / Dense Layers | $O(\text{sequence\_length} \cdot d_{\text{model}}^2)$ |

---

## 🚀 Deployment Operations & Runtime Protocol

### 1. Initialize Backend Environment (Uvicorn Service Gateway)
```bash
python -m uvicorn app:app --reload
2. Initialize Frontend Interface (Streamlit Telemetry Dashboard)
Apne dusre terminal me jao aur live cluster control board graphical panel chalu karne ke liye yeh command run karo:

Bash
python -m streamlit run ui.py
Yeh automatic aapke default web browser me dashboard dynamic access panel http://localhost:8501 launch kar dega.
