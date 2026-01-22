# 1. Comparative Analysis of Models for Document Type Classification

The objective of this analysis is to design an Ai-based document type detection system that classifies scanned document pages into predefined document categories such as Invoice, Resume, Bank Statement, Insurance/Policy Document, and Government Forms.

The classification is primarily based on visual layout, structural patterns, and document organisation, rather than relying solely on textual content.

---


## 2. Why Multiple Model Families Are Evaluated

Document Type Detection for scanned documents is a **non-trivial classification problem** due to the high variability in document appearance, structure, and content. Although the end goal is a single classification label per document page, the **signals required to make that decision differ significantly across document types**.

### 2.1 Variability in Document Quality and Structure

Scanned documents often exhibit wide variations in:

* Resolution and scan quality
* Skew, noise, and partial occlusion
* Presence or absence of clear visual boundaries

For example, invoices and resumes often have **distinct visual layouts**, while bank statements and government forms may appear visually similar across different issuers. This variability makes it important to evaluate models with **different inductive biases** toward layout, texture, and structure.

---

### 2.2 Trade-off Between Visual Layout and Textual Semantics

Not all document types rely on the same information cues:

* **Visually distinctive documents** (e.g., resumes, invoices) can often be identified primarily through layout patterns such as headers, spacing, tables, and alignment.
* **Semantically driven documents** (e.g., insurance policies, government forms) may require understanding specific textual content or key phrases to differentiate between document categories.

As a result, both **vision-only models** and **text-aware models** need to be evaluated to understand where each approach performs well or fails.

---

### 2.3 OCR Dependency and Its Implications

OCR-based approaches introduce an additional dependency in the pipeline:

* OCR accuracy is sensitive to scan quality and language
* OCR adds computational cost and latency
* Errors at the OCR stage can propagate downstream and affect classification performance

Given these constraints, it is important to compare **OCR-free vision models** against **OCR-dependent multimodal models** to assess whether the added complexity results in measurable performance gains for this task.

---

### 2.4 Architectural Diversity and Inductive Biases

Each model family brings a fundamentally different architectural assumption:

* **Swin Transformer** emphasizes hierarchical visual structure and local-global spatial relationships.
* **State-of-the-art Vision Transformers** provide highly expressive visual embeddings learned from large-scale pretraining.
* **LayoutLM-family models** jointly encode text content and spatial layout.
* **PaddleOCR-based pipelines** follow an OCR-first strategy with modular downstream processing.

Since document type detection can be framed as either a **visual layout problem**, a **multimodal understanding problem**, or a **pipeline-based enterprise OCR task**, evaluating multiple model families ensures that architectural limitations do not bias the final system design.

---

### 2.5 Absence of a Universally Optimal Architecture

There is no single model architecture that is universally optimal across:

* All document types
* All scan qualities
* All production constraints (latency, cost, scalability)

Therefore, a comparative evaluation across Swin Transformer, SOTA Vision Transformers, LayoutLM, and PaddleOCR-based approaches is necessary to:

* Identify trade-offs
* Understand failure modes
* Support an informed and defensible architectural decision
