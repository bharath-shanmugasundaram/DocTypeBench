# 1. Comparative Analysis of Models for Document Type Classification

The objective of this analysis is to design an Ai-based document type detection system that classifies scanned document pages into predefined document categories such as Invoice, Resume, Bank Statement, Insurance/Policy Document, and Government Forms.

The classification is primarily based on visual layout, structural patterns, and document organisation, rather than relying solely on textual content.

---

# 2. Why Multiple Model Families Are Evaluated ?

Document Type Detection for Scanned  Documents is a non-trivial classification problem due to the high variability in document appearance, structure, and content. Although the end goal is a single classification label per document pages, the signals required to make that decision differ significant across document types. 

## 2.1 Variability in Document Quality and Structure 

Scanned document often exhibits wide variations in:
-  Resolution and quality 
- Skew noise and partial occlusion 
- Presence or absence of clear visual boundaries 

For example, invoices and resumes often have distinct visual layouts, while bank statements and government forms may appear visually similar across different issuers. This variability makes it important to evaluate models with different inductive biases toward layout, texture, and structure.

## 2.2 Trade-off Between Visual Layout and Textual Semantics

Not all document types rely on the same information cues:
- Visually distinctive documents (e.g., resumes, invoices) can often be identified primarily through layout patterns such as headers, spacing, tables, and alignment.
- Semantically driven documents (e.g., insurance policies, government forms) may require understanding specific textual content or key phrases to differentiate between document categories.

As a result, both vision-only models and text-aware models need to be evaluated to understand where each approach performs well or fails.

