# scientific\_agent

**Scientific Agent: A Retrieval-Augmented Generation (RAG) System for Domain-Aware Literature Review Automation**

---

## Overview

**scientific\_agent** is an AI-powered literature review system that combines the power of **Retrieval-Augmented Generation (RAG)** with structured metadata from scientific databases. It enables the generation of **accurate, context-rich, and citation-ready** literature reviews across various research domains including—but not limited to—biomedicine, computer science, engineering, social sciences, and environmental studies.

The system leverages modern language models, real-time knowledge retrieval, and multi-source citation formatting to support scalable, credible, and multilingual scholarly writing.

---

## Key Features

* 🔍 **Domain-Agnostic RAG Pipeline**
  Capable of generating literature reviews on any topic where academic publications are available—across disciplines.

* 📚 **Multi-Source Retrieval**
  Retrieves metadata and abstracts from **Crossref**, with optional support for **Scopus** and **Web of Science**, ensuring comprehensive coverage.

* 💡 **Knowledge-Grounded Generation**
  Uses OpenAI embeddings and FAISS vector search to ground language model outputs in retrieved, relevant documents.

* 🌐 **REST API Access**
  Offers a FastAPI-based endpoint for integrating RAG-based review generation into research tools or academic workflows.

* 🧾 **Flexible Citation Formats**
  Outputs citations in **BibTeX**, **APA 7**, or raw key/DOI formats based on user preference.

* 🌍 **Multilingual Support**
  Literature reviews can be generated in multiple languages by specifying the target language.

---

## Use Cases

* 📖 Academic Literature Reviews (e.g., AI ethics, climate change, neural network architectures)
* 🧬 Biomedical Research Summaries
* ⚖️ Legal and Policy Analysis with Reference Tracking
* 📊 Technical Surveys in Engineering and Data Science
* 🧠 Scientific Knowledge Gap Identification

---

## Installation & Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your API credentials:

```dotenv
OPENAI_API_KEY=your_openai_key
CROSSREF_MAILTO=your_email@example.com
# Optional:
ELSEVIER_API_KEY=...
WOS_API_KEY=...
```

---

## Usage

### CLI Mode

```bash
python scientific_rag_agent.py
```

You will be prompted for a topic. The resulting `literature_review.md` will be created in the current directory.

### API Mode

Start the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 7001
```

Then send a POST request to:

```
POST /literature-review
```

#### Example Request

```json
{
  "topic": "The impact of generative AI on educational assessment",
  "citation_format": "apa7",
  "language": "English"
}
```

---

## System Architecture

* **Retrievers**: Modular design for metadata extraction from Crossref, Scopus, and Web of Science.
* **Embedding + Retrieval**: Uses OpenAI embeddings and FAISS for nearest-neighbor search.
* **LLM Backend**: `gpt-4o` and `gpt-4o-mini` models for natural language generation.
* **Citation Formatter**: Styles include raw, BibTeX, and APA 7.
* **Prompt Templates**: Designed for domain-aware, citation-rich synthesis.

---

## Limitations

* **Retrieval Dependency**: Quality depends on the abstracts and metadata available from external APIs.
* **Citation Bias**: May inherit biases from underlying datasets or retrieval algorithms.
* **Factual Constraints**: Only generates from retrieved content; does not fabricate knowledge.

---

## Future Work

* 📑 PDF ingestion and multimodal source support
* 📌 Full-text summarization (beyond abstracts)
* 🧠 Continual learning for evolving knowledge bases
* 🧮 Improved metrics for coherence, factual grounding, and citation coverage

---

## License

Licensed under the [MIT License](LICENSE)
© 2025 Emirhan Soylu

---

## References

Literature used for model synthesis and benchmarking is cited in BibTeX format in the output files. For a conceptual background, see:

```bibtex
@article{Genesis2025,
  title={Large Language Models (LLMs)...},
  author={Genesis},
  year={2025},
  doi={10.20944/preprints202504.0443.v1}
}
@article{Li2024,
  title={Biomedrag: A Retrieval Augmented Large Language Model for Biomedicine},
  author={Li},
  year={2024},
  doi={10.2139/ssrn.4910081}
}
@article{Huang2024,
  title={Adapting LLMs for Biomedicine through Retrieval-Augmented Generation},
  author={Huang},
  year={2024},
  doi={10.1109/bibm62325.2024.10822725}
}
@article{Singh2024,
  title={A Multimodal Framework for Quantifying RAG Efficacy},
  author={Singh},
  year={2024},
  doi={10.36227/techrxiv.173152556.61823435/v1}
}
```

---

## Contributing

Pull requests and feature suggestions are welcome, particularly around new retriever integrations, multilingual capabilities, or review evaluation methods.
