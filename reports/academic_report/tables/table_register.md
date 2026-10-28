# Table Register: Complete Inventory of Tabular Artifacts

This register indexes every table generated for the **Maritime Operations AI** project report. In adherence to academic report guidelines, all table captions are placed above the respective table using the format `Table X.Y: Table Title`. Separate CSV versions are provided for every table to facilitate direct insertion into Word templates.

---

| Table Number | Table Title / Caption | CSV File Path | Markdown Section Reference | Source Data or Code | Intended Chapter & Section | Representation Type |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: |
| **Table 3.1** | Table 3.1: Hardware Infrastructure Requirements | `tables/table3_1_hardware_requirements.csv` | `system_design/chapter3_system_design.md` | System environment audit | Chapter 3, Section 3.1.1 | Illustrative / Specification |
| **Table 3.2** | Table 3.2: Software Environment, Frameworks, and Package Dependencies | `tables/table3_2_software_requirements.csv` | `system_design/chapter3_system_design.md` | `pyproject.toml`, `package.json`, Dockerfiles | Chapter 3, Section 3.1.2 | Measured / Inventory |
| **Table 3.3** | Table 3.3: Core Project Modules and Architectural Responsibilities | `tables/table3_3_project_modules.csv` | `system_design/chapter3_system_design.md` | Codebase architecture audit | Chapter 3, Section 3.5 | Illustrative / Architecture |
| **Table 3.4** | Table 3.4: REST API Endpoints Specification | `tables/table3_4_api_endpoints.csv` | `system_design/chapter3_system_design.md` | `backend/app/api/v1/` routes | Chapter 3, Section 3.6 | Measured / API Contract |
| **Table 4.1** | Table 4.1: SAR Ship Detection Dataset Statistics | `tables/table4_1_dataset_statistics.csv` | `implementation/chapter4_implementation_testing.md` | LS-SSDD & OpenSAR dataset manifests | Chapter 4, Section 4.1 | Measured (Dataset Counts) |
| **Table 4.2** | Table 4.2: Model Training Hyperparameters and Execution Environments | `tables/table4_2_training_configurations.csv` | `implementation/chapter4_implementation_testing.md` | Training logs, `args.yaml`, metadata JSON | Chapter 4, Section 4.1 | Measured (Hyperparameters) |
| **Table 4.3** | Table 4.3: Quantitative Model Performance Comparison Matrix | `tables/table4_3_model_performance_comparison.csv` | `implementation/chapter4_implementation_testing.md` | Evaluated test metrics from LS-SSDD & OpenSAR | Chapter 4, Section 4.5.1 | Measured (Held-Out Test Splits) |
| **Table 4.4** | Table 4.4: Automated Test Cases and Verification Outcomes | `tables/table4_4_test_cases_results.csv` | `implementation/chapter4_implementation_testing.md` | Container Pytest execution logs (109 tests) | Chapter 4, Section 4.4 | Measured (Test Execution) |
| **Table 4.5** | Table 4.5: Existing Maritime Surveillance Systems vs Proposed System | `tables/table4_5_existing_vs_proposed_system.csv` | `implementation/chapter4_implementation_testing.md` | Comparative capability analysis | Chapter 4, Section 4.7 | Illustrative / Comparison |
| **Table 4.6** | Table 4.6: Research Literature Comparison Matrix | `tables/table4_6_research_paper_comparison.csv` | `literature/chapter2_literature_survey.md` | Published research papers & SOTA baselines | Chapter 2, Section 2.4 | Derived from Literature |
| **Table 4.7** | Table 4.7: Operational Compatibility Audit on Real Sentinel-1 GRD Imagery | Inline in `dataset_and_model_experiments.md` | `experiments/dataset_and_model_experiments.md` | `backend/ml/grd_compatibility_test.py` | Chapter 4, Section 4.5.3 | Measured (Operational Swath) |
| **Table 4.8** | Table 4.8: Master Model Benchmark Comparison Matrix | Inline in `dataset_and_model_experiments.md` | `experiments/dataset_and_model_experiments.md` | Comprehensive evaluation summary | Chapter 4, Section 4.5 | Measured / Summary |
| **Table 4.9** | Table 4.9: Coordinate Convention and Axis-Order Standards Across Subsystems | Inline in `georeferencing_and_ais_correlation.md` | `experiments/georeferencing_and_ais_correlation.md` | `coordinates.py` implementation invariants | Chapter 3, Section 3.5.4 | Illustrative / Mathematical Standard |
