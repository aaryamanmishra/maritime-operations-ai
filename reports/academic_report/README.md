# Maritime Operations AI: Academic Report & Synopsis Artifact Repository

> **Academic Institution:** Bansilal Ramnath Agarwal Charitable Trust's, Vishwakarma Institute of Technology, Pune-37  
> **Department:** Department of Computer Science and Engineering (Artificial Intelligence)  
> **Course Name & Code:** Deep Learning (CI3201)  
> **Academic Year:** 2025–2026  
> **Internal Guide:** Dr. Nilesh B. Korade  
> **Head of Department:** Dr. Nilesh P. Sable  
> **Project Title:** Maritime Operations AI: Spaceborne Synthetic Aperture Radar (SAR) Vessel Detection and Kinematic AIS Dark-Vessel Intelligence System  

---

## 1. Overview of the Artifact Repository

This repository contains the complete collection of academic project artifacts prepared to populate the supplied **`Index.docx`** and **`Content.docx`** templates. 

In strict adherence to project instructions:
- **No Word documents have been overwritten or created.** All artifacts are structured as ready-to-copy Markdown (`.md`), raw CSV tables (`.csv`), editable Mermaid diagram source files (`.mmd`), vector graphics (`.svg`), and high-resolution raster images (`.png`, `.jpg`).
- **Zero Hallucinated Metrics:** All numerical figures, detection accuracies, inference latencies, and georeferencing gates are extracted from empirical test runs, dataset manifests, and container execution logs.
- **Academic Placeholders:** Group numbers, student names, roll numbers, G.R. numbers, and contact details are clearly marked with `[PLACEHOLDER]` tags for manual insertion.

---

## 2. Directory Structure & Report Section Mapping

```
reports/academic_report/
│
├── README.md                                  <- Master guide and chapter cross-reference (This File)
├── artifact_index.csv                         <- Tabular index of all 42 generated project artifacts
│
├── synopsis/
│   ├── synopsis_content.md                    <- Complete Project Synopsis (Front Matter, Abstract, 8 Sections)
│   └── chapter1_introduction.md               <- Chapter 1: Introduction (Overview, Problem, Objectives, Scope)
│
├── literature/
│   └── chapter2_literature_survey.md          <- Chapter 2: Literature Survey (CFAR, S-1 Modes, LS-SSDD, SOTA Baselines)
│
├── system_design/
│   └── chapter3_system_design.md              <- Chapter 3: System Analysis & Design (Requirements, Architecture, DB, Modules)
│
├── implementation/
│   ├── chapter4_implementation_testing.md     <- Chapter 4: Implementation & Testing (Algorithms, Tests, Benchmarks)
│   ├── chapter5_conclusion_future_scope.md    <- Chapter 5: Conclusion & Future Scope (Contributions, Limitations, Future)
│   └── tools_and_environment.md               <- Tools, Environment & Reproducibility Specification
│
├── experiments/
│   ├── dataset_and_model_experiments.md       <- Forensic Audit of All 4 Model Experiments & Swath Compatibility
│   └── georeferencing_and_ais_correlation.md  <- GCP Mathematical Derivations & Kinematic Uncertainty Budget
│
├── figures/                                   <- System Diagrams (.svg and .mmd source) & Empirical Charts
│   ├── figure_register.md                     <- Master register of all visual figures (Captions & Mappings)
│   ├── fig3_1_overall_system_architecture.svg / .mmd
│   ├── fig3_2_end_to_end_data_flow.svg / .mmd
│   ├── fig3_3_sar_acquisition_preprocessing_flowchart.svg / .mmd
│   ├── fig3_4_native_resolution_chip_extraction.svg / .mmd
│   ├── fig3_5_ship_detection_inference_pipeline.svg / .mmd
│   ├── fig3_6_pixel_to_geo_coordinate_transformation_flowchart.svg / .mmd
│   ├── fig3_7_ais_ingestion_storage_architecture.svg / .mmd
│   ├── fig3_8_sar_ais_temporal_spatial_matching_flowchart.svg / .mmd
│   ├── fig3_9_dark_vessel_candidate_classification.svg / .mmd
│   ├── fig3_10_database_er_diagram.svg / .mmd
│   ├── fig3_11_uml_use_case_diagram.svg / .mmd
│   ├── fig3_12_backend_frontend_interaction.svg / .mmd
│   ├── fig4_3_model_performance_comparison.svg / .png
│   ├── fig4_4_inshore_vs_offshore_clutter.svg / .png
│   ├── fig4_5_grd_slc_domain_compatibility.svg / .png
│   ├── fig4_6_uncertainty_budget_breakdown.svg / .png
│   └── [OpenSAR Evaluation Curves: BoxPR_curve.png, confusion_matrix_normalized.png, BoxF1_curve.png]
│
├── tables/                                    <- CSV Data Tables & Master Table Register
│   ├── table_register.md                      <- Master register of all tables (Captions & References)
│   ├── table3_1_hardware_requirements.csv
│   ├── table3_2_software_requirements.csv
│   ├── table3_3_project_modules.csv
│   ├── table3_4_api_endpoints.csv
│   ├── table4_1_dataset_statistics.csv
│   ├── table4_2_training_configurations.csv
│   ├── table4_3_model_performance_comparison.csv
│   ├── table4_4_test_cases_results.csv
│   ├── table4_5_existing_vs_proposed_system.csv
│   └── table4_6_research_paper_comparison.csv
│
├── screenshots/                               <- Live Application Screenshots & Visual Radar Overlays
│   ├── screenshot_main_dashboard.png          <- Live React / MapLibre GL Dashboard Screenshot
│   ├── sentinel1_real_scene_fused.png         <- Real Sentinel-1C English Channel Radar Swath
│   ├── lsssd_training_train_batch0.jpg        <- Native-Resolution LS-SSDD Radar Training Chips
│   └── opensar_baseline_val_batch0_pred.jpg   <- Deep-Learning Detection Overlays on Test Batches
│
└── references/
    ├── references.md                          <- IEEE Bibliography (17 Real Academic Papers & Standards)
    └── dataset_and_software_sources.md        <- Official Dataset Repositories & Software Licenses
```

---

## 3. Template-to-Artifact Cross-Reference Guide

The table below instructs exactly which file to open when populating each section of the Word templates:

### 3.1 Populating `Index.docx` (Project Synopsis & Preliminaries)

| Section in `Index.docx` | Corresponding Generated Artifact | Required Manual Student Action |
| :--- | :--- | :--- |
| **Title Page & Student Table** | `synopsis/synopsis_content.md` | Fill in Group No, Student Names, Roll Nos, G.R. Nos, Contacts, Emails. |
| **Acknowledgement** | `synopsis/synopsis_content.md` (Preliminaries) | Ready to copy. Guide: Dr. Nilesh B. Korade; HOD: Dr. Nilesh P. Sable. |
| **Abstract** | `synopsis/synopsis_content.md` (Abstract) | Ready to copy. Summarizes problem, methodology, and $80.7\%$ precision. |
| **Table of Contents** | `Index.docx` (Template page iv) | Update page numbers based on final pagination. |
| **List of Figures** | `figures/figure_register.md` | Copy figure numbers and captions (`Fig X.Y: ...`). |
| **List of Tables** | `tables/table_register.md` | Copy table numbers and captions (`Table X.Y: ...`). |

### 3.2 Populating `Content.docx` (Full Final Project Report)

| Chapter & Section in `Content.docx` | Generated Content File | Recommended Figures & Tables to Embed |
| :--- | :--- | :--- |
| **Chapter 1: Introduction**<br/>• 1.1 Overview of Project<br/>• 1.2 Problem Statement<br/>• 1.3 Objective of Thesis<br/>• 1.4 Scope of the Project | `synopsis/chapter1_introduction.md` | Embed `screenshots/screenshot_main_dashboard.png` (Fig 4.7). |
| **Chapter 2: Literature Survey**<br/>• 2.1 Existing System & Limitations<br/>• 2.2 Proposed System Overview<br/>• 2.3 Comparative Study | `literature/chapter2_literature_survey.md` | Embed `tables/table4_6_research_paper_comparison.csv` (Table 4.6). |
| **Chapter 3: System Analysis & Design**<br/>• 3.1 System Requirements<br/>• 3.2 System Architecture<br/>• 3.3 Use Case Description<br/>• 3.4 ER Diagram & Schema<br/>• 3.5 Module Description<br/>• 3.6 API Architecture<br/>• 3.7 End-to-End Data Flow<br/>• 3.8 Security & Ethical Guardrails | `system_design/chapter3_system_design.md` | • `table3_1_hardware_requirements.csv` (Table 3.1)<br/>• `table3_2_software_requirements.csv` (Table 3.2)<br/>• `table3_3_project_modules.csv` (Table 3.3)<br/>• `table3_4_api_endpoints.csv` (Table 3.4)<br/>• `figures/fig3_1_overall_system_architecture.svg` (Fig 3.1)<br/>• `figures/fig3_2_end_to_end_data_flow.svg` (Fig 3.2)<br/>• `figures/fig3_10_database_er_diagram.svg` (Fig 3.10)<br/>• `figures/fig3_11_uml_use_case_diagram.svg` (Fig 3.11) |
| **Chapter 4: Implementation & Testing**<br/>• 4.1 Experimental Setup<br/>• 4.2 Algorithm & Flowcharts<br/>• 4.3 Screenshots & Description<br/>• 4.4 Test Cases and Results<br/>• 4.5 Performance Analysis<br/>• 4.6 Architecture Flowcharts<br/>• 4.7 Advantages of Methodology | `implementation/chapter4_implementation_testing.md` | • `table4_1_dataset_statistics.csv` (Table 4.1)<br/>• `table4_2_training_configurations.csv` (Table 4.2)<br/>• `table4_3_model_performance_comparison.csv` (Table 4.3)<br/>• `table4_4_test_cases_results.csv` (Table 4.4, 109 tests)<br/>• `table4_5_existing_vs_proposed_system.csv` (Table 4.5)<br/>• `figures/fig4_3_model_performance_comparison.svg` (Fig 4.3)<br/>• `figures/fig4_4_inshore_vs_offshore_clutter.svg` (Fig 4.4)<br/>• `figures/fig4_5_grd_slc_domain_compatibility.svg` (Fig 4.5)<br/>• `figures/fig4_6_uncertainty_budget_breakdown.svg` (Fig 4.6)<br/>• `screenshots/sentinel1_real_scene_fused.png` (Fig 4.8)<br/>• `screenshots/lsssd_training_train_batch0.jpg` (Fig 4.9)<br/>• `screenshots/opensar_baseline_val_batch0_pred.jpg` (Fig 4.10) |
| **Chapter 5: Conclusion & Future Scope**<br/>• 5.1 Conclusion<br/>• 5.2 Current System Limitations<br/>• 5.3 Future Scope & Roadmap | `implementation/chapter5_conclusion_future_scope.md` | Ready to copy. Features rigorous limitations and 5-point future scope. |
| **References / Bibliography** | `references/references.md` | Copy references [1] through [17] in standard IEEE format. |

---

## 4. Key Academic Details to Fill in Manually

Before final PDF generation or printing from Word, manually supply the following student group details in `Index.docx` on the Title Page (and wherever group details appear):

1. **Group Number:** e.g., `Group 14` or `Group CI-A-04`.
2. **Student Names & Signatures:**
   - Student 1 (Group Leader): Name, Roll No., G.R. No., Contact Phone, College Email.
   - Student 2: Name, Roll No., G.R. No., Contact Phone, College Email.
   - Student 3: Name, Roll No., G.R. No., Contact Phone, College Email.
3. **Faculty / Guide Signatures:**
   - Internal Guide: Dr. Nilesh B. Korade
   - Head of Department: Dr. Nilesh P. Sable
4. **Final Pagination:** Update the Roman numeral page numbers in the Table of Contents (`SYNOPSIS (i)`, `ACKNOWLEDGEMENT (ii)`, `ABSTRACT (iii)`, `TABLE OF CONTENTS (iv)`, `LIST OF FIGURES (v)`, `LIST OF TABLES (vi)`) to match the final pagination of your Word document.
