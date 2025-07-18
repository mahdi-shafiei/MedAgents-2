"""
Centralized schema and constants definitions for the Medical Agent System.
Breaks out response schemas, RAG/search tool definition, difficulty parameters, and specialty list.
"""
from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from natsort import natsorted

# === MEDICAL SPECIALTIES LIST ===
MEDICAL_SPECIALTIES_GPT_SELECTED: List[str] = [
    "Internal Medicine", "Emergency Medicine", "Infectious Disease", "Pediatrics",
    "Neurology", "Endocrinology", "Cardiology", "Pharmacology", "Pulmonology",
    "Gastroenterology", "Hematology", "Radiology", "Nephrology", "Psychiatry",
    "Oncology", "Genetics", "Dermatology", "Pathology", "Geriatrics",
    "Immunology", "Rheumatology", "Urology", "Obstetrics and Gynecology",
    "Surgery", "Preventive Medicine", "Critical Care Medicine", "Toxicology",
    "Anesthesiology", "Neurosurgery", "Family Medicine", "Vascular Medicine",
    "Ophthalmology", "Orthopedics", "Occupational Medicine", "Sports Medicine",
    "Public Health", "Clinical Research", "Sleep Medicine", "Allergy and Immunology",
    "Biostatistics", "Medical Ethics", "Neonatology", "Nutrition", "Epidemiology",
    "Rehabilitation Medicine", "Sexual Health", "Reproductive Medicine",
    "Transplant Medicine", "Clinical Pathology"
]

MEDICAL_SPECIALTIES_GPT_SELECTED = natsorted(set(MEDICAL_SPECIALTIES_GPT_SELECTED))
