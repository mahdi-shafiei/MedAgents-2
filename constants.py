FORMAT_INST = "Please format your response as a JSON object with the following structure:\n{}"

DECOMPOSE_QUERY_SCHEMA = {
    "name": "decomposed_query_response",
    "schema": {
        "type": "object",
        "properties": {
            "Query": {"type": "string"}
        },
        "required": ["Query"],
        "additionalProperties": False
    },
    "strict": True
}

EXPERT_RESPONSE_SCHEMA = {
    "name": "expert_response",
    "schema": {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "answer": {"type": "string"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "justification": {"type": "string"}
        },
        "required": ["thought", "answer", "confidence", "justification"],
        "additionalProperties": False
    },
    "strict": True
}

DIFFICULTY_ASSESSMENT_SCHEMA = {
    "name": "difficulty_assessment",
    "schema": {
        "type": "object",
        "properties": {
            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
            "justification": {"type": "string"}
        },
        "required": ["difficulty", "justification"],
        "additionalProperties": False
    },
    "strict": True
}

DIFFICULTY_TO_PARAMETERS = {
    "easy": {
        "num_experts": 2,
        "max_round": 2,
        "gather_knowledge": False,
    },
    "medium": {
        "num_experts": 3,
        "max_round": 2,
        "gather_knowledge": True,
    },
    "hard": {
        "num_experts": 5,
        "max_round": 3,
        "gather_knowledge": True,
    }
}

MODERATOR_RESPONSE_SCHEMA = {
    "name": "moderator_response",
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "justification": {"type": "string"},
            "limitations": {"type": "string"},
            "isFinal": {"type": "boolean"}
        },
        "required": ["answer", "justification", "limitations", "isFinal"],
        "additionalProperties": False
    },
    "strict": True
}

SEARCH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_medical_knowledge",
        "description": "Search for relevant medical information to help perform the task",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The medical question or search query made by the agent"
                },
                "options": {
                    "type": "object",
                    "properties": {
                        "rewrite": {
                            "type": "boolean",
                            "description": "Whether to rewrite the query for better search results. The query should be rewritten to be more specific and to include more details, or to be more general and to include less details."
                        }
                    },
                    "required": ["rewrite"],
                    "additionalProperties": False
                }
            },
            "required": ["query", "options"],
            "additionalProperties": False
        },
        "strict": True
        }
    }
]

MEDICAL_SPECIALTIES_GPT_SELECTED = [
    "Internal Medicine",
    "Emergency Medicine",
    "Infectious Disease",
    "Pediatrics",
    "Neurology",
    "Endocrinology",
    "Cardiology",
    "Pharmacology",
    "Pulmonology",
    "Gastroenterology",
    "Hematology",
    "Radiology",
    "Nephrology",
    "Psychiatry",
    "Oncology",
    "Genetics",
    "Dermatology",
    "Pathology",
    "Geriatrics",
    "Immunology",
    "Rheumatology",
    "Urology",
    "Obstetrics and Gynecology",
    "Surgery",
    "Preventive Medicine",
    "Critical Care Medicine",
    "Toxicology",
    "Anesthesiology",
    "Neurosurgery",
    "Family Medicine",
    "Vascular Medicine",
    "Ophthalmology",
    "Orthopedics",
    "Occupational Medicine",
    "Sports Medicine",
    "Public Health",
    "Clinical Research",
    "Sleep Medicine",
    "Allergy and Immunology",
    "Biostatistics",
    "Medical Ethics",
    "Neonatology",
    "Nutrition",
    "Epidemiology",
    "Gastroenterology",
    "Rehabilitation Medicine",
    "Sexual Health",
    "Reproductive Medicine",
    "Transplant Medicine",
    "Clinical Pathology"
]