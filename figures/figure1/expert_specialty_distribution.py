"""Expert Specialty Distribution by Dataset - Figure 1f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    from .plot_utils import get_figure_1_colors, apply_medagents_style
except ImportError:
    from plot_utils import get_figure_1_colors, apply_medagents_style

def extract_official_specialty(job_title):
    """
    Extract medical specialty from job title using the official specialties list.
    
    This function performs comprehensive matching including exact matches, partial matches,
    common abbreviations, and specialty-specific variations to accurately categorize
    medical professionals into their respective specialties.
    
    Args:
        job_title (str): The job title to analyze
        
    Returns:
        str: The matched medical specialty or "Other/Unspecified" if no match found
    """
    if pd.isna(job_title):
        return "Unknown"
    
    job_title = str(job_title).lower().strip()
    
    specialty_mappings = {
        'Internal Medicine': ['internal', 'general medicine', 'internist', 'general practitioner', 'gp'],
        'Cardiology': ['cardiac', 'heart', 'cardiovascular', 'cardiologist'],
        'Neurology': ['neuro', 'neurologist', 'brain', 'nervous system'],
        'Emergency Medicine': ['emergency', 'er', 'acute care', 'trauma', 'urgent care'],
        'Infectious Disease': ['infectious', 'infection', 'id', 'communicable disease', 'tropical medicine'],
        'Pediatrics': ['pediatric', 'child', 'infant', 'neonatal', 'adolescent medicine'],
        'Endocrinology': ['endocrine', 'hormone', 'diabetes', 'thyroid', 'metabolic'],
        'Pharmacology': ['pharmacy', 'drug', 'medication', 'pharmaceutical', 'clinical pharmacy'],
        'Pulmonology': ['pulmonary', 'lung', 'respiratory', 'chest medicine', 'breathing'],
        'Gastroenterology': ['gastro', 'gi', 'digestive', 'stomach', 'liver', 'hepatology'],
        'Hematology': ['blood', 'hematologic', 'coagulation', 'bleeding disorders'],
        'Radiology': ['imaging', 'x-ray', 'mri', 'ct', 'ultrasound', 'diagnostic imaging'],
        'Nephrology': ['kidney', 'renal', 'dialysis', 'transplant nephrology'],
        'Psychiatry': ['mental health', 'psychological', 'behavioral health', 'psychiatric'],
        'Oncology': ['cancer', 'tumor', 'chemotherapy', 'radiation oncology', 'hematology-oncology'],
        'Genetics': ['genetic', 'genomic', 'hereditary', 'molecular genetics', 'clinical genetics'],
        'Dermatology': ['skin', 'dermatologic', 'cosmetic', 'aesthetic medicine'],
        'Pathology': ['laboratory', 'lab', 'anatomic pathology', 'forensic pathology'],
        'Geriatrics': ['elderly', 'aging', 'senior', 'gerontology'],
        'Immunology': ['immune', 'autoimmune', 'immunologic'],
        'Rheumatology': ['arthritis', 'joint', 'autoimmune', 'connective tissue'],
        'Urology': ['urologic', 'kidney', 'bladder', 'prostate', 'genitourinary'],
        'Obstetrics and Gynecology': ['obstetrics', 'gynecology', 'ob/gyn', 'obgyn', 'womens health', 'maternal'],
        'Surgery': ['surgical', 'surgeon', 'operative', 'general surgery'],
        'Preventive Medicine': ['prevention', 'public health', 'occupational health', 'community medicine'],
        'Critical Care Medicine': ['critical care', 'intensive care', 'icu', 'ccm', 'critical'],
        'Toxicology': ['poison', 'toxic', 'environmental health', 'occupational toxicology'],
        'Anesthesiology': ['anesthesia', 'pain management', 'perioperative', 'anesthetist'],
        'Neurosurgery': ['brain surgery', 'spine surgery', 'neurosurgeon', 'cranial'],
        'Family Medicine': ['family practice', 'primary care', 'family physician', 'community medicine'],
        'Vascular Medicine': ['vascular', 'angiology', 'blood vessels', 'circulation'],
        'Ophthalmology': ['eye', 'vision', 'retina', 'glaucoma', 'cataract'],
        'Orthopedics': ['bone', 'joint', 'musculoskeletal', 'sports medicine', 'fracture'],
        'Occupational Medicine': ['workplace', 'industrial', 'occupational health', 'work-related'],
        'Sports Medicine': ['athletic', 'exercise', 'fitness', 'sports injury'],
        'Public Health': ['epidemiology', 'population health', 'community health', 'health policy'],
        'Clinical Research': ['research', 'clinical trial', 'biomedical research', 'translational'],
        'Sleep Medicine': ['sleep', 'sleep disorders', 'sleep study', 'insomnia'],
        'Allergy and Immunology': ['allergy', 'allergic', 'asthma', 'immunologic'],
        'Biostatistics': ['statistics', 'data analysis', 'epidemiologic', 'biostatistical'],
        'Medical Ethics': ['ethics', 'bioethics', 'medical law', 'healthcare ethics'],
        'Neonatology': ['newborn', 'nicu', 'premature', 'neonatal intensive care'],
        'Nutrition': ['dietitian', 'nutritionist', 'clinical nutrition', 'dietary'],
        'Epidemiology': ['disease surveillance', 'outbreak', 'population studies', 'epidemiologic'],
        'Rehabilitation Medicine': ['rehabilitation', 'rehab', 'physical medicine', 'disability'],
        'Sexual Health': ['sexual', 'reproductive health', 'std', 'sexual dysfunction'],
        'Reproductive Medicine': ['fertility', 'infertility', 'ivf', 'reproductive endocrinology'],
        'Transplant Medicine': ['transplant', 'organ donation', 'transplantation'],
        'Clinical Pathology': ['clinical lab', 'laboratory medicine', 'diagnostic pathology']
    }
    
    # Check for exact specialty match first
    for specialty in specialty_mappings:
        specialty_lower = specialty.lower()
        if specialty_lower in job_title:
            return specialty
        # Check variants
        for variant in specialty_mappings[specialty]:
            if variant in job_title:
                return specialty
    
    # If no match found but contains medical terms, try partial matching
    common_medical_terms = ['doctor', 'physician', 'md', 'do', 'nurse', 'practitioner', 'specialist', 'consultant']
    if any(term in job_title for term in common_medical_terms):
        for specialty in specialty_mappings:
            specialty_words = specialty.lower().split()
            if any(word in job_title for word in specialty_words if len(word) > 3):
                return specialty
    
    return "Other/Unspecified"

def plot_expert_specialty_distribution(ax, expert_profiles_df, colors, panel_label='F'):
    """Plot stacked bar chart showing expert specialty distribution by dataset"""
    
    if expert_profiles_df.empty:
        # Create sample data if empty
        datasets = ['medqa', 'medmcqa', 'pubmedqa', 'mmlu']
        specialties = ['Internal Medicine', 'Cardiology', 'Emergency Medicine', 'Pediatrics']
        
        sample_data = []
        for dataset in datasets:
            for specialty in specialties:
                sample_data.append({'dataset': dataset, 'job_title': specialty})
        
        expert_profiles_df = pd.DataFrame(sample_data)
    
    # Extract official specialties
    expert_profiles_df['official_specialty'] = expert_profiles_df['job_title'].apply(extract_official_specialty)
    
    # Dataset mapping for cleaner labels
    dataset_mapping = {
        'medbullets': 'MedBullets',
        'medexqa': 'MedExQA',
        'medmcqa': 'MedMCQA',
        'medqa': 'MedQA',
        'medxpertqa-r': 'MedXpertQA-R',
        'medxpertqa-u': 'MedXpertQA-U',
        'mmlu': 'MMLU-Med',
        'mmlu-pro': 'MMLU-Pro-Med',
        'pubmedqa': 'PubMedQA',
    }
    
    datasets = expert_profiles_df['dataset'].unique()
    specialties = expert_profiles_df['official_specialty'].value_counts()
    specialties = specialties[specialties.index != "Other/Unspecified"]
    top_specialties = specialties.head(12).index.tolist()
    
    # Calculate specialty distribution by dataset
    specialty_by_dataset = {}
    for dataset in datasets:
        dataset_df = expert_profiles_df[expert_profiles_df['dataset'] == dataset]
        specialty_counts = dataset_df['official_specialty'].value_counts()
        specialty_by_dataset[dataset] = specialty_counts
    
    # Use Nature Biotechnology journal color palette for professional scientific appearance
    from plot_utils import get_nature_biotechnology_colors
    nature_biotech_colors = get_nature_biotechnology_colors()
    
    # Nature Biotechnology standard color palette for scientific publications
    nature_biotech_palette = [
        nature_biotech_colors['nature_red'],          # Nature signature red
        nature_biotech_colors['nature_blue'],         # Nature blue
        nature_biotech_colors['nature_green'],        # Nature green
        nature_biotech_colors['nature_orange'],       # Nature orange
        nature_biotech_colors['nature_purple'],       # Nature purple
        nature_biotech_colors['nature_brown'],        # Nature brown
        nature_biotech_colors['nature_pink'],         # Nature pink
        nature_biotech_colors['nature_light_blue'],   # Nature light blue
        nature_biotech_colors['nature_light_green'],  # Nature light green
        nature_biotech_colors['nature_light_orange'], # Nature light orange
        nature_biotech_colors['nature_light_purple'], # Nature light purple
        nature_biotech_colors['nature_dark_red'],     # Nature dark red
        nature_biotech_colors['nature_dark_blue'],    # Nature dark blue
        nature_biotech_colors['nature_teal'],         # Nature teal
        nature_biotech_colors['nature_yellow']        # Nature yellow
    ]
    
    specialty_colors = {}
    for i, specialty in enumerate(top_specialties):
        color_idx = i % len(nature_biotech_palette)
        specialty_colors[specialty] = nature_biotech_palette[color_idx]
    specialty_colors['Others'] = nature_biotech_colors['nature_grey']  # Nature journal grey for others category
    
    # Calculate percentages
    dataset_data = {}
    for dataset in datasets:
        dataset_data[dataset] = []
        total_experts = len(expert_profiles_df[(expert_profiles_df['dataset'] == dataset) & 
                                               (expert_profiles_df['official_specialty'] != "Other/Unspecified")])
        
        for specialty in top_specialties:
            specialty_count = specialty_by_dataset[dataset].get(specialty, 0)
            percentage = (specialty_count / total_experts) * 100 if total_experts > 0 else 0
            dataset_data[dataset].append(percentage)
        
        # Calculate 'Others' percentage
        others_count = sum(specialty_by_dataset[dataset].get(spec, 0) 
                          for spec in specialty_by_dataset[dataset].index 
                          if spec not in top_specialties and spec != "Other/Unspecified")
        others_percentage = (others_count / total_experts) * 100 if total_experts > 0 else 0
        dataset_data[dataset].append(others_percentage)
    
    # Plot stacked bars
    x_positions = np.arange(len(datasets))
    bottom_values = np.zeros(len(datasets))
    
    specialty_labels = list(top_specialties) + ['Others']
    
    for i, specialty in enumerate(specialty_labels):
        specialty_percentages = [dataset_data[dataset][i] for dataset in datasets]
        
        bars = ax.bar(x_positions, specialty_percentages, 
                     bottom=bottom_values,
                     color=specialty_colors[specialty], alpha=0.8, 
                     edgecolor='black', linewidth=1.5, 
                     label=specialty)
        
        bottom_values += specialty_percentages
    
    # Styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels([dataset_mapping.get(dataset, dataset.title()) for dataset in datasets], 
                       fontsize=13, rotation=45, ha='right')
    ax.set_ylabel('Percentage of Experts (%)', fontweight='bold', fontsize=16, color='black')
    ax.set_xlabel('Medical Datasets', fontweight='bold', fontsize=16, color='black')
    ax.tick_params(axis='both', labelsize=13, colors='black')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend - narrower with fewer columns
    ax.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.12), 
             fontsize=8, frameon=True, fancybox=True, shadow=True)
    
    # Panel label
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=25, fontweight='bold')
    
    return ax

def main():
    """Example usage when run as standalone script"""
    apply_medagents_style()
    colors = get_figure_1_colors()
    
    # Create sample data
    sample_data = []
    datasets = ['medqa', 'medmcqa', 'pubmedqa', 'mmlu']
    specialties = ['Internal Medicine', 'Cardiology', 'Emergency Medicine', 'Pediatrics', 
                  'Neurology', 'Infectious Disease', 'Endocrinology', 'Pharmacology']
    
    for dataset in datasets:
        for specialty in specialties:
            # Add multiple entries to simulate distribution
            for _ in range(np.random.randint(1, 5)):
                sample_data.append({'dataset': dataset, 'job_title': specialty})
    
    expert_df = pd.DataFrame(sample_data)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_expert_specialty_distribution(ax, expert_df, colors, panel_label='F')
    plt.tight_layout()
    plt.savefig('expert_specialty_distribution_example.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()