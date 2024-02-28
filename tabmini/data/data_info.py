files = [
    # M = 32 - 100 (12 datasets)
    ["analcatdata_aids", "analcatdata_asbestos", "analcatdata_bankruptcy", "analcatdata_creditscore",
     "analcatdata_cyyoung8092", "analcatdata_cyyoung9302", "analcatdata_fraud", "analcatdata_japansolvent",
     "labor", "lupus", "parity5", "postoperative_patient_data"],
    # M = 101 - 200 (10 datasets)
    ["analcatdata_boxing1", "analcatdata_boxing2", "appendicitis", "backache", "corral", "glass2",
     "hepatitis", "molecular_biology_promoters", "mux6", "prnn_crabs"],
    # M = 201 - 300 (9 datasets)
    ["analcatdata_lawsuit", "biomed", "breast_cancer", "heart_h", "heart_statlog", "hungarian", "prnn_synth",
     "sonar", "spect"],
    # M = 301 - 400 (8 datasets)
    ["bupa", "cleve", "colic", "haberman", "heart_c", "horse_colic", "ionosphere", "spectf"],
    # M = 401 - 500 (5 datasets)
    ["clean1", "house_votes_84", "irish", "saheart", "vote"]
]

# Taken from table 8 of the tabPFN paper...

_excluded_1 = [
    "breast-cancer",
    # "colic", # We don't exclude colic as n_features is not consistent with the paper
    "dermatology",
    "sonar",
    "glass",
    "haberman",
    "tae",
    "heart-c",
    "heart-h",
    "heart-statlog",
    "hepatitis",
    "vote",
    "ionosphere",
    "iris",
    "rmftsa_ctoarrivals",
    "chscase_vine2",
    "chatfield_4",
    "boston_corrected",
    "sensory",
    "disclosure_x_noise",
    "autoMpg",
    "kdd_el_nino-small",
    "autoHorse",
    "stock",
    "breastTumor",
    "analcatdata_gsssexsurvey",
    "boston",
    "fishcatch",
    "vinnie",
    "mu284",
    "no2",
    "chscase_geyser1",
    "chscase_census6",
    "chscase_census5",
    "chscase_census4",
    "chscase_census3",
    "chscase_census2",
    "plasma_retinol",
    "visualizing_galaxy",
    "colleges_usnews",
    "wine",
    "flags",
    "hayes-roth",
    "monks-problems-1",
    "monks-problems-2",
    "monks-problems-3",
    "SPECT",
    "SPECTF",
    "grub-damage",
    "synthetic_control",
    "prnn_crabs",
    "analcatdata_lawsuit",
    "irish",
    "analcatdata_broadwaymult",
    "analcatdata_reviewer",
    "backache",
    "prnn_synth",
    "schizo",
    "profb",
    "analcatdata_germangss",
    "biomed",
    "rmftsa_sleepdata",
    "diggle_table_a2",
    "rmftsa_ladata",
    "pwLinear",
    "analcatdata_vineyard",
    "machine_cpu",
    "pharynx",
    "auto_price",
    "servo",
    "analcatdata_wildcat",
    "pm10",
    "wisconsin",
    "autoPrice",
    "meta",
    "analcatdata_apnea3",
    "analcatdata_apnea2",
    "analcatdata_apnea1",
    "disclosure_x_bias",
    "bodyfat",
    "cleveland",
    "triazines",
    "disclosure_x_tampered",
    "cpu",
    "cholesterol",
    "chscase_funds",
    "pbcseq",
    "pbc"
]

# ... and table 9 of the tabPFN paper
_excluded_2 = [
    "disclosure_z",
    "socmob",
    "chscase_whale",
    "water-treatment",
    "lowbwt",
    "arsenic-female-bladder",
    "analcatdata_halloffame",
    "analcatdata_birthday",
    "analcatdata_draft",
    "collins",
    "prnn_fglass",
    "jEdit_4.2_4.3",
    "mc2",
    "mw1",
    "jEdit_4.0_4.2",
    "PopularKids",
    "teachingAssistant",
    "lungcancer_GSE31210",
    "MegaWatt1",
    "PizzaCutter1",
    "PizzaCutter3",
    "CostaMadre1",
    "CastMetal1",
    "KnuggetChase3",
    "PieChart1",
    "PieChart3",
    "parkinsons",
    "planning-relax",
    "qualitative-bankruptcy",
    "sa-heart",
    "seeds",
    "thoracic-surgery",
    "user-knowledge",
    "wholesale-customers",
    "heart-long-beach",
    "robot-failures-lp5",
    "vertebra-column",
    "Smartphone-Based...",
    "breast-cancer-...",
    "LED-display-...",
    "GAMETES_Epistasis...",
    "calendarDOW",
    "corral",
    "mofn-3-7-10",
    "thyroid-new",
    "solar-flare",
    "threeOf9",
    "xd6",
    "tokyo1",
    "parity5_plus_5",
    "cleve",
    "cleveland-nominal",
    "Australian",
    "DiabeticMellitus",
    "conference_attendance",
    "CPMP-2015-...",
    "TuningSVMs",
    "regime_alimentaire",
    "iris-example",
    "Touch2",
    "penguins",
    "titanic"
]

# remove duplicates, strip all strings of leading and trailing whitespace, and sort
_excluded = [s.strip() for s in list(set(_excluded_1 + _excluded_2))]
_excluded.sort()


def is_excluded(name: str) -> bool:
    # if the name in the list ends with "...", we only check if the name starts with the name in the list
    # otherwise we check if the name is in the list
    for e in _excluded:
        if e.endswith("..."):
            if name.startswith(e):
                return True
        elif e == name:
            return True


def is_not_excluded(name: str) -> bool:
    return not is_excluded(name)
