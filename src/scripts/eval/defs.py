all_labels = {
    "BrainStem": 1,
    "Chiasm": 2,
    "OpticNerve_L": 3,
    "OpticNerve_R": 4,
    "Parotid_L": 5,
    "Parotid_R": 6,
    "Mandible": 7
}

all_labels_pddca = {
    "BrainStem": 1,
    "Chiasm": 2,
    "OpticNerve_L": 3,
    "OpticNerve_R": 4,
    "Parotid_L": 5,
    "Parotid_R": 6,
    "Mandible": 7,
    "Submandibular_L": 8,
    "Submandibular_R": 9,
}

all_labels_no_chiasm = {
    "BrainStem": 1,
    "OpticNerve_L": 3,
    "OpticNerve_R": 4,
    "Parotid_L": 5,
    "Parotid_R": 6,
    "Mandible": 7
}

tau = {
    "BrainStem": 2.5,
    "Chiasm": 2.5,
    "OpticNerve_L": 2.5,
    "OpticNerve_R": 2.5,
    "Parotid_L": 2.85,
    "Parotid_R": 2.85,
    "Mandible": 1.01
}

small_organs = {
    "Chiasm": 2,
    "OpticNerve_L": 3,
    "OpticNerve_R": 4
}

small_organs_no_chiasm = {
    "OpticNerve_L": 3,
    "OpticNerve_R": 4
}

big_organs = {
    "BrainStem": 1,
    "Parotid_L": 5,
    "Parotid_R": 6,
    "Mandible": 7
}

one_label = {
    "Foreground": 1
}

two_label = {
    "Big": 1,
    "Small": 2
}

eight_classes = dict(
    labels=all_labels,
    small_organs=small_organs,
    big_organs=big_organs,
    gt_filter='segmentation'
)

seven_classes = dict(
    labels=all_labels_no_chiasm,
    small_organs=small_organs_no_chiasm,
    big_organs=big_organs,
    gt_filter='segmentation'
)

two_classes = dict(
    labels=one_label,
    gt_filter='segma'
)

three_classes = dict(
    labels=two_label,
    gt_filter='small_big_org'
)

def_by_class = dict({
    2: two_classes,
    3: three_classes,
    7: seven_classes,
    8: eight_classes
})

