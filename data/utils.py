PHONEMES = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ',
]

DIPHONES = [(l, r) for l in PHONEMES for r in PHONEMES]


def get_labels_transitions_ids(labels: list):
    return [DIPHONES[(l1, l2)] for l1 in labels for l2 in labels]  # TODO


def cut_phonemes_ids(phonemes_ids):
    for i in range(len(phonemes_ids)-1, -1, -1):
        if phonemes_ids[i] != 0:
            return phonemes_ids[:i]
    return phonemes_ids