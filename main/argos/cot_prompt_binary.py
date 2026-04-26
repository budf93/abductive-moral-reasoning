_MORAL_NORMS = [
    'violate_care', 'violate_fairness', 'violate_loyalty',
    'violate_authority', 'violate_sanctity', 'violate_liberty',
]
NORMS_LIST_STR = ', '.join(_MORAL_NORMS)

FEW_SHOT_BINARY = (
    f"The possible moral norm violations are: {NORMS_LIST_STR}.\n"
    "Context: I crush the frog. "
    # f"Question: Does this violate violate_care? Of these norm violations ({NORMS_LIST_STR}), which one does this action most violate? "
    f"Question: Does this violate violate_care? "
    "Answer: Let's think step by step. "
    "1. Frogs are animals. 2. Crushing causes physical harm to an animal. "
    "3. Causing harm to a living being violates the norm of care. "
    "Therefore the answer is violate_care.\n"
    "Context: I won my match by moving my ball when the other player wasn't looking. "
    # f"Question: Does this violate violate_fairness? Of these norm violations ({NORMS_LIST_STR}), which one does this action most violate? "
    f"Question: Does this violate violate_fairness? "
    "Answer: Let's think step by step. "
    "1. Moving the ball secretly is cheating. 2. Cheating involves deception. "
    "3. Deception in a match breaks the principles of fairness. "
    "Therefore the answer is violate_fairness.\n"
    "Context: I gave my coworker a birthday present. "
    # f"Question: Does this violate violate_loyalty? Of these norm violations ({NORMS_LIST_STR}), which one does this action most violate? "
    f"Question: Does this violate violate_loyalty? "
    "Answer: Let's think step by step. "
    "1. Giving a present is a kind, caring action. 2. No harm is caused to any norm. "
    "Therefore the answer is none.\n"
    "Context: I ignored my boss's direct order. "
    # f"Question: Does this violate violate_authority? Of these norm violations ({NORMS_LIST_STR}), which one does this action most violate? "
    f"Question: Does this violate violate_authority? "
    "Answer: Let's think step by step. "
    "1. Ignoring a boss's order is disobedience. "
    "2. Bosses are traditional authority figures. "
    "3. Disobeying authority violates the norm of authority. "
    "Therefore the answer is violate_authority.\n"
)
