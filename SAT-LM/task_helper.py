from utils import *
import os

def load_train_test_set(args):
    if os.path.exists("data/{}_{}.json".format(args.task, "train")):
        train_data = read_json("data/{}_{}.json".format(args.task, "train"))
    else:
        train_data = []
    eval_split = args.eval_split
    dev_data = read_json("data/{}_{}.json".format(args.task, eval_split))
    if args.num_train == -1:
        args.num_train = len(train_data)
    if args.num_dev == -1:
        args.num_dev = len(dev_data)
    train_data = train_data[args.slice_train:args.slice_train+args.num_train]
    dev_data = dev_data[args.slice_dev:args.slice_dev+args.num_dev]
    return train_data, dev_data

class TaskHelper:
    style_to_completion_length = {}
    style_to_train_sep = {}

    def __init__(self, style):
        self.style = style

    @classmethod
    def from_taskname(cls, taskname, style):
        if taskname == "gsm":
            return GSMTaskHelper(style)
        elif taskname == "False_j1":
            print("False_j1")
            return ProofD5TaskHelper(style)
        elif taskname == "clutrr":
            return CLUTRRTaskHelper(style)
        elif taskname == "proofd5" or taskname == "pronto":
            return ProofD5TaskHelper(style)
        elif taskname == "arlsat":
            return ArLSATTaskHelper(style)
        elif taskname == "boardmaindp1":
            return Boardmaindp1TaskHelper(style)
        elif taskname == "boardmaindp2":
            return Boardmaindp2TaskHelper(style)
        elif taskname == "boardmaindp3":
            return Boardmaindp3TaskHelper(style)
        # >>> [ExplainEthics Adaptation] BEGIN: register explain_ethics task
        elif taskname == "explainethics":
            # Route the 'explain_ethics' task name to the new ExplainEthicsTaskHelper.
            # This allows run_manual.py to use --task explain_ethics without other changes.
            return ExplainEthicsTaskHelper(style)
        # <<< [ExplainEthics Adaptation] END: register explain_ethics task
        else:
            raise RuntimeError("Not Implemented Yet")

    def prompt_func(self, test_ex, shots):
        raise RuntimeError("Not Implemented Yet")

    def get_completion_length(self):
        return self.style_to_completion_length[self.style]

    def get_train_sep(self):
        return self.style_to_train_sep[self.style]


class GSMTaskHelper(TaskHelper):
    style_to_completion_length = {
        "std": 32,
        "cot": 160,
        "proglm": 320,
        "satlm": 320,
        "satcotsolver": 576,
    }

    style_to_train_sep = {
        "std": "\n\n",
        "cot": "\n\n",
        "proglm": "\n\n\n\n\n\n",
        "satlm": "\n\n\n\n\n\n",
        "satcotsolver": "\n\n\n\n\n\n",
    }

    def prompt_func(self, test_ex, shots):
        if self.style == "std" or self.style == "cot":
            return self.cot_prompt(test_ex, shots)
        elif self.style == "proglm":
            return self.proglm_prompt(test_ex, shots)
        elif self.style == "satlm":
            return self.satlm_prompt(test_ex, shots)
        elif self.style == "satcotsolver":
            return self.satlm_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def cot_prompt(self, test_ex, shots):
        assert len(shots) == 0
        test_example = "Q: {}\nA:".format(test_ex["question"])
        return test_example

    def proglm_prompt(self, test_ex, shots):
        assert len(shots) == 0
        test_example = "Q: {}\n\n# solution in Python:\n\n\n".format(test_ex["question"])
        return test_example

    def satlm_prompt(self, test_ex, shots):
        return self.proglm_prompt(test_ex, shots)


class ProofWriterTaskHelper(TaskHelper):
    style_to_completion_length = {
        "std": 16,
        "cot": 512,
        "satlm": 768,
        "proglm": 512,
    }

    style_to_train_sep = {
        "std": "\n\n\n",
        "cot": "\n\n\n",
        "satlm": "\n\n\n\n\n",
        "proglm": "\n\n\n\n\n",
    }

    def prompt_func(self, test_ex, shots):
        if self.style == "cot":
            return self.cot_prompt(test_ex, shots)
        elif self.style == "std":
            return self.std_prompt(test_ex, shots)
        elif self.style == "satlm":
            return self.satlm_prompt(test_ex, shots)
        elif self.style == "proglm":
            return self.satlm_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def cot_prompt(self, test_ex, shots):
        assert len(shots) == 0
        test_example = (
            'Here are some facts and rules:\n' + 
            '\n'.join(test_ex["context"]) + 
            'Does it imply that the statement "{}" is True?\n'.format(test_ex["question"].rstrip('.')) +
            'Reasoning:\n'
        )
        return test_example

    def std_prompt(self, test_ex, shots):
        assert len(shots) == 0
        test_example = (
            'Here are some facts and rules:\n' + 
            '\n'.join(test_ex["context"]) + 
            'Does it imply that the statement "{}" is True?\n'.format(test_ex["question"].rstrip('.')) +
            'Answer:'
        )
        return test_example

    def satlm_prompt(self, test_ex, shots):
        assert len(shots) == 0
        try:
            test_example = (
                '"""\n' +
                'Here are some facts and rules:\n' + 
                '\n'.join(test_ex["context"]) + 
                '\nQuestion: The statement "{}" is True or False?\n'.format(test_ex["question"].rstrip('.')) +
                '"""\n' + 
                '# solution in Python:\n' +
                'def solution():\n'
            )
        except:
            test_example = (
                '"""\n' +
                'Here are some facts and rules:\n' + 
                '\n'.join(test_ex["theory"]) + 
                '\nQuestion: The statement "{}" is True or False?\n'.format(test_ex["question"].rstrip('.')) +
                '"""\n' + 
                '# solution in Python:\n' +
                'def solution():\n'
            )
        return test_example


class ProofD5TaskHelper(ProofWriterTaskHelper):
    pass


class CLUTRRTaskHelper(TaskHelper):
    style_to_completion_length = {
        "proglm": 512,
        "satlm": 512,
        "satcotsolver": 768,
    }

    style_to_train_sep = {
        "proglm": "\n\n",
        "satlm": "\n\n\n\n\n",
        "satcotsolver": "\n\n\n\n\n",
    }

    def prompt_func(self, test_ex, shots):
        if self.style == "proglm":
            return self.proglm_prompt(test_ex, shots)
        elif self.style == "satlm":
            return self.satlm_prompt(test_ex, shots)
        elif self.style == "satcotsolver":
            return self.satlm_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def proglm_prompt(self, test_ex, shots):
        assert len(shots) == 0
        test_example = ("# Context: {}\n# Question: How is [{}] related to [{}]?\n"
            + "# To answer this question, we write a program to answer the following subquestions:\n").format(
            test_ex["context"], test_ex["query"].replace('(', '').replace(')', '').replace('\'', '').split(',')[1], test_ex["query"].replace('(', '').replace(')', '').replace('\'', '').split(',')[0]
        )
        return test_example

    def satlm_prompt(self, test_ex, shots):
        print("satlm prompt")
        assert len(shots) == 0
        test_example = '"""\n{}\nQuestion: How is [{}] related to [{}]?\n"""\n'.format(
            test_ex["context"], test_ex["query"].replace('(', '').replace(')', '').replace('\'', '').split(',')[1], test_ex["query"].replace('(', '').replace(')', '').replace('\'', '').split(',')[0]
        )
        return test_example


class LongContextMCQAHelper(TaskHelper):
    style_to_completion_length = {
        "std": 16,
        "cot": 512,
        "satlm": 1024,
    }

    style_to_train_sep = {
        "std": "\n\n",
        "cot": "\n\n\n\n",
        "satlm": "\n\n\n\n",
    }

    CHOICE_IDX = ["(A)", "(B)", "(C)", "(D)", "(E)"]
    CODE_HEADER = "### write python code to answer the question"
    CODE_BLOCK_COMMENT = '"""'
    def prompt_func(self, test_ex, shots):
        if self.style == "std":
            return self.std_prompt(test_ex, shots)
        elif self.style == "cot":
            return self.cot_prompt(test_ex, shots)
        elif self.style == "satlm":
            return self.satlm_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def std_prompt(self, test_ex, shots):
        def _single_ex_func(ex, is_train):
            choice_str = "\n".join([self.CHOICE_IDX[i] + " " + x for (i, x) in enumerate(ex["choices"])])
            p_ex = "{}\nQuestion: {}\nChoices:\n{}\nAnswer:".format(ex["context"], ex["question"], choice_str)
            if is_train:
                p_ex = p_ex + " The answer is {}.".format(self.CHOICE_IDX[ex["label"]])
            return p_ex

        showcase_examples = [
            _single_ex_func(s, True) for s in shots
        ]
        test_example = [_single_ex_func(test_ex, False)]
        return self.get_train_sep().join(showcase_examples + test_example)

    def cot_prompt(self, test_ex, shots):
        def _single_ex_func(ex, is_train):
            assert not is_train
            choice_str = "\n".join([self.CHOICE_IDX[i] + " " + x for (i, x) in enumerate(ex["choices"])])
            p_ex = "{}\nQuestion: {}\nChoices:\n{}\nAnswer:".format(ex["context"], ex["question"], choice_str)
            return p_ex

        showcase_examples = [
            _single_ex_func(s, True) for s in shots
        ]
        test_example = [_single_ex_func(test_ex, False)]
        return  self.get_train_sep().join(showcase_examples + test_example)

    def satlm_prompt(self, test_ex, shots):
        def _single_ex_func(ex, is_train):
            assert not is_train
            choice_str = "\n".join([self.CHOICE_IDX[i] + " " + x for (i, x) in enumerate(ex["choices"])])
            p_ex = "{}\n{}\n{}\nQuestion: {}\nChoices:\n{}\n{}\n".format(
                self.CODE_HEADER,
                self.CODE_BLOCK_COMMENT,
                ex["context"], ex["question"], choice_str,
                self.CODE_BLOCK_COMMENT)
            return p_ex

        showcase_examples = [
            _single_ex_func(s, True) for s in shots
        ]
        test_example = [_single_ex_func(test_ex, False)]
        return  self.get_train_sep().join(showcase_examples + test_example)


class ArLSATTaskHelper(LongContextMCQAHelper):
    pass

class BoardgameQATaskHelper(TaskHelper):
    style_to_completion_length = {
        "cot": 512,
        "satlm": 768,
    }

    style_to_train_sep = {
        "cot": "\n\n",
        "satlm": "\n\n\n\n\n",
    }

    def prompt_func(self, test_ex, shots):
        if self.style == "cot":
            return self.cot_prompt(test_ex, shots)
        elif self.style == "satlm":
            return self.satlm_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def cot_prompt(self, test_ex, shots):
        assert len(shots) == 0
        # return test_example
        return f"Q: {test_ex['question']}\nA:"

    def satlm_prompt(self, test_ex, shots):
        assert len(shots) == 0
        test_example = (
            '"""\n'
            f'{test_ex["question"]}\n'
            '"""\n' + 
            '# solution in Python:\n' +
            'def solution():\n'
        )
        return test_example

class Boardmaindp1TaskHelper(BoardgameQATaskHelper):
    style_to_completion_length = {
        "cot": 512,
        "satlm": 768,
    }

class Boardmaindp2TaskHelper(BoardgameQATaskHelper):
    style_to_completion_length = {
        "cot": 512,
        "satlm": 1536,
    }


class Boardmaindp3TaskHelper(BoardgameQATaskHelper):
    style_to_completion_length = {
        "cot": 768,
        "satlm": 1536,
    }

# >>> [ExplainEthics Adaptation] BEGIN: ExplainEthicsTaskHelper
class ExplainEthicsTaskHelper(TaskHelper):
    """
    TaskHelper for the ExplainEthics dataset.

    This helper translates ExplainEthics dataset examples (which have keys
    'context', 'explanation', 'label', 'agents', 'actions', 'patients') into
    prompts for each supported prompting style (satlm, satcotsolver, satnosolver,
    proglm). The manual few-shot examples live in manual_prompts/explain_ethics.jsonline.

    Unlike CLUTRRTaskHelper (which formats kinship pairs), this helper formats
    the ethical context and moral explanation into a Python-style solution block
    that the SAT/logic solver can execute.
    """

    # Token budget per prompting style.
    # satlm/proglm need more tokens because they emit full implies() chains,
    # whereas satcotsolver emits a reasoning paragraph followed by the code.
    style_to_completion_length = {
        "satlm": 512,        # implies() chain is usually ≤ 20 lines
        "satcotsolver": 768, # includes CoT reasoning paragraph + code
        "satnosolver": 512,  # same code format but no SAT solver check
        "proglm": 512,       # Python function with moral predicates
    }

    # Separator between few-shot examples in the concatenated prompt.
    # Four newlines matches the convention used by CLUTRRTaskHelper for satlm.
    style_to_train_sep = {
        "satlm": "\n\n\n\n",
        "satcotsolver": "\n\n\n\n",
        "satnosolver": "\n\n\n\n",
        "proglm": "\n\n\n\n",
    }

    def prompt_func(self, test_ex, shots):
        """Dispatch to the appropriate per-style prompt builder."""
        if self.style == "satlm":
            return self.satlm_prompt(test_ex, shots)
        elif self.style == "satcotsolver":
            return self.satcotsolver_prompt(test_ex, shots)
        elif self.style == "satnosolver":
            return self.satnosolver_prompt(test_ex, shots)
        elif self.style == "proglm":
            return self.proglm_prompt(test_ex, shots)
        else:
            raise RuntimeError(f"ExplainEthicsTaskHelper: unsupported style '{self.style}'")

    def _format_satlm_example(self, ex, is_train):
        """
        Format a single example in the satlm (SAT-LM) prompting style.

        The LLM is asked to produce Python pseudo-code with:
          - action(agent, predicate) = True  for each observed fact
          - implies(antecedent, consequent) = True  for each causal chain step
          - return <norm>(agent)  to state the violated norm

        is_train=True appends the ground-truth code block from the 'output' key
        (used for few-shot examples). is_train=False leaves the block open for
        the LLM to complete (used for the test example).
        """
        # The prompt header mirrors the CLUTRR satlm style: docstring context + code scaffold
        header = (
            '"""\n'
            f'{ex["context"]}\n'
            # f'Question: Does this violate {ex.get("label", "")}?\n'
            'Question: Of these norm violations (violate_care, violate_fairness, violate_loyalty, violate_authority, violate_sanctity, violate_liberty), which one does it most violate?\n'
            '"""\n'
        )
        if "explanation" in ex and ex["explanation"]:
            header += f'# explanation: {ex["explanation"]}\n'
        header += (
            '# solution in Python:\n'
            'def solution():\n'
        )
        if is_train and "output" in ex:
            # Append the reference output for few-shot demonstration
            return header + ex["output"]
        return header  # test example: LLM fills in the rest

    def satlm_prompt(self, test_ex, shots):
        """Build the full satlm prompt: few-shot examples + test query."""
        showcase = [self._format_satlm_example(s, True) for s in shots]
        test = [self._format_satlm_example(test_ex, False)]
        return self.get_train_sep().join(showcase + test)

    def satcotsolver_prompt(self, test_ex, shots):
        """
        satcotsolver style: like satlm but adds a natural-language reasoning
        paragraph before the code so the model explains its moral reasoning.
        The CoT reasoning helps the model make fewer errors in the implies() chain.
        """
        def _fmt(ex, is_train):
            # Same header as satlm; the CoT paragraph is embedded inside the block
            header = (
                '"""\n'
                f'{ex["context"]}\n'
                # f'Question: Does this violate {ex.get("label", "")}?\n'
                'Question: Of these norm violations (violate_care, violate_fairness, violate_loyalty, violate_authority, violate_sanctity, violate_liberty), which one does it most violate?\n'
                '"""\n'
            )
            if "explanation" in ex and ex["explanation"]:
                header += f'# explanation: {ex["explanation"]}\n'
            header += (
                '# Reasoning:\n'
                '# solution in Python:\n'
                'def solution():\n'
            )
            if is_train and "output" in ex:
                return header + ex["output"]
            return header
        showcase = [_fmt(s, True) for s in shots]
        test = [_fmt(test_ex, False)]
        return self.get_train_sep().join(showcase + test)

    def satnosolver_prompt(self, test_ex, shots):
        """
        satnosolver style: identical format to satlm but the pipeline does NOT
        run the SAT solver — the final answer is read directly from the 'return'
        line of the LLM output. Used for ablation experiments.
        """
        # Re-use the satlm formatter since the prompt format is identical
        return self.satlm_prompt(test_ex, shots)

    def proglm_prompt(self, test_ex, shots):
        """
        proglm style: the LLM writes a plain Python function that directly
        returns the norm label as a string, without using implies() chains.
        Simpler but less interpretable than satlm.
        """
        def _fmt(ex, is_train):
            header = (
                f'# Context: {ex["context"]}\n'
            )
            if "explanation" in ex and ex["explanation"]:
                header += f'# Explanation: {ex["explanation"]}\n'
            header += (
                # f'# Question: Does this violate {ex.get("label", "")}?\n'
                '# Question: Of these norm violations (violate_care, violate_fairness, violate_loyalty, violate_authority, violate_sanctity, violate_liberty), which one does it most violate?\n'
                '# To answer this question, we write a program to answer the following subquestions:\n'
                'def solution():\n'
            )
            if is_train and "output" in ex:
                return header + ex["output"]
            return header
        showcase = [_fmt(s, True) for s in shots]
        test = [_fmt(test_ex, False)]
        return self.get_train_sep().join(showcase + test)
# <<< [ExplainEthics Adaptation] END: ExplainEthicsTaskHelper
