import json
from swiftlab.tasks import Task, TaskBank, verify
from swiftlab.tasks.coding import synthetic_coding_tasks, extract_code
from swiftlab.tasks.knowledge import synthetic_knowledge_tasks
from swiftlab.tasks.math_ import synthetic_math_tasks, extract_number
from swiftlab.tasks.decontam import Decontaminator


def test_coding_reference_passes_and_wrong_fails():
    for t in synthetic_coding_tasks(7):
        assert verify(t, "```python\n" + t.meta["reference_code"] + "\n```").correct
        fn = t.verify["tests"][0]["fn"]
        assert not verify(t, f"```python\ndef {fn}(*a):\n    return None\n```").correct
    assert extract_code("no code here") is None


def test_coding_timeout_and_slow():
    t = synthetic_coding_tasks(1)[0]
    t.verify["timeout"] = 1
    v = verify(t, "```python\nimport time\ntime.sleep(5)\n```")
    assert not v.correct and "timeout" in v.detail


def test_knowledge_and_math():
    for t in synthetic_knowledge_tasks(25):
        assert verify(t, "blah\nFinal answer: " + t.meta["gold_text"]).correct
        assert not verify(t, "Final answer: unknown").correct
    for t in synthetic_math_tasks(12):
        assert verify(t, f"\\boxed{{{t.verify['gold']}}}").correct
        assert not verify(t, f"\\boxed{{{t.verify['gold'] + 1}}}").correct
    assert extract_number("Final answer: 3/4") == 0.75
    assert extract_number("so 1,234 is it") == 1234


def test_decontam_and_split():
    ev = ["What is the greatest common divisor of 12 and 18?"]
    d = Decontaminator(ev, ngram=5)
    assert d.contaminated("Compute the greatest common divisor of 12 and 18 please")
    assert not d.contaminated("Write a function that sums squares")
    bank, dropped = TaskBank.build([], synthetic_per_domain=20, decontam_against=[], seed=1)
    assert len(dropped) == 0
    sp = bank.split({"mine": 0.6, "calib": 0.2, "eval": 0.2})
    ids = [t.id for s in sp.values() for t in s]
    assert len(ids) == len(set(ids)) == len(bank.tasks)
    # split is stable
    sp2 = TaskBank(bank.tasks).split({"mine": 0.6, "calib": 0.2, "eval": 0.2})
    assert [t.id for t in sp["eval"]] == [t.id for t in sp2["eval"]]
