"""One Token battery loading and answer normalization."""

from modeldetective.probes.one_token import battery_probes, load_battery, normalize_answer


def _task(tid: str):  # type: ignore[no-untyped-def]
    _, tasks = load_battery()
    return next(t for t in tasks if t.id == tid)


def test_battery_is_ten_tasks_four_languages() -> None:
    system, tasks = load_battery()
    assert [t.id for t in tasks] == [
        "num100-random",
        "num10-random",
        "num-favorite",
        "letter-random",
        "word-random",
        "color-random",
        "color-favorite",
        "animal-random",
        "city-random",
        "coin-flip",
    ]
    probes = battery_probes()
    assert len(probes) == 40
    assert set(system) == {"en", "ru", "zh", "ar"}
    p, s = probes[0]
    assert p.id == "ot__num100-random__en" and p.max_tokens == 16 and p.parser == "raw"
    assert s.startswith("Answer with a single word or a single number only.")
    assert p.prompt == "Name a random number between 1 and 100."


def test_normalize_numbers() -> None:
    t = _task("num100-random")
    assert normalize_answer("42.", t) == ("42", "valid")
    assert normalize_answer("**7**\n", t) == ("7", "valid")
    assert normalize_answer("١٧", t) == ("17", "valid")  # Arabic-Indic digits
    assert normalize_answer("二十三", t) == ("23", "valid")  # Chinese numerals
    assert normalize_answer("150", t) == ("150", "invalid")
    assert normalize_answer("forty-two", t)[1] == "invalid"
    assert normalize_answer("", t) == ("", "empty")
    assert normalize_answer("I can't pick a random number.", t) == ("", "refusal")


def test_normalize_words_and_coin() -> None:
    assert normalize_answer('"Blue"', _task("color-random")) == ("blue", "valid")
    assert normalize_answer("Heads!", _task("coin-flip")) == ("heads", "valid")
    assert normalize_answer("Орёл", _task("coin-flip")) == ("орёл", "valid")
    assert normalize_answer("maybe", _task("coin-flip")) == ("maybe", "invalid")
    assert normalize_answer("Elephant seal", _task("animal-random")) == ("elephant", "valid")
