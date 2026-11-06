from ckl.books import (
    BIO_PROMPTS,
    FACT_PROMPTS,
    RELATIONS,
    TEST_QUESTIONS,
    TRAIN_QUESTIONS,
    is_correct,
    key_questions,
    make_books,
    training_examples,
)


def test_books_are_deterministic_and_disjoint():
    a = make_books(people=40)
    b = make_books(people=40)
    assert a == b
    names = {book: {f.subject for f in facts} for book, facts in a.items()}
    assert all(len(n) == 40 for n in names.values())
    assert not (names["A"] & names["B"]) and not (names["A"] & names["C"]) and not (names["B"] & names["C"])
    assert all(len(facts) == 40 * len(RELATIONS) for facts in a.values())


def test_mentor_is_another_person_of_the_same_book():
    for facts in make_books(people=10).values():
        people = {f.subject for f in facts}
        for f in facts:
            if f.relation == "mentor":
                assert f.obj in people and f.obj != f.subject


def test_test_templates_never_appear_in_training():
    books = make_books(people=10)
    train_prompts = {ex.prompt for ex in training_examples(books["A"])}
    for _, q in key_questions(books["A"]):
        assert q not in train_prompts
    for rel in RELATIONS:
        assert not set(TEST_QUESTIONS[rel]) & set(TRAIN_QUESTIONS[rel])
        assert not set(TEST_QUESTIONS[rel]) & set(FACT_PROMPTS + BIO_PROMPTS)


def test_one_key_question_per_fact():
    facts = make_books(people=10)["B"]
    assert len(key_questions(facts)) == len(facts)


def test_is_correct_is_strict():
    f = next(f for f in make_books(people=5)["A"] if f.relation == "birth_year")
    assert is_correct(f, f"Born in {f.obj}.")
    assert not is_correct(f, "Born in 1066.")
    g = next(f for f in make_books(people=5)["A"] if f.relation == "birth_town")
    assert is_correct(g, f"The town was {g.obj.upper()}.")
    assert not is_correct(g, "I do not know.")
