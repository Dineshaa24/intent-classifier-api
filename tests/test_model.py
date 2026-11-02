from app.model import OUT_OF_SCOPE, TwoStageClassifier


def test_predicts_domain_and_intent(model_dir):
    clf = TwoStageClassifier(model_dir)
    p = clf.predict("what's the balance in my savings account")
    assert (p.domain, p.intent) == ("banking", "balance")
    assert 0 < p.confidence <= p.domain_confidence <= 1


def test_intent_models_load_lazily(model_dir):
    clf = TwoStageClassifier(model_dir)
    assert clf.loaded_domains == []
    clf.predict("play some music")
    assert clf.loaded_domains == ["media"]


def test_low_confidence_is_out_of_scope(model_dir):
    clf = TwoStageClassifier(model_dir)
    clf.threshold = 1.01  # nothing can clear this
    assert clf.predict("play some music").intent == OUT_OF_SCOPE
