from swiftlab.trace import split_thinking, segment, marker_stats, repeated_ngram_ratio


def test_split():
    assert split_thinking("<think>\nabc\n</think>\n\nans") == ("abc", "ans")
    assert split_thinking("abc</think>ans") == ("abc", "ans")
    assert split_thinking("<think>never closed") == ("never closed", "")
    assert split_thinking("plain") == ("", "plain")


def test_segments_and_loops():
    th = "Step 1: compute x. Wait, let me double-check that. " + "Hmm, actually maybe I should reconsider this whole thing. " * 3 + "Step 2: done."
    segs = segment(th)
    kinds = [s.kind for s in segs]
    assert "reverify" in kinds and "loop" in kinds and kinds[0] == "productive"
    st = marker_stats(th)
    assert st["loop_share"] > 0 and st["marker_counts"]
    assert repeated_ngram_ratio("a b c d e f g h i " * 4) > 0.5
