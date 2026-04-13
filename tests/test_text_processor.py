"""Test harness for text_processor — imports TextStats from solution.py"""
from solution import TextStats

passed = 0
total = 12

def check(condition, name, detail=""):
    global passed
    if condition:
        passed += 1
        print(f"PASS test_{name}")
    else:
        print(f"FAIL test_{name} ({detail})")

t = TextStats("The cat sat on the mat. The cat liked the mat!")
check(t.word_count() == 12, "word_count", f"got {t.word_count()}")
check(t.unique_words() == 6, "unique_words", f"got {t.unique_words()}")

mc = t.most_common(3)
check(mc[0] == ("the", 4), "most_common_first", f"got {mc}")
check(mc[1][1] == 2, "most_common_second_count", f"got {mc}")
check(t.sentences() == 2, "sentences", f"got {t.sentences()}")

awl = t.avg_word_length()
check(2.5 < awl < 3.5, "avg_word_length", f"got {awl}")
check(t.reading_level() == "medium", "reading_level", f"got {t.reading_level()}")

t2 = TextStats("Hi. Bye. Ok.")
check(t2.sentences() == 3, "short_sentences", f"got {t2.sentences()}")
check(t2.reading_level() == "easy", "easy_reading", f"got {t2.reading_level()}")

t3 = TextStats("")
check(t3.word_count() == 0, "empty_word_count", f"got {t3.word_count()}")
check(t3.unique_words() == 0, "empty_unique", f"got {t3.unique_words()}")
check(t3.most_common() == [], "empty_most_common", f"got {t3.most_common()}")

print(f"RESULT {passed}/{total}")
