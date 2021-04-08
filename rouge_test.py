# from rouge_score import rouge_scorer

# scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
# # scores = scorer.score("The quick brown fox jumps over the lazy dog", "the quick brown dog jumps on the log.")

# scores = scorer.score("the quick brown dog jumps on the log.", "the quick brown dog jumps on log, it is a quick thing, I like it so much")

# print("scores", scores)

from rouge import Rouge

hyp = "the quick brown dog jumps on the log."
ref = "the quick brown dog jumps on log, it is a quick thing, I like it so much"

rouge = Rouge()
scores = rouge.get_scores(ref, hyp)

print("scores", scores)