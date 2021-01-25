# Socrates - Reviewer Recommendation

Experimental project on reviewer recommendation based on source code and srcops analysis.

Example usage:

.. code-block::

    >>> from reviewer_recommender import ReviewerRecommender
    >>> r = ReviewerRecommender()
    >>> r.initialize_knowledge(repository="thoth-station/mi-scheduler")
    >>> r.initialize_doc2vec()
    >>> r.recommend_based_on_pr_description("Remove unnecesary multiple initializations")
    [['xtuchyna', 0.9834040403366089], ['khebhut[bot]', 0.9280412197113037]]