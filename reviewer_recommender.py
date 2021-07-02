# Copyright (C) 2020 Dominik Tuchyna
#
# This file is part of thoth-station/mi - Meta-information Indicators.
#
# thoth-station/mi - Meta-information Indicators is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# thoth-station/mi - Meta-information Indicators is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with thoth-station/mi - Meta-information Indicators.  If not, see <http://www.gnu.org/licenses/>.
"""Utilities for reviewer recommendation."""

import os
from github import Github

from srcopsmetrics.entities.pull_request import PullRequest
from srcopsmetrics.iterator import KnowledgeAnalysis

from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from typing import Any, List


class ReviewerRecommender:
    """Reviewer Recommender class."""

    def __init__(self):
        """Initialize with repository fullname."""
        self.gh = Github(login_or_token=os.getenv("GITHUB_ACCESS_TOKEN"))

        self.docs = []

        self.data = None
        self.index = None
        self.doc2vec = None

    def initialize_doc2vec(self):
        """Get doc2vec model and its inverted index for PR authors as tuple."""
        authors = set()
        for pr in self.data.values():
            self.docs.append((pr["created_by"], word_tokenize(f'{pr["title"]} {pr["body"]}')))
            authors.add(pr["created_by"])

        author_index = {}
        counter = 0
        for pr in self.data.values():
            author = pr["created_by"]
            if author not in author_index:
                author_index[author] = counter
                counter += 1

        self.index = {idx: author for author, idx in author_index.items()}

        tagged = [TaggedDocument(pr_text, [author_index[author]]) for author, pr_text in self.docs]
        self.doc2vec = Doc2Vec(tagged, vector_size=20, window=5, min_count=1, workers=4, epochs=100)

    def initialize_knowledge(self, repository: str, is_local=False):
        """Add repository knowledge to the data."""
        if is_local:
            pr = PullRequest()
            pr.load_previous_knowledge(is_local=True)
            self.data = pr.stored_entitie
        else:
            gh_repo = self.gh.get_repo(repository)
            with KnowledgeAnalysis(entity=PullRequest(gh_repo), is_local=True) as analysis:
                analysis.run()
                self.data = analysis.entity.stored_entities

    def get_pr_tf_idf(self):
        """Initialize corpus for TF-IDF."""
        corpus = [f'{pr["title"]} {pr["body"]}' for pr in self.data.values()]

        vector_space = self.tf_idf.fit_transform(corpus)
        index_value = {i[1]: i[0] for i in self.tf_idf.vocabulary_.items()}

        fully_indexed = {}
        for row in vector_space:
            fully_indexed.update({index_value[column]: value for (column, value) in zip(row.indices, row.data)})

        raise NotImplementedError

    def recommend_based_on_pr_description(self, description, topn=3) -> List[List[Any]]:
        """Recommend reviewer based on PR desciption (title or/and body).

        Assumption here is, that if PR description is similar to descriptions
        another author often writes, that author might be interested in reviewing the PR.
        """
        test_pr_title = word_tokenize(description)
        similar = self.doc2vec.docvecs.most_similar(positive=[self.doc2vec.infer_vector(test_pr_title)], topn=topn)
        return [[self.index[tag], probability] for tag, probability in similar]

    def recommend_for_author(self, author: str):
        """Recommend reviewer based on author."""
        pass
