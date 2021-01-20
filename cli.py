#!/usr/bin/env python3
# SrcOpsMetrics
# Copyright(C) 2019, 2020 Francesco Murdaca, Dominik Tuchyna
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""This is the CLI for SrcOpsMetrics to create, visualize, use bot knowledge."""

import logging
import os
from typing import List, Optional

import click

_LOGGER = logging.getLogger("aicoe-src-ops-metrics")
logging.basicConfig(level=logging.INFO)

@click.command()
@click.option(
    "--repository",
    "-r", type=str,
    required=True,
    help="Repository that will be used for recommendation training (e.g thoth-station/performance)",
)
def cli(
    repository: Optional[str],
):
    """Command Line Interface for SrcOpsMetrics."""

    scheduler = ReviewerRecommender(repository=project)
    scheduler.initialize_doc2vec()



if __name__ == "__main__":
    cli()
