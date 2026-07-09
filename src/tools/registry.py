"""
The only file that imports LangChain into the tools layer -- query_segment.py,
search_guidelines.py, and create_campaign.py have zero framework knowledge and
are unit-tested standalone (see their own test files).

Each tool needs a runtime resource (a DB connection, a FAISS store) that the
LLM must never see or control. build_tools() closes over those resources so
the exposed args_schema only contains the fields the LLM is actually allowed
to populate.
"""
import sqlite3
from typing import List

import yaml
from langchain_community.vectorstores import FAISS
from langchain_core.tools import StructuredTool

from src.config import PROMPTS_PATH
from src.tools.create_campaign import CampaignInput, create_campaign
from src.tools.query_segment import SegmentFilters, query_segment
from src.tools.search_guidelines import GuidelinesQuery, search_guidelines

with open(PROMPTS_PATH) as f:
    _TOOL_META = yaml.safe_load(f)["tools"]


def build_tools(conn: sqlite3.Connection, guidelines_store: FAISS) -> List[StructuredTool]:
    def _query_segment(**kwargs) -> dict:
        return query_segment(SegmentFilters(**kwargs), conn).model_dump()

    def _search_guidelines(**kwargs) -> list:
        chunks = search_guidelines(GuidelinesQuery(**kwargs), guidelines_store)
        return [c.model_dump() for c in chunks]

    def _create_campaign(**kwargs) -> dict:
        return create_campaign(CampaignInput(**kwargs), conn).model_dump()

    return [
        StructuredTool.from_function(
            func=_query_segment,
            name=_TOOL_META["query_segment"]["name"],
            description=_TOOL_META["query_segment"]["description"],
            args_schema=SegmentFilters,
        ),
        StructuredTool.from_function(
            func=_search_guidelines,
            name=_TOOL_META["search_guidelines"]["name"],
            description=_TOOL_META["search_guidelines"]["description"],
            args_schema=GuidelinesQuery,
        ),
        StructuredTool.from_function(
            func=_create_campaign,
            name=_TOOL_META["create_campaign"]["name"],
            description=_TOOL_META["create_campaign"]["description"],
            args_schema=CampaignInput,
        ),
    ]
