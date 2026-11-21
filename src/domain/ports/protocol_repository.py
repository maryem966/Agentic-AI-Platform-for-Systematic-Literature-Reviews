"""Port for versioned research protocol persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..entities.research_protocol import ResearchProtocol


class ProtocolRepository(ABC):
    @abstractmethod
    async def save(self, protocol: ResearchProtocol) -> ResearchProtocol: ...

    @abstractmethod
    async def find_by_id(self, protocol_id: str) -> ResearchProtocol | None: ...

    @abstractmethod
    async def find_latest_by_research(self, research_id: str) -> ResearchProtocol | None: ...
