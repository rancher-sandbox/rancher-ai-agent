"""Shared types and enums for the application."""

from enum import Enum


class RequestType(Enum):
    """Types of requests the agent can handle."""
    MESSAGE = "message"
    SUMMARY = "summary"
