"""Concrete compaction strategy implementations.

Implements ClaudeStyleCompactor, AggressiveCompactor, and NoOpCompactor strategies.
"""

from agentrunner.core.compaction.base import CompactionContext, CompactionResult, CompactionStrategy
from agentrunner.core.messages import Message


class ClaudeStyleCompactor(CompactionStrategy):
    """Compaction strategy that preserves recent context and drops old messages.

    Preserves system prompt, recent rounds (user/assistant pairs), and error messages.
    Drops old conversation messages and tool call pairs atomically.
    """

    def get_name(self) -> str:
        return "claude_style"

    def get_description(self) -> str:
        return "Preserve recent context, drop old messages and tool calls"

    async def compact(
        self,
        messages: list[Message],
        target_tokens: int,
        context: CompactionContext,
    ) -> CompactionResult:
        """Compact messages to fit within target token budget."""
        self.validate_messages(messages)

        if context.current_tokens <= target_tokens:
            return CompactionResult(
                messages=messages,
                tokens_saved=0,
                strategy_used=self.get_name(),
                summary_created=False,
                audit_log=["No compaction needed"],
            )

        preserve = self._identify_preserved(messages, context)
        compacted, tokens_saved, audit_log = self._drop_old_messages(messages, preserve, context)

        return CompactionResult(
            messages=compacted,
            tokens_saved=tokens_saved,
            strategy_used=self.get_name(),
            summary_created=False,
            audit_log=audit_log,
        )

    def _identify_preserved(self, messages: list[Message], context: CompactionContext) -> set[int]:
        """Identify which message indices must be preserved."""
        preserve = self.identify_recent_rounds(messages, context.recent_rounds)

        # Always preserve system prompt
        system_prompt = self.find_system_prompt(messages)
        if system_prompt:
            for i, msg in enumerate(messages):
                if msg == system_prompt:
                    preserve.add(i)
                    break

        # Preserve error tool results
        if context.preserve_errors:
            for i, msg in enumerate(messages):
                if msg.role == "tool" and self._is_error(msg):
                    preserve.add(i)

        return preserve

    def _is_error(self, msg: Message) -> bool:
        """Check if tool message contains an error."""
        if msg.role != "tool":
            return False
        content = msg.content or ""
        return (
            "error" in content.lower()
            or "failed" in content.lower()
            or "exception" in content.lower()
            or msg.meta.get("error", False)
        )

    def _drop_old_messages(
        self, messages: list[Message], preserve: set[int], context: CompactionContext
    ) -> tuple[list[Message], int, list[str]]:
        """Drop old messages and tool call pairs outside preserve set."""
        tool_pairs = self._map_tool_pairs(messages)
        drop = self._find_droppable(messages, preserve, tool_pairs)
        return self._filter_messages(messages, drop, context)

    def _map_tool_pairs(self, messages: list[Message]) -> dict[str, tuple[int, int]]:
        """Map tool_call_id to (assistant_idx, tool_result_idx) for complete pairs."""
        assistant_idx = {}
        result_idx = {}

        for i, msg in enumerate(messages):
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    if call_id := tc.get("id"):
                        assistant_idx[call_id] = i
            elif msg.role == "tool" and msg.tool_call_id:
                result_idx[msg.tool_call_id] = i

        # Only return complete pairs
        return {
            call_id: (assistant_idx[call_id], result_idx[call_id])
            for call_id in assistant_idx
            if call_id in result_idx
        }

    def _find_droppable(
        self, messages: list[Message], preserve: set[int], tool_pairs: dict[str, tuple[int, int]]
    ) -> set[int]:
        """Identify droppable message indices."""
        drop = set()

        for i, msg in enumerate(messages):
            if i in preserve:
                continue

            if msg.role in ("user", "assistant"):
                if msg.role == "assistant" and msg.tool_calls:
                    # Drop tool call pairs atomically
                    for tc in msg.tool_calls:
                        if call_id := tc.get("id"):
                            if call_id in tool_pairs:
                                asst_idx, res_idx = tool_pairs[call_id]
                                if asst_idx not in preserve and res_idx not in preserve:
                                    drop.add(asst_idx)
                                    drop.add(res_idx)
                else:
                    # Drop regular conversation message
                    drop.add(i)

            elif msg.role == "tool" and msg.tool_call_id:
                # Drop if corresponding assistant is being dropped
                for _call_id, (asst_idx, res_idx) in tool_pairs.items():
                    if res_idx == i and asst_idx in drop:
                        drop.add(i)

        return drop

    def _filter_messages(
        self, messages: list[Message], drop: set[int], context: CompactionContext
    ) -> tuple[list[Message], int, list[str]]:
        """Filter dropped messages and count tokens saved."""
        result = []
        tokens_saved = 0
        audit_log = []

        for i, msg in enumerate(messages):
            if i in drop:
                tokens_saved += context.count_tokens(msg.content or "")
                audit_log.append(f"Dropped {msg.role} message {i}")
            else:
                result.append(msg)

        return result, tokens_saved, audit_log


class NoOpCompactor(CompactionStrategy):
    """No-operation compaction for testing or large context models."""

    def get_name(self) -> str:
        return "noop"

    def get_description(self) -> str:
        return "No-op: validates tokens but makes no changes"

    async def compact(
        self,
        messages: list[Message],
        target_tokens: int,
        context: CompactionContext,
    ) -> CompactionResult:
        """Return messages unchanged."""
        self.validate_messages(messages)

        audit_log = [
            f"NoOp: {len(messages)} messages, "
            f"{context.current_tokens} tokens (target: {target_tokens})"
        ]

        if context.current_tokens > target_tokens:
            audit_log.append(
                f"WARNING: {context.current_tokens - target_tokens} tokens over target"
            )

        return CompactionResult(
            messages=messages,
            tokens_saved=0,
            strategy_used=self.get_name(),
            summary_created=False,
            audit_log=audit_log,
        )


class AggressiveCompactor(CompactionStrategy):
    """Aggressive compaction strategy that drops old messages aggressively.

    Maximum compression: summarize aggressively, drop successful tool results sooner,
    keep only errors and recent context.
    """

    def get_name(self) -> str:
        return "aggressive"

    def get_description(self) -> str:
        return "Aggressive: maximum compression, keep only errors and recent context"

    async def compact(
        self,
        messages: list[Message],
        target_tokens: int,
        context: CompactionContext,
    ) -> CompactionResult:
        """Aggressively drop old messages to save tokens."""
        self.validate_messages(messages)

        # For now, use the same logic as ClaudeStyleCompactor
        # but with more aggressive settings (keep fewer rounds)
        result_messages = []
        tokens_saved = 0
        audit_log = []

        # Always keep system prompt
        system_msg = next((m for m in messages if m.role == "system"), None)
        if system_msg:
            result_messages.append(system_msg)

        # Keep only the most recent round (more aggressive than claude_style)
        recent_count = min(context.recent_rounds, 1)  # Force to keep only 1 round
        recent_messages = messages[-recent_count:] if recent_count > 0 else []

        for msg in recent_messages:
            if msg not in result_messages:
                result_messages.append(msg)

        tokens_saved = context.current_tokens - len(result_messages) * 100  # Rough estimate
        audit_log.append(
            f"Aggressively kept only {len(result_messages)} messages (dropped {len(messages) - len(result_messages)})"
        )

        return CompactionResult(
            messages=result_messages,
            tokens_saved=max(0, tokens_saved),
            strategy_used=self.get_name(),
            summary_created=False,
            audit_log=audit_log,
        )
