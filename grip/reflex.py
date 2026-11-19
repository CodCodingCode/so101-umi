"""Squeeze reflex for a position-controlled gripper: keep pushing once the jaw stops on something.

A policy or replayed demo that learned jaw ANGLES (UMI handheld, teleop) commands the angle where the jaw touched the
object. A position servo pushes in proportion to (command - position), so when it reaches that angle it pushes with
~0 force and the object slips. In one of our real replays the jaw sat at 22.7 and 13.3 open with the command within
0.1 of it for seconds.

This sits between the policy and the gripper servo, one call per control tick, in gripper units (0 shut .. 100 open).
When the command has just closed and settled, and the jaw has (nearly) reached it or is stopped short of it without
being shut, it commands `depth` units past where the jaw actually is, and keeps doing so (tracking the jaw, so a soft
object keeps a constant squeeze) until the command opens.
The servo then pushes at its torque limit, so the squeeze force is set by the gripper torque limit (STS3215 register
48), not by the policy.

Until the jaw has stopped while held back from its command (something is between the jaws: `holding`), the push
closes it by at most `max_travel`. If it reaches that without being held back, it was free air: the reflex hands
control back to the command and does not push again until the command moves. Once holding, there is no travel limit,
so a soft object that keeps giving is still held; only an opening command, or the jaw closing shut (the object has
gone), ends the squeeze.

Pure Python, no dependencies: it runs unchanged wherever the servo loop runs.
"""
from collections import deque


class GripReflex:
    def __init__(self, depth=4.0, near=1.5, settle=0.5, settle_ticks=2, still=0.3, still_ticks=3, min_open=3.0,
                 intent_drop=5.0, intent_ticks=75, release=3.0, max_travel=12.0, held_gap=0.5):
        """Defaults are for 25 Hz control and 0..100 gripper units (1 unit ~ 1.1-1.2 deg).

        depth        how far past the jaw to command while squeezing (must exceed the servo's dead band)
        near         the jaw within this of a settled closing command has arrived: squeeze now. Waiting for it to
                     stop dead is too slow: a servo creeps the last units into its target, and a grasp that lifts
                     right after closing (a fast demo) leaves the object behind.
        settle       the command counts as settled (done closing) when it changes less than this per tick ...
        settle_ticks ... for this many ticks (a slow closing ramp the jaw keeps up with never triggers a push)
        still        a jaw that moves less than this per tick counts as stopped (blocked short of its command)
        still_ticks  ... for this many ticks
        min_open     a jaw below this is shut: nothing between the jaws
        intent_drop  the command must have closed by this much ...
        intent_ticks ... within this many ticks for a stop to count as a grasp (pre-opening never triggers it)
        release      the command opening this far above where the jaw stopped lets go at once
        max_travel   before holding is confirmed, the most the reflex closes the jaw past where it stopped (bounds a
                     free-air twitch)
        held_gap     a jaw stopped at least this far short of what the reflex commands is held back by something
        """
        self.depth, self.near, self.settle, self.settle_ticks = depth, near, settle, settle_ticks
        self.still, self.still_ticks = still, still_ticks
        self.min_open, self.intent_drop, self.release, self.max_travel = min_open, intent_drop, release, max_travel
        self.held_gap = held_gap
        self._hist = deque(maxlen=intent_ticks)
        self.reset()

    def reset(self):
        self.state, self.anchor, self.holding = "follow", None, False
        self._prev, self._still_n, self._lock, self._sent = None, 0, None, None
        self._settled_n = 0
        self._hist.clear()

    @property
    def squeezing(self):
        return self.state == "squeeze"

    def _squeeze_cmd(self, cmd, meas):
        out = min(cmd, meas - self.depth)
        if not self.holding:
            out = max(out, self.anchor - self.max_travel)
        self._sent = max(out, 0.0)
        return self._sent

    def _follow(self, lock):
        self.state, self.holding, self._lock, self._sent = "follow", False, lock, None

    def step(self, cmd, meas):
        """cmd: the policy's gripper command, meas: the measured gripper position. Returns the command to send."""
        settled = bool(self._hist) and abs(cmd - self._hist[-1]) < self.settle
        self._settled_n = self._settled_n + 1 if settled else 0
        self._hist.append(cmd)
        stopped = self._prev is not None and abs(meas - self._prev) <= self.still
        self._still_n = self._still_n + 1 if stopped else 0
        self._prev = meas

        if self.state == "squeeze":
            stopped_n = self._still_n >= self.still_ticks
            if stopped_n and meas - self._sent >= self.held_gap:
                self.holding = True                                    # stopped short of the push: something's there
            if cmd > self.anchor + self.release:                       # the policy opens: let go at once
                self._follow(lock=None)
            elif meas < self.min_open:                                 # closed shut: nothing (left) to hold
                self._follow(lock=cmd)
            elif not self.holding and stopped_n and meas - self._sent < self.held_gap:
                self._follow(lock=cmd)                                 # reached the push unopposed: free air
            else:
                return self._squeeze_cmd(cmd, meas)

        closing = max(self._hist) - cmd > self.intent_drop
        unlocked = self._lock is None or abs(cmd - self._lock) > self.release
        arrived = self._settled_n >= self.settle_ticks and (meas - cmd <= self.near or self._still_n >= self.still_ticks)
        if arrived and meas > self.min_open and closing and cmd <= meas + 1.0 and unlocked:
            self.state, self.anchor, self.holding = "squeeze", meas, False
            self._still_n = 0                     # the jaw must stop again under the push to count as held
            return self._squeeze_cmd(cmd, meas)
        return cmd
