"""Bounded admission control; model slots and Torch threads remain independent."""
import asyncio


class AdjustableLimiter:
    def __init__(self, limit):
        self.limit = limit
        self.active = 0
        self.waiting = 0
        self._condition = asyncio.Condition()

    def locked(self):
        return self.active >= self.limit

    async def resize(self, limit):
        async with self._condition:
            self.limit = limit
            self._condition.notify_all()

    async def __aenter__(self):
        async with self._condition:
            self.waiting += 1
            try:
                await self._condition.wait_for(lambda: self.active < self.limit)
                self.active += 1
            finally:
                self.waiting -= 1
        return self

    async def __aexit__(self, *args):
        async with self._condition:
            self.active -= 1
            self._condition.notify_all()


class AdaptivePolicy:
    def __init__(self, capacity, memory_reserve, vram_reserve):
        self.capacity = capacity
        self.memory_reserve = memory_reserve
        self.vram_reserve = vram_reserve
        self.pressure_ticks = self.recovery_ticks = self.cooldown = 0
        self.baseline = None
        self.probe = None
        self._rates = []
        self._probe_rates = []
        self.reason = 'Waiting for resource samples'

    def decide(self, current, snapshot, backlog, completed, rate, can_grow=True):
        pressure = (snapshot.memory_available_mb < self.memory_reserve or
                    (self.vram_reserve and snapshot.gpu_free_mb < self.vram_reserve) or
                    snapshot.cpu_percent >= 90)
        comfortable = (snapshot.memory_available_mb > self.memory_reserve * 1.15 and
                       (not self.vram_reserve or snapshot.gpu_free_mb > self.vram_reserve * 1.15) and
                       snapshot.cpu_percent < 75)
        self.pressure_ticks = self.pressure_ticks + 1 if pressure else 0
        self.recovery_ticks = self.recovery_ticks + 1 if comfortable and backlog and can_grow else 0
        self.cooldown = max(0, self.cooldown - 1)
        if self.pressure_ticks >= 2:
            self.probe = self.baseline = None
            self._rates.clear()
            self._probe_rates.clear()
            self.recovery_ticks = self.pressure_ticks = 0
            self.cooldown = 3
            self.reason = 'Resource pressure; reduce new admissions'
            return max(1, current - 1)
        if self.probe is not None and completed >= current * 4:
            self._probe_rates.append(rate)
            if len(self._probe_rates) >= 2:
                measured = sum(self._probe_rates) / len(self._probe_rates)
                if measured < self.probe * 1.05:
                    self.reason = 'Throughput probe did not improve; revert'
                    self.baseline, self.probe = self.probe, None
                    self._rates = [self.baseline]
                    self.cooldown = 60
                    self.recovery_ticks = 0
                    return max(1, current - 1)
                self.baseline, self.probe = measured, None
                self._rates = [measured]
                self.reason = 'Throughput probe improved'
        elif self.probe is None and completed >= current * 4:
            self._rates.append(rate)
            self._rates = self._rates[-3:]
            self.baseline = sum(self._rates) / len(self._rates)
        if self.recovery_ticks >= 3 and not self.cooldown and self.probe is None and current < self.capacity:
            # A measured busy baseline is required; idle service never expands.
            if self.baseline is not None and completed >= current * 4:
                self.probe = self.baseline
                self._probe_rates.clear()
                self.recovery_ticks = 0
                self.reason = 'Resources recovered; probe one extra slot'
                return current + 1
        return current
