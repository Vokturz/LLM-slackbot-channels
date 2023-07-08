import time
import asyncio
import nest_asyncio
nest_asyncio.apply()

class SimpleThrottle:
    def __init__(self, coro, delay):
        self.coro = coro
        self.delay = delay
        self.last_call = None
        self.update_task = None
        self.queue_count = 0

    async def _wrapper(self):
        if self.queue_count > 0:
            self.queue_count -= 1

        if self.last_call is not None:
            elapsed_time = time.time() - self.last_call
            if elapsed_time < self.delay:
                await asyncio.sleep(self.delay - elapsed_time)

        await self.coro()
        self.last_call = time.time()
        self.update_task = None

        if self.queue_count > 0:
            await self.call()

    async def call(self):
        if self.update_task is None:
            self.update_task = asyncio.ensure_future(self._wrapper())
        else:
            self.queue_count = min(self.queue_count + 1, 1)

    async def call_and_wait(self):
        if self.update_task is not None:
            await self.update_task
        await self.coro()
